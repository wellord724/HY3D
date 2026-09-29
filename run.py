from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from dataset.loader import TorchDataLoader, TorchDataset
from dataset.reader import read_h_matrix_file_list
from .config import load_config
from .eval import test
from .io import IOStream, save_model
from .loss import (
    HierarchicalCrossEntropyLoss,
    LogitsConsistencyLoss,
    PCTLoss,
    PrototypeBank,
)
from .metric import IouMetric
from .pointnet2 import PointNet2


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent

# Ablation presets. Each entry overrides the corresponding TRAIN fields so that
# a single code path covers all variants.
MODEL_PRESETS = {
    "baseline": {"SEG_HEAD_TYPE": "euclidean_raw", "PCT_ENABLE": False, "CONSISTENCY_LOSS": False},
    "m0":       {"SEG_HEAD_TYPE": "euclidean_hyp", "PCT_ENABLE": False, "CONSISTENCY_LOSS": False},
    "m1":       {"SEG_HEAD_TYPE": "mobius",        "PCT_ENABLE": False, "CONSISTENCY_LOSS": False},
    "m2":       {"SEG_HEAD_TYPE": "euclidean_hyp", "PCT_ENABLE": True,  "CONSISTENCY_LOSS": False},
    "hy3d":     {"SEG_HEAD_TYPE": "mobius",        "PCT_ENABLE": True,  "CONSISTENCY_LOSS": False},
}

DEFAULT_EXP_NAMES = {
    "baseline": "BASELINE",
    "m0": "M0",
    "m1": "M1",
    "m2": "M2",  # Deprecated
    "hy3d": "HY3D",
}


def _apply_model_preset(cfg, model):
    preset = MODEL_PRESETS[model]
    for key, value in preset.items():
        cfg.TRAIN[key] = value


def _init_experiment(args):
    checkpoint_dir = REPO_ROOT / "checkpoints" / args.exp_name
    model_dir = checkpoint_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = checkpoint_dir / "{}_source".format(PACKAGE_DIR.name)
    shutil.copytree(
        PACKAGE_DIR,
        source_snapshot,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )


def _get_label_weights(dataset):
    if hasattr(dataset, "label_weights"):
        return dataset.label_weights
    if hasattr(dataset, "data_sampler") and hasattr(dataset.data_sampler, "label_weights"):
        return dataset.data_sampler.label_weights
    raise AttributeError("Dataset does not expose label_weights.")


def _resolve_momentum(epoch, schedule, default):
    for end_epoch, momentum in schedule:
        if int(end_epoch) < 0 or epoch <= int(end_epoch):
            return float(momentum)
    return float(default)


def _ramp_weight(epoch, start_epoch, ramp_epochs, target):
    if epoch < int(start_epoch):
        return 0.0
    progress = min(1.0, (epoch - int(start_epoch) + 1) / float(max(1, int(ramp_epochs))))
    return float(target) * progress


def _balanced_leaf_indices(labels, cfg):
    max_points = int(cfg.MAX_POINTS)
    min_points = int(cfg.MIN_POINTS_PER_PRESENT_LEAF)
    max_per_class = int(cfg.MAX_POINTS_PER_PRESENT_LEAF)
    total = labels.numel()
    if max_points <= 0 or total <= max_points:
        return torch.arange(total, device=labels.device)

    selected = []
    for label in torch.unique(labels):
        class_indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        take = min(max_per_class, int(class_indices.numel()))
        take = max(min_points if class_indices.numel() >= min_points else int(class_indices.numel()), take)
        perm = torch.randperm(class_indices.numel(), device=labels.device)[:take]
        selected.append(class_indices[perm])
    indices = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long, device=labels.device)
    if indices.numel() > max_points:
        indices = indices[torch.randperm(indices.numel(), device=labels.device)[:max_points]]
    elif indices.numel() < max_points:
        available = torch.ones(total, dtype=torch.bool, device=labels.device)
        available[indices] = False
        remaining = torch.nonzero(available, as_tuple=False).flatten()
        fill = min(max_points - indices.numel(), remaining.numel())
        if fill > 0:
            indices = torch.cat([indices, remaining[torch.randperm(remaining.numel(), device=labels.device)[:fill]]])
    return indices


def _taxonomy_depth(entry):
    key = entry.get("key", "")
    if key == "__zero__":
        return 0
    try:
        return int(key.split("|", 2)[1])
    except (IndexError, ValueError):
        return int(entry.get("level_idx", 0))


def _build_depth_deltas(taxonomy_file, parent_maps):
    if not taxonomy_file or not os.path.isfile(taxonomy_file):
        # Datasets without taxonomy metadata (e.g. Campus3D) fall back to unit
        # depth deltas, which makes the radius margin uniform across edges.
        return [np.ones_like(parent_map, dtype=np.float32) for parent_map in parent_maps]
    with open(taxonomy_file, "r", encoding="utf-8") as handle:
        taxonomy = json.load(handle)
    level_depths = [np.asarray([_taxonomy_depth(entry) for entry in level], dtype=np.float32) for level in taxonomy["levels"]]
    depth_deltas = []
    for level_idx, parent_map in enumerate(parent_maps, start=1):
        child_depth = level_depths[level_idx]
        parent_depth = level_depths[level_idx - 1][parent_map]
        depth_deltas.append(np.maximum(child_depth - parent_depth, 0.0))
    return depth_deltas


def _validate_config(cfg, hierarchy):
    classes = [int(item) for item in hierarchy.classes_num]
    if list(cfg.DATASET.DATA.LABEL_NUMBER) != classes:
        raise ValueError("LABEL_NUMBER does not match hierarchy matrices.")
    dataset_name = str(cfg.DATASET.DATA.DATASET_NAME).lower()
    if dataset_name not in ("partnext", "campus3d"):
        raise ValueError("Unsupported DATASET_NAME: {} (expected partnext or campus3d)".format(dataset_name))
    if len(cfg.TRAIN.LOSS_WEIGHTS) != len(classes):
        raise ValueError("LOSS_WEIGHTS length must match hierarchy levels.")


def _count_consistent_paths(hierarchy, labels):
    labels = np.ascontiguousarray(labels)
    valid_paths = np.ascontiguousarray(hierarchy.all_valid_h_label).astype(labels.dtype, copy=False)
    path_dtype = np.dtype((np.void, labels.dtype.itemsize * labels.shape[1]))
    valid_view = valid_paths.view(path_dtype).reshape(-1)
    label_view = labels.view(path_dtype).reshape(-1)
    return int(np.isin(label_view, valid_view).sum())


def _validate(model, loader, hierarchy, device, io, epoch):
    num_levels = len(model.num_class)
    metrics = [IouMetric(list(range(count))) for count in model.num_class]
    correct = [0] * num_levels
    total_points = 0
    consistent = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            points, labels, colors, _ = batch
            inputs = torch.as_tensor(np.concatenate([points, colors], axis=-1), dtype=torch.float32, device=device)
            outputs, _ = model(inputs.permute(0, 2, 1))
            labels_flat = labels.reshape(-1, num_levels)
            paths = []
            for level_idx, output in enumerate(outputs):
                pred = output.argmax(dim=1).cpu().numpy().reshape(-1)
                target = labels_flat[:, level_idx]
                metrics[level_idx].update(pred, target)
                correct[level_idx] += int((pred == target).sum())
                paths.append(pred)
            paths = np.stack(paths, axis=1)
            total_points += labels_flat.shape[0]
            consistent += _count_consistent_paths(hierarchy, paths)
    level_iou = [float(metric.avg_iou()) for metric in metrics]
    level_acc = [value / float(max(total_points, 1)) for value in correct]
    mean_iou = float(np.mean(level_iou))
    mean_acc = float(np.mean(level_acc))
    io.cprint("validation epoch {}: mean_iou={:.6f}, mean_acc={:.6f}, consistency={:.6f}".format(
        epoch, mean_iou, mean_acc, consistent / float(max(total_points, 1))))
    return mean_iou, mean_acc


def train(args, io, cfg, hierarchy):
    device = torch.device("cuda" if args.cuda else "cpu")
    num_levels = len(hierarchy.classes_num)
    model = PointNet2(cfg, args).to(device)
    train_dataset = TorchDataset("TRAIN_SET", params=cfg.DATASET, is_training=True)
    val_dataset = TorchDataset("VALIDATION_SET", params=cfg.DATASET, is_training=False)
    train_loader = TorchDataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=int(cfg.TRAIN.NUM_WORKERS),
        shuffle=True,
    )
    val_loader = TorchDataLoader(
        dataset=val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=int(cfg.TRAIN.NUM_WORKERS),
        shuffle=False,
    )

    segmentation_loss = HierarchicalCrossEntropyLoss(_get_label_weights(train_dataset), device)

    logits_consistency = None
    if bool(cfg.TRAIN.CONSISTENCY_LOSS):
        io.cprint("use consistency loss")
        matrices = [hierarchy[level + 1, level] for level in range(num_levels - 1)]
        logits_consistency = LogitsConsistencyLoss(matrices, cfg.TRAIN.CONSISTENCY_WEIGHTS, device)

    pct_enabled = bool(cfg.TRAIN.PCT_ENABLE)
    bank = None
    pct_loss = None
    if pct_enabled:
        parent_maps = [np.argmax(hierarchy[level - 1, level], axis=1) for level in range(1, num_levels)]
        depth_deltas = _build_depth_deltas(getattr(cfg.DATASET.DATA, "TAXONOMY_FILE", None), parent_maps)
        bank = PrototypeBank(
            hierarchy.classes_num,
            sz_embed=cfg.TRAIN.SZ_EMBED,
            hyp_c=cfg.TRAIN.HYP_C,
            momentum=cfg.TRAIN.PCT_MOMENTUM,
        ).to(device)
        pct_loss = PCTLoss(
            hierarchy.classes_num,
            parent_maps,
            depth_deltas,
            hyp_c=cfg.TRAIN.HYP_C,
            temp=cfg.TRAIN.PCT_TEMP,
            tri_margin=cfg.TRAIN.PCT_TRI_MARGIN,
            radius_margin=cfg.TRAIN.PCT_RADIUS_MARGIN,
            skip_repeated_edges=cfg.TRAIN.PCT_RADIUS_POLICY.SKIP_REPEATED_NODE_EDGE,
            scale_margin_by_depth=cfg.TRAIN.PCT_RADIUS_POLICY.SCALE_MARGIN_BY_REAL_DEPTH_DELTA,
        ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-4)
    if str(cfg.TRAIN.SCHEDULER).lower() == "cos":
        scheduler = CosineAnnealingLR(optimizer, cfg.TRAIN.MAX_EPOCH, eta_min=1e-3)
    else:
        scheduler = StepLR(optimizer, 20, 0.5)

    best_iou = None
    best_acc = None

    bank_start = int(cfg.TRAIN.PCT_START_EPOCH) if pct_enabled else 0
    pct_start = bank_start + int(cfg.TRAIN.PCT_BANK_WARMUP_EPOCH) if pct_enabled else 0
    curriculum = cfg.TRAIN.PCT_CURRICULUM if pct_enabled else None
    ramp_epochs = curriculum.RAMP_EPOCHS if pct_enabled else 0

    for epoch in range(int(cfg.TRAIN.MAX_EPOCH)):
        model.train()
        if pct_enabled:
            bank.set_momentum(_resolve_momentum(epoch, cfg.TRAIN.PCT_MOMENTUM_SCHEDULE, cfg.TRAIN.PCT_MOMENTUM))
            cons_weight = _ramp_weight(epoch, curriculum.CONS_START_EPOCH, ramp_epochs, cfg.TRAIN.PCT_CONS_WEIGHT)
            tri_weight = _ramp_weight(epoch, curriculum.TRI_START_EPOCH, ramp_epochs, cfg.TRAIN.PCT_TRI_WEIGHT)
            radius_weight = _ramp_weight(epoch, curriculum.RADIUS_START_EPOCH, ramp_epochs, cfg.TRAIN.PCT_RADIUS_WEIGHT)
            if epoch < pct_start:
                cons_weight = tri_weight = radius_weight = 0.0
            io.cprint("epoch {}: bank_momentum={:.3f}, pct_weights=({:.4f},{:.4f},{:.4f})".format(
                epoch, bank.momentum, cons_weight, tri_weight, radius_weight))
        else:
            cons_weight = tri_weight = radius_weight = 0.0

        loss_sum = 0.0
        batches = 0
        for batch_idx, batch in enumerate(train_loader):
            points, labels, colors, _ = batch
            inputs = torch.as_tensor(np.concatenate([points, colors], axis=-1), dtype=torch.float32, device=device)
            labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
            optimizer.zero_grad()
            outputs, embeddings = model(inputs.permute(0, 2, 1))

            seg_loss = outputs[0].new_zeros(())
            for level_idx, output in enumerate(outputs):
                seg_loss = seg_loss + segmentation_loss(
                    output.permute(0, 2, 1).contiguous(),
                    labels_tensor[..., level_idx],
                    level_idx,
                ) * cfg.TRAIN.LOSS_WEIGHTS[level_idx]
            total_loss = seg_loss
            if logits_consistency is not None:
                total_loss = total_loss + logits_consistency(outputs)

            pct_stats = None
            if pct_enabled and epoch >= pct_start and (cons_weight > 0 or tri_weight > 0 or radius_weight > 0):
                embedding_bn = [embedding.permute(0, 2, 1).contiguous() for embedding in embeddings]
                flat_labels = labels_tensor.reshape(-1, num_levels)
                indices = _balanced_leaf_indices(flat_labels[:, -1], cfg.TRAIN.PCT_SAMPLING)
                sampled_features = [item.reshape(-1, item.shape[-1])[indices] for item in embedding_bn]
                pct_value, pct_stats = pct_loss(
                    sampled_features,
                    flat_labels[indices],
                    bank,
                    cons_weight,
                    tri_weight,
                    radius_weight,
                )
                total_loss = total_loss + pct_value

            total_loss.backward()
            optimizer.step()

            if pct_enabled and epoch >= bank_start:
                embedding_bn = [embedding.permute(0, 2, 1).contiguous() for embedding in embeddings]
                for level_idx, embedding in enumerate(embedding_bn):
                    bank.update_level_object_balanced(level_idx, embedding, labels_tensor[..., level_idx])

            loss_sum += float(total_loss.item())
            batches += 1
            if batch_idx and batch_idx % 200 == 0:
                pct_total = float(pct_stats["total"].item()) if pct_stats else 0.0
                io.cprint("batch {}: total={:.4f}, seg={:.4f}, pct={:.4f}".format(
                    batch_idx, total_loss.item(), seg_loss.item(), pct_total))

        scheduler.step()
        io.cprint("train epoch {}: mean_loss={:.6f}".format(epoch, loss_sum / float(max(batches, 1))))

        if epoch % 3 == 0:
            mean_iou, mean_acc = _validate(model, val_loader, hierarchy, device, io, epoch)
            better = best_iou is None or mean_iou > best_iou + 1e-4
            better = better or (best_iou is not None and abs(mean_iou - best_iou) <= 1e-4 and mean_acc > best_acc)
            if better:
                best_iou, best_acc = mean_iou, mean_acc
                save_model(model, cfg, args, "model_best")
                io.cprint("saved model_best at epoch {}".format(epoch))

    save_model(model, cfg, args, "model_final")
    if pct_enabled:
        bank_path = REPO_ROOT / "checkpoints" / args.exp_name / "prototype_bank_final.pt"
        torch.save(bank.export_state(), str(bank_path))
        io.cprint("saved prototype bank: {}".format(bank_path))


def main():
    parser = argparse.ArgumentParser(description="Unified PartNeXt / Campus3D ablation experiment")
    parser.add_argument("--model", default="hy3d", choices=list(MODEL_PRESETS.keys()),
                        help="Ablation variant: baseline | m0 | m1 | m2 | hy3d")
    parser.add_argument("--dataset", default="partnext", choices=["partnext", "campus3d"],
                        help="Dataset: partnext | campus3d (selects the corresponding config set)")
    parser.add_argument("--exp_name", default=None,
                        help="Experiment name; defaults to a model-specific name")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--mc_level", type=int, default=-1)
    parser.add_argument("--config_dir", default=None,
                        help="Custom config directory; defaults to configs/<dataset>")
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = DEFAULT_EXP_NAMES[args.model]

    if args.mc_level != -1:
        raise ValueError("Joint five-level training requires mc_level=-1.")

    os.chdir(str(REPO_ROOT))
    if args.config_dir is None:
        args.config_dir = str(PACKAGE_DIR / "configs" / args.dataset)
    args.config_dir = str(Path(args.config_dir).resolve())
    cfg = load_config(args.config_dir)
    _apply_model_preset(cfg, args.model)

    hierarchy = read_h_matrix_file_list(
        cfg.DATASET.DATA.H_MATRIX_LIST_FILE,
        sort_by_class_num=cfg.DATASET.DATA.SORT_H_MATRIX_BY_CLASS_NUM,
    )
    _validate_config(cfg, hierarchy)
    _init_experiment(args)

    # Resolve checkpoint path for evaluation from the current exp_name.
    cfg.TRAIN.PRETRAINED_MODEL_PATH = str(
        REPO_ROOT / "checkpoints" / args.exp_name / "models" / (cfg.EVAL.MODEL_NAME + ".t7")
    )

    args.cuda = torch.cuda.is_available()
    random.seed(cfg.DEVICES.SEED)
    np.random.seed(cfg.DEVICES.SEED)
    torch.manual_seed(cfg.DEVICES.SEED)
    if args.cuda:
        torch.cuda.set_device(cfg.DEVICES.GPU_ID[0])
        torch.cuda.manual_seed_all(cfg.DEVICES.SEED)

    log_name = "evalrun.log" if args.eval else "run.log"
    io = IOStream(str(REPO_ROOT / "checkpoints" / args.exp_name / log_name))
    io.cprint("Using {}".format("GPU" if args.cuda else "CPU"))
    io.cprint("dataset={}, model={}, exp_name={}, seg_head={}, pct={}, consistency={}".format(
        args.dataset, args.model, args.exp_name, cfg.TRAIN.SEG_HEAD_TYPE,
        cfg.TRAIN.PCT_ENABLE, cfg.TRAIN.CONSISTENCY_LOSS))
    if args.eval:
        test(args, io, cfg, hierarchy)
    else:
        train(args, io, cfg, hierarchy)


if __name__ == "__main__":
    main()
