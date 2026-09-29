from __future__ import annotations

import numpy as np
import torch

from dataset.loader import TorchDataLoader, TorchDataset

from .io import load_model
from .metric import IouMetric
from .pointnet2 import PointNet2


def _stable_softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.maximum(exp_logits.sum(axis=1, keepdims=True), 1e-12)


def _hierarchical_ensemble(logits, hierarchy, chunk_size=8192):
    num_points = logits[0].shape[0]
    leaf_count = hierarchy.classes_num[-1]
    paths = []
    for start in range(0, num_points, chunk_size):
        end = min(start + chunk_size, num_points)
        score = np.zeros((end - start, leaf_count), dtype=np.float32)
        for level_logits, matrix in zip(logits, hierarchy.hierarchical_matrices):
            score += np.dot(_stable_softmax(level_logits[start:end]), matrix).astype(np.float32)
        leaf_labels = np.argmax(score, axis=1)
        paths.append(hierarchy.all_valid_h_label[leaf_labels])
    return np.concatenate(paths, axis=0)


def _count_consistent_paths(hierarchy, labels):
    labels = np.ascontiguousarray(labels)
    valid_paths = np.ascontiguousarray(hierarchy.all_valid_h_label).astype(labels.dtype, copy=False)
    path_dtype = np.dtype((np.void, labels.dtype.itemsize * labels.shape[1]))
    valid_view = valid_paths.view(path_dtype).reshape(-1)
    label_view = labels.view(path_dtype).reshape(-1)
    return int(np.isin(label_view, valid_view).sum())


def _update_metrics(paths, targets, metrics, correct):
    for level_idx, metric in enumerate(metrics):
        pred = paths[:, level_idx]
        target = targets[:, level_idx]
        metric.update(pred, target)
        correct[level_idx] += int((pred == target).sum())


def _print_metrics(io, name, metrics, correct, total_points, consistent):
    io.cprint(name)
    mean_ious = []
    for level_idx, metric in enumerate(metrics):
        mean_iou = float(metric.avg_iou())
        mean_ious.append(mean_iou)
        io.cprint("level {}: OA={:.6f}, mIoU={:.6f}".format(
            level_idx + 1,
            correct[level_idx] / float(max(total_points, 1)),
            mean_iou,
        ))
    io.cprint("mean mIoU: {:.6f}".format(float(np.mean(mean_ious))))
    io.cprint("consistency: {:.6f}".format(consistent / float(max(total_points, 1))))


def test(args, io, cfg, hierarchy):
    device = torch.device("cuda" if args.cuda else "cpu")
    dataset = TorchDataset("TEST_SET", params=cfg.DATASET, is_training=False)
    loader = TorchDataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=int(cfg.TRAIN.NUM_WORKERS),
        shuffle=False,
    )
    io.cprint("{} test objects: {}".format(cfg.DATASET.DATA.DATASET_NAME, len(dataset)))

    model = PointNet2(cfg, args).to(device)
    model = load_model(args, cfg, model)
    model.eval()

    num_levels = len(cfg.DATASET.DATA.LABEL_NUMBER)
    mt_metrics = [IouMetric(list(range(count))) for count in cfg.DATASET.DATA.LABEL_NUMBER]
    he_metrics = [IouMetric(list(range(count))) for count in cfg.DATASET.DATA.LABEL_NUMBER]
    mt_correct = [0] * num_levels
    he_correct = [0] * num_levels
    mt_consistent = 0
    he_consistent = 0
    total_points = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            points, labels, colors, _ = batch
            inputs = torch.as_tensor(np.concatenate([points, colors], axis=-1), dtype=torch.float32, device=device)
            outputs, _ = model(inputs.permute(0, 2, 1))
            labels_flat = labels.reshape(-1, labels.shape[-1])
            logits = [
                output.permute(0, 2, 1).contiguous().cpu().numpy().reshape(-1, output.shape[1])
                for output in outputs
            ]
            mt_paths = np.stack([np.argmax(level_logits, axis=1) for level_logits in logits], axis=1)
            he_paths = _hierarchical_ensemble(logits, hierarchy)

            total_points += labels_flat.shape[0]
            _update_metrics(mt_paths, labels_flat, mt_metrics, mt_correct)
            _update_metrics(he_paths, labels_flat, he_metrics, he_correct)
            mt_consistent += _count_consistent_paths(hierarchy, mt_paths)
            he_consistent += _count_consistent_paths(hierarchy, he_paths)
            if batch_idx and batch_idx % 20 == 0:
                io.cprint("eval batch: {}/{}".format(batch_idx, len(loader)))

    _print_metrics(io, "{} MT".format(args.exp_name), mt_metrics, mt_correct, total_points, mt_consistent)
    _print_metrics(io, "{} HE".format(args.exp_name), he_metrics, he_correct, total_points, he_consistent)
