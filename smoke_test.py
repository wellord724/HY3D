from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from .config import load_config
from .loss import HierarchicalCrossEntropyLoss, PCTLoss, PrototypeBank
from .pointnet2 import PointNet2
from .hyptorch import pmath


CONFIG_PATH = Path(__file__).resolve().parent / "configs"


def _finite_gradient(parameter):
    return parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def _make_config(seg_head_type, pct_enable, dataset="partnext"):
    cfg = load_config(str(CONFIG_PATH / dataset))
    cfg.TRAIN.SEG_HEAD_TYPE = seg_head_type
    cfg.TRAIN.PCT_ENABLE = pct_enable
    return cfg


def test_model_shapes(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2(cfg, SimpleNamespace(mc_level=-1)).to(device).eval()
    logits, features = model(torch.randn(1, 6, 1024, device=device))
    expected = [(1, count, 1024) for count in cfg.DATASET.DATA.LABEL_NUMBER]
    actual = [tuple(item.shape) for item in logits]
    assert actual == expected, (actual, expected)
    assert all(torch.isfinite(item).all() for item in logits)
    # euclidean_raw returns None hyp features; others return Poincare features.
    if cfg.TRAIN.SEG_HEAD_TYPE == "euclidean_raw":
        assert all(item is None for item in features)
    else:
        assert all(torch.isfinite(item).all() for item in features)
    return model, logits, features


def test_baseline():
    cfg = _make_config("euclidean_raw", False)
    model, logits, features = test_model_shapes(cfg)
    keys = list(model.state_dict())
    assert not any("shared_hyp_projector" in key for key in keys), "baseline must not include SharedHypProjector"
    assert sum("conv_out.weight" in key for key in keys) == len(cfg.DATASET.DATA.LABEL_NUMBER)
    assert not any("mobius_out" in key for key in keys)
    print("baseline smoke test passed:", [tuple(x.shape) for x in logits])


def test_m0():
    cfg = _make_config("euclidean_hyp", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointNet2(cfg, SimpleNamespace(mc_level=-1)).to(device)
    dropout_calls = [0]
    hooks = [decoder.drop1.register_forward_hook(lambda *unused: dropout_calls.__setitem__(0, dropout_calls[0] + 1))
             for decoder in model.decoders]
    model.eval()
    logits, hyp_features = model(torch.randn(1, 6, 1024, device=device))
    expected = [(1, count, 1024) for count in cfg.DATASET.DATA.LABEL_NUMBER]
    assert [tuple(item.shape) for item in logits] == expected
    assert dropout_calls[0] == 0, "euclidean_hyp head must not apply dropout"
    ball_radius = 1.0 / float(cfg.TRAIN.HYP_C) ** 0.5
    assert all(bool(torch.isfinite(item).all()) and float(item.norm(dim=1).max()) < ball_radius for item in hyp_features)
    keys = list(model.state_dict())
    assert any("shared_hyp_projector.pre_poincare" in key for key in keys)
    assert sum("conv_out.weight" in key for key in keys) == len(cfg.DATASET.DATA.LABEL_NUMBER)
    assert not any("mobius_out" in key for key in keys)
    for hook in hooks:
        hook.remove()
    print("m0 smoke test passed:", expected)


def test_m1():
    cfg = _make_config("mobius", False)
    model, logits, _ = test_model_shapes(cfg)
    keys = list(model.state_dict())
    assert any("shared_hyp_projector" in key for key in keys)
    assert sum("mobius_out" in key for key in keys) >= len(cfg.DATASET.DATA.LABEL_NUMBER)
    print("m1 smoke test passed:", [tuple(x.shape) for x in logits])


def test_hierhyp():
    cfg = _make_config("mobius", True)
    model, logits, _ = test_model_shapes(cfg)
    keys = list(model.state_dict())
    assert any("shared_hyp_projector" in key for key in keys)
    assert sum("mobius_out" in key for key in keys) >= len(cfg.DATASET.DATA.LABEL_NUMBER)
    print("hierhyp smoke test passed:", [tuple(x.shape) for x in logits])


def test_m2():
    cfg = _make_config("euclidean_hyp", True)
    model, logits, _ = test_model_shapes(cfg)
    keys = list(model.state_dict())
    assert any("shared_hyp_projector" in key for key in keys)
    assert sum("conv_out.weight" in key for key in keys) == len(cfg.DATASET.DATA.LABEL_NUMBER)
    assert not any("mobius_out" in key for key in keys)
    print("m2 smoke test passed:", [tuple(x.shape) for x in logits])


def test_pct_loss():
    classes = [3, 5, 7]
    parents = [
        torch.tensor([0, 1, 1, 2, 2]),
        torch.tensor([0, 1, 1, 2, 2, 3, 4]),
    ]
    depth_deltas = [
        torch.tensor([0, 1, 1, 1, 1]),
        torch.tensor([0, 1, 0, 1, 1, 2, 0]),
    ]
    labels = torch.stack([
        torch.randint(1, 3, (2, 16)),
        torch.randint(1, 5, (2, 16)),
        torch.randint(1, 7, (2, 16)),
    ], dim=-1)
    raw = [(torch.randn(2, 16, 8) * 0.05).requires_grad_() for _ in classes]
    features = [pmath.expmap0(item, c=0.2) for item in raw]
    bank = PrototypeBank(classes, sz_embed=8, hyp_c=0.2, momentum=0.5)
    for level_idx, feature in enumerate(features):
        bank.update_level_object_balanced(level_idx, feature, labels[..., level_idx])
    loss_fn = PCTLoss(classes, parents, depth_deltas, hyp_c=0.2, temp=2.0)
    loss, _ = loss_fn(
        [item.reshape(-1, 8) for item in features],
        labels.reshape(-1, 3),
        bank,
        cons_weight=0.05,
        tri_weight=0.10,
        radius_weight=0.02,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(item.grad is not None and torch.isfinite(item.grad).all() for item in raw)
    print("PCT backward loss:", float(loss.detach()))


def test_dataset_configs():
    """Both dataset configs must load and expose the expected label counts."""
    partnext_cfg = load_config(str(CONFIG_PATH / "partnext"))
    assert str(partnext_cfg.DATASET.DATA.DATASET_NAME).lower() == "partnext"
    assert list(partnext_cfg.DATASET.DATA.LABEL_NUMBER) == [63, 264, 638, 919, 1260]
    campus3d_cfg = load_config(str(CONFIG_PATH / "campus3d"))
    assert str(campus3d_cfg.DATASET.DATA.DATASET_NAME).lower() == "campus3d"
    assert list(campus3d_cfg.DATASET.DATA.LABEL_NUMBER) == [3, 4, 6, 9, 15]
    assert campus3d_cfg.TRAIN.MAX_EPOCH == 50
    print("dataset configs OK: partnext={}, campus3d={}".format(
        list(partnext_cfg.DATASET.DATA.LABEL_NUMBER),
        list(campus3d_cfg.DATASET.DATA.LABEL_NUMBER)))


if __name__ == "__main__":
    test_dataset_configs()
    test_baseline()
    test_m0()
    test_m1()
    test_m2()
    test_hierhyp()
    test_pct_loss()
    print("All PartNeXt ablation smoke tests passed.")
