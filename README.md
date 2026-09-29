# HY3D: TAHL3DPC

![model](model.png)
The unified PartNeXt/Campus3D ablation experiment code allows switching between the model and dataset using the `--model` and `--dataset` parameters.

## Environmental installation

```bash
conda create -n hy3d python=3.8
conda activate hy3d
pip install -r requirements.txt
```

## Quick Start

```bash
# Training (default dataset PartNeXt)
python -m Hy3D.run --model baseline   # PointNet++ Euclidean baseline
python -m Hy3D.run --model m0 # + hyperbolic projection + Euclidean classifier head
python -m Hy3D.run --model m1 # + Mobius classifier head
python -m Hy3D.run --model m2 # m0 + PCT loss
python -m Hy3D.run --model hy3d # m1 + PCT loss (complete model)

# Switch to Campus3D
python -m Hy3D.run --dataset campus3d --model hy3d

# Eval
python -m Hy3D.run --model hy3d --eval
python -m Hy3D.run --dataset campus3d --model m1 --eval

# Custom experiment name (default name will be used if not specified)
python -m Hy3D.run --model m0 --exp_name MY_M0_RUN
```

## Model Preset

| model    | SEG_HEAD_TYPE  | PCT_ENABLE | Explanation                       |
|----------|----------------|------------|-----------------------------------|
| baseline | euclidean_raw  | False      | Pure Euclidean PointNet++           |
| m0       | euclidean_hyp  | False      | + hyperbolic projection + Conv head          |
| m1       | mobius         | False      | + Mobius head                   |
| m2       | euclidean_hyp  | True       | m0 + PCT loss                     |
| hy3d     | mobius         | True       | m1 + PCT loss (complete model)         |

## Datasets

| dataset  | Explanation                      | Config                         |
|----------|-------------------------|-----------------------------------|
| partnext | PartNeXt | `Hy3D/configs/partnext/`          |
| campus3d | Campus3D         | `Hy3D/configs/campus3d/`          |

You can also use `--config_dir` to specify a custom configuration directory.

- PartNeXt Download：https://github.com/AuthorityWang/PartNeXt
- Campus3D Download：https://github.com/shinke-li/Campus3D


