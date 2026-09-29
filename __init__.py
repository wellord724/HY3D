"""Unified PartNeXt ablation package.

Supports five ablation variants selected via ``--model``:

* ``baseline`` : PointNet++ with Euclidean raw heads (no hyperbolic, no PCT).
* ``m0``       : + SharedHypProjector, Euclidean classifier on Poincare features.
* ``m1``       : + Mobius classification head.
* ``m2``       : m0 + Prototype Bank and PCT losses.
* ``hierhyp``  : m1 + Prototype Bank and PCT losses (full model).
"""
