from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyptorch import pmath


def _artanh(x):
    x = x.clamp(-1 + 1e-5, 1 - 1e-5)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def logmap0(y, c=1.0):
    c = torch.as_tensor(c, dtype=y.dtype, device=y.device)
    sqrt_c = torch.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True, p=2).clamp_min(1e-5)
    scale = _artanh(sqrt_c * y_norm) / (sqrt_c * y_norm)
    return scale * y


class HierarchicalCrossEntropyLoss:
    def __init__(self, weights, device):
        self.weights = [torch.as_tensor(weight, dtype=torch.float32, device=device) for weight in weights]
        self.losses = [nn.CrossEntropyLoss(weight=weight) for weight in self.weights]

    def __call__(self, pred, target, level):
        pred = pred.reshape(-1, len(self.weights[level]))
        target = target.reshape(-1)
        return self.losses[level](pred, target)


class LogitsConsistencyLoss:
    def __init__(self, matrices, weights, device):
        self.gather_ids = [
            torch.as_tensor(np.argmax(matrix, axis=0), dtype=torch.long, device=device)
            for matrix in matrices
        ]
        self.weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

    def __call__(self, preds):
        probs = [F.softmax(pred.permute(0, 2, 1), dim=-1) for pred in preds]
        terms = []
        for level_idx, gather_id in enumerate(self.gather_ids):
            child = probs[level_idx + 1]
            parent = probs[level_idx][..., gather_id]
            terms.append(F.relu(child - parent).mean() * self.weights[level_idx])
        return torch.stack(terms).mean() if terms else probs[0].new_zeros(())


class PrototypeBank(nn.Module):
    def __init__(self, classes_num, sz_embed=128, hyp_c=0.1, momentum=0.5):
        super().__init__()
        self.classes_num = list(classes_num)
        self.sz_embed = int(sz_embed)
        self.hyp_c = float(hyp_c)
        self.momentum = float(momentum)
        self.prototype_names = []
        self.initialized_names = []
        self.update_count_names = []

        for level_idx, class_num in enumerate(self.classes_num):
            proto_name = "prototype_tangent_{}".format(level_idx)
            init_name = "prototype_initialized_{}".format(level_idx)
            count_name = "prototype_update_count_{}".format(level_idx)
            self.register_buffer(proto_name, torch.zeros(class_num, self.sz_embed))
            self.register_buffer(init_name, torch.zeros(class_num, dtype=torch.bool))
            self.register_buffer(count_name, torch.zeros(class_num, dtype=torch.long))
            self.prototype_names.append(proto_name)
            self.initialized_names.append(init_name)
            self.update_count_names.append(count_name)

    def set_momentum(self, momentum):
        self.momentum = float(momentum)

    def _tangent(self, level_idx):
        return getattr(self, self.prototype_names[level_idx])

    def _initialized(self, level_idx):
        return getattr(self, self.initialized_names[level_idx])

    def _update_count(self, level_idx):
        return getattr(self, self.update_count_names[level_idx])

    def get_level_valid_mask(self, level_idx):
        return self._initialized(level_idx)

    def get_level_prototypes(self, level_idx):
        tangent = self._tangent(level_idx)
        return pmath.project(pmath.expmap0(tangent, c=self.hyp_c), c=self.hyp_c)

    @torch.no_grad()
    def update_level_object_balanced(self, level_idx, hyp_features, labels):
        """Aggregate each class per object first, then average object means."""
        if hyp_features.dim() != 3 or labels.dim() != 2:
            raise ValueError("Expected hyp_features [B,N,D] and labels [B,N].")
        if hyp_features.shape[:2] != labels.shape:
            raise ValueError("Feature and label shapes do not align.")

        batch_size, num_points, dim = hyp_features.shape
        tangent = logmap0(hyp_features.detach().reshape(-1, dim), c=self.hyp_c)
        flat_labels = labels.detach().reshape(-1)
        object_ids = torch.arange(batch_size, device=labels.device).repeat_interleave(num_points)
        valid = flat_labels > 0
        if not bool(valid.any()):
            return

        tangent = tangent[valid]
        flat_labels = flat_labels[valid]
        object_ids = object_ids[valid]
        prototypes = self._tangent(level_idx)
        initialized = self._initialized(level_idx)
        update_counts = self._update_count(level_idx)

        for label in torch.unique(flat_labels).tolist():
            cls_idx = int(label)
            cls_mask = flat_labels == cls_idx
            cls_features = tangent[cls_mask]
            cls_objects = object_ids[cls_mask]
            object_sums = cls_features.new_zeros((batch_size, dim))
            object_counts = cls_features.new_zeros((batch_size, 1))
            object_sums.index_add_(0, cls_objects, cls_features)
            object_counts.index_add_(0, cls_objects, torch.ones_like(cls_objects, dtype=cls_features.dtype).unsqueeze(1))
            present = object_counts.squeeze(1) > 0
            cls_mean = (object_sums[present] / object_counts[present]).mean(dim=0)
            if initialized[cls_idx]:
                prototypes[cls_idx].mul_(self.momentum).add_(cls_mean * (1.0 - self.momentum))
            else:
                prototypes[cls_idx].copy_(cls_mean)
                initialized[cls_idx] = True
            update_counts[cls_idx] += 1

    @torch.no_grad()
    def export_state(self):
        state = {
            "classes_num": list(self.classes_num),
            "sz_embed": self.sz_embed,
            "hyp_c": self.hyp_c,
            "momentum": self.momentum,
        }
        for level_idx in range(len(self.classes_num)):
            state["prototype_tangent_{}".format(level_idx)] = self._tangent(level_idx).cpu()
            state["prototype_hyperbolic_{}".format(level_idx)] = self.get_level_prototypes(level_idx).cpu()
            state["prototype_initialized_{}".format(level_idx)] = self._initialized(level_idx).cpu()
            state["prototype_update_count_{}".format(level_idx)] = self._update_count(level_idx).cpu()
        return state


class PCTLoss(nn.Module):
    def __init__(self, classes_num, parent_maps, depth_deltas, hyp_c=0.1, temp=1.0,
                 tri_margin=0.05, radius_margin=0.05,
                 skip_repeated_edges=True, scale_margin_by_depth=True):
        super().__init__()
        self.classes_num = list(classes_num)
        self.hyp_c = float(hyp_c)
        self.temp = max(float(temp), 1e-6)
        self.tri_margin = max(float(tri_margin), 0.0)
        self.radius_margin = max(float(radius_margin), 0.0)
        self.skip_repeated_edges = bool(skip_repeated_edges)
        self.scale_margin_by_depth = bool(scale_margin_by_depth)
        self.parent_map_names = []
        self.depth_delta_names = []

        for level_idx, (parent_map, depth_delta) in enumerate(zip(parent_maps, depth_deltas), start=1):
            parent_name = "parent_map_{}".format(level_idx)
            delta_name = "depth_delta_{}".format(level_idx)
            self.register_buffer(parent_name, torch.as_tensor(parent_map, dtype=torch.long))
            self.register_buffer(delta_name, torch.as_tensor(depth_delta, dtype=torch.float32))
            self.parent_map_names.append(parent_name)
            self.depth_delta_names.append(delta_name)

    def _parent_map(self, level_idx):
        return getattr(self, self.parent_map_names[level_idx - 1])

    def _depth_delta(self, level_idx):
        return getattr(self, self.depth_delta_names[level_idx - 1])

    def _level_probabilities(self, features, level_idx, bank):
        valid_prototypes = bank.get_level_valid_mask(level_idx)
        if int(valid_prototypes.sum().item()) == 0:
            return None, valid_prototypes
        prototypes = bank.get_level_prototypes(level_idx)
        distances = pmath.dist_matrix(features, prototypes, c=self.hyp_c)
        distances = distances.masked_fill((~valid_prototypes).unsqueeze(0), 1e6)
        return F.softmax(-distances / self.temp, dim=1), valid_prototypes

    def _consistency_loss(self, features, labels, bank):
        terms = []
        for level_idx in range(1, len(self.classes_num)):
            sample_mask = (labels[:, level_idx - 1] > 0) & (labels[:, level_idx] > 0)
            if not bool(sample_mask.any()):
                continue
            parent_probs, parent_valid = self._level_probabilities(features[level_idx - 1][sample_mask], level_idx - 1, bank)
            child_probs, child_valid = self._level_probabilities(features[level_idx][sample_mask], level_idx, bank)
            if parent_probs is None or child_probs is None:
                continue
            child_to_parent = parent_probs.new_zeros(parent_probs.shape)
            child_to_parent.index_add_(1, self._parent_map(level_idx), child_probs)
            valid_columns = parent_valid
            if bool(valid_columns.any()):
                terms.append((child_to_parent[:, valid_columns] - parent_probs[:, valid_columns]).square().mean())
        return torch.stack(terms).mean() if terms else features[0].new_zeros(())

    def _triplet_loss(self, features, labels, bank):
        terms = []
        for level_idx in range(1, len(self.classes_num)):
            parent_targets = labels[:, level_idx - 1]
            parent_valid = bank.get_level_valid_mask(level_idx - 1)
            sample_mask = parent_targets > 0
            sample_mask = sample_mask & parent_valid[parent_targets.clamp_min(0)]
            if not bool(sample_mask.any()) or int(parent_valid.sum().item()) <= 1:
                continue
            prototypes = bank.get_level_prototypes(level_idx - 1)
            distances = pmath.dist_matrix(features[level_idx][sample_mask], prototypes, c=self.hyp_c)
            distances = distances.masked_fill((~parent_valid).unsqueeze(0), 1e6)
            targets = parent_targets[sample_mask]
            positive = distances.gather(1, targets.unsqueeze(1)).squeeze(1)
            negatives = distances.clone()
            negatives.scatter_(1, targets.unsqueeze(1), 1e6)
            negative = negatives.min(dim=1)[0]
            valid_negative = torch.isfinite(negative) & (negative < 1e5)
            if bool(valid_negative.any()):
                terms.append(F.relu(positive[valid_negative] - negative[valid_negative] + self.tri_margin).mean())
        return torch.stack(terms).mean() if terms else features[0].new_zeros(())

    def _radius_loss(self, features, labels):
        terms = []
        for level_idx in range(1, len(self.classes_num)):
            child_targets = labels[:, level_idx]
            valid = (labels[:, level_idx - 1] > 0) & (child_targets > 0)
            deltas = self._depth_delta(level_idx)[child_targets.clamp_min(0)]
            if self.skip_repeated_edges:
                valid = valid & (deltas > 0)
            if not bool(valid.any()):
                continue
            parent_radius = pmath.dist(features[level_idx - 1][valid], torch.zeros_like(features[level_idx - 1][valid]), c=self.hyp_c)
            child_radius = pmath.dist(features[level_idx][valid], torch.zeros_like(features[level_idx][valid]), c=self.hyp_c)
            margin_scale = deltas[valid] if self.scale_margin_by_depth else torch.ones_like(deltas[valid])
            terms.append(F.relu(self.radius_margin * margin_scale + parent_radius - child_radius).mean())
        return torch.stack(terms).mean() if terms else features[0].new_zeros(())

    def forward(self, features, labels, bank, cons_weight, tri_weight, radius_weight):
        zero = features[0].new_zeros(())
        cons = self._consistency_loss(features, labels, bank) if cons_weight > 0 else zero
        tri = self._triplet_loss(features, labels, bank) if tri_weight > 0 else zero
        radius = self._radius_loss(features, labels) if radius_weight > 0 else zero
        total = cons_weight * cons + tri_weight * tri + radius_weight * radius
        return total, {
            "cons": cons.detach(),
            "tri": tri.detach(),
            "radius": radius.detach(),
            "cons_weighted": (cons_weight * cons).detach(),
            "tri_weighted": (tri_weight * tri).detach(),
            "radius_weighted": (radius_weight * radius).detach(),
            "total": total.detach(),
        }
