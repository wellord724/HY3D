from __future__ import division
from __future__ import print_function

import os

import numpy as np

from .data_utils.point_util import PointModifier
from .data_utils.random_machine import RandomMachine


class PartNeXtDataset(object):
    def __init__(self, set_name, params, is_training=True):
        self.params = params
        self.is_training = is_training
        self.num_classes = list(params.DATA.LABEL_NUMBER)
        self.num_points_per_sample = int(params.SAMPLE.SETTING.NUM_POINTS_PER_SAMPLE)
        self.modify_type = list(params.SAMPLE.SETTING.MODIFY_TYPE)
        self.remove_zero_label = bool(params.DATA.REMOVE_ZERO_LABEL)
        self.cache_root = params.DATA.PARTNEXT_CACHE
        self.split_dir = params.DATA.SPLIT_DIR or self.cache_root
        self.split_file = self._resolve_split_file(set_name)
        self.samples = self._load_split(self.split_file)
        self.samples = self._filter_usable_samples(self.samples)
        if len(self.samples) == 0:
            raise ValueError('PartNeXt split {} has no usable samples.'.format(self.split_file))
        self.modify_func = PointModifier(self.modify_type)
        self.random_machine = RandomMachine(
            basis_seed=params.SAMPLE.RANDOM_SEED_BASIS,
            call_length=max(1, len(self.samples)),
        )
        self.label_distribution = self._build_label_distribution()
        self.label_weights = self._cal_label_weights(
            self.label_distribution,
            setting=params.SAMPLE.LABEL_WEIGHT_POLICY,
        )

    def _resolve_split_file(self, set_name):
        mapping = {
            'TRAIN_SET': 'split_train.txt',
            'VALIDATION_SET': 'split_val.txt',
            'TEST_SET': 'split_test.txt',
        }
        if set_name not in mapping:
            raise KeyError('Unknown PartNeXt set name: {}'.format(set_name))
        return os.path.join(self.split_dir, mapping[set_name])

    def _load_split(self, split_file):
        if not os.path.isfile(split_file):
            raise IOError(
                'PartNeXt split file not found: {}. Run tools/prepare_partnext.py first.'.format(split_file)
            )

        samples = []
        with open(split_file, 'r') as f:
            for line in f:
                item = line.strip()
                if not item:
                    continue
                sample_path = item if os.path.isabs(item) else os.path.join(self.cache_root, item)
                if os.path.isdir(sample_path):
                    continue
                if not sample_path.endswith('.npz'):
                    sample_path = sample_path + '.npz'
                if os.path.isfile(sample_path):
                    samples.append(sample_path)
        return samples

    def _filter_usable_samples(self, samples):
        usable = []
        for sample_path in samples:
            try:
                points, colors, labels = self._load_npz(sample_path)
                points, colors, labels = self._filter_zero(points, colors, labels)
                if len(points) > 0:
                    usable.append(sample_path)
            except Exception as exc:
                print('Skip invalid PartNeXt sample {}: {}'.format(sample_path, exc))
        return usable

    def _load_npz(self, path):
        data = np.load(path, allow_pickle=True)
        points = data['points'].astype(np.float32)
        colors = data['colors'].astype(np.float32) if 'colors' in data else np.zeros_like(points, dtype=np.float32)
        labels = data['labels'].astype(np.int64)
        if labels.ndim != 2:
            raise ValueError('{} labels must have shape [N, K], got {}'.format(path, labels.shape))
        if labels.shape[1] != len(self.num_classes):
            raise ValueError(
                '{} label levels mismatch: labels has {}, config has {}'.format(
                    path,
                    labels.shape[1],
                    len(self.num_classes),
                )
            )
        if points.shape[0] != labels.shape[0]:
            raise ValueError('{} points and labels length mismatch.'.format(path))
        if colors.shape[0] != points.shape[0]:
            raise ValueError('{} colors and points length mismatch.'.format(path))
        return points, colors, labels

    def _filter_zero(self, points, colors, labels):
        if not self.remove_zero_label:
            return points, colors, labels
        valid = np.prod(labels != 0, axis=1).astype(np.bool_)
        return points[valid], colors[valid], labels[valid]

    def _build_label_distribution(self):
        label_dist = [np.zeros(num_class, dtype=np.float64) for num_class in self.num_classes]
        for path in self.samples:
            _, _, labels = self._load_npz(path)
            labels = labels.reshape(-1, labels.shape[-1])
            if self.remove_zero_label:
                labels = labels[np.prod(labels != 0, axis=1).astype(np.bool_)]
            for level_idx, num_class in enumerate(self.num_classes):
                valid_labels = labels[:, level_idx]
                valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < num_class)]
                if valid_labels.size == 0:
                    continue
                label_dist[level_idx] += np.bincount(valid_labels, minlength=num_class)
        return label_dist

    @staticmethod
    def _cal_label_weights(label_dist, setting='log', log_shift=1.2):
        all_label_weights = []
        for dist in label_dist:
            if setting == 'log':
                total = np.sum(dist)
                if total <= 0:
                    all_label_weights.append(np.ones_like(dist, dtype=np.float64))
                    continue
                label_weights = dist / total
                label_weights = 1 / np.log(log_shift + label_weights)
                all_label_weights.append(label_weights)
            elif setting == 'ones':
                all_label_weights.append(np.full(len(dist), 1.0))
            else:
                raise ValueError('Unknown LABEL_WEIGHT_POLICY: {}'.format(setting))
        return all_label_weights

    def get_label_weights(self, labels):
        weights = [self.label_weights[i][labels[..., i]] for i in range(len(self.label_weights))]
        weights = np.asarray(weights)
        if len(weights.shape) == 1:
            weights = np.expand_dims(weights, axis=0)
        permut = list(range(1, len(weights.shape)))
        permut.append(0)
        return weights.transpose(*permut)

    def _sample_indices(self, length, random_machine):
        if length <= 0:
            raise ValueError('Cannot sample from an empty PartNeXt object after zero-label filtering.')
        if length >= self.num_points_per_sample:
            return random_machine.choice(length, self.num_points_per_sample, replace=False)
        return random_machine.choice(length, self.num_points_per_sample, replace=True)

    @staticmethod
    def _normalize_points(points):
        centroid = np.mean(points, axis=0)
        points = points - centroid
        scale = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if scale > 0:
            points = points / scale
        return points

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        random_machine = self.random_machine.get_fix_machine(item)
        points, colors, labels = self._load_npz(self.samples[item])
        points, colors, labels = self._filter_zero(points, colors, labels)
        sample_index = self._sample_indices(len(points), random_machine)
        points = points[sample_index]
        colors = colors[sample_index]
        labels = labels[sample_index]
        points_centered = self._normalize_points(points)
        if self.modify_type not in (['raw'], ['block_centeralization']):
            try:
                points_centered = self.modify_func(
                    points,
                    center=np.mean(points, axis=0),
                    min_bounds=np.amin(points, axis=0),
                    max_bounds=np.amax(points, axis=0),
                    block_size_x=1.0,
                    block_size_y=1.0,
                )
            except TypeError:
                points_centered = self._normalize_points(points)
        weights = self.get_label_weights(labels) if self.is_training else points
        return points_centered.astype(np.float32), labels.astype(np.int64), colors.astype(np.float32), weights
