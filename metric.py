import numpy as np


class IouMetric:
    def __init__(self, label_range):
        self.label_range = list(label_range)
        self.class_count = len(self.label_range)
        self.confusion_matrix = np.zeros((self.class_count, self.class_count), dtype=np.int64)

    def update(self, pred, target):
        pred = np.asarray(pred, dtype=np.int64).reshape(-1)
        target = np.asarray(target, dtype=np.int64).reshape(-1)
        if pred.shape != target.shape:
            raise ValueError("Prediction and target shapes differ.")
        valid = (pred >= 0) & (pred < self.class_count) & (target >= 0) & (target < self.class_count)
        encoded = pred[valid] * self.class_count + target[valid]
        self.confusion_matrix += np.bincount(
            encoded,
            minlength=self.class_count * self.class_count,
        ).reshape(self.class_count, self.class_count)

    def iou(self):
        intersection = np.diag(self.confusion_matrix).astype(np.float64)
        union = self.confusion_matrix.sum(axis=0) + self.confusion_matrix.sum(axis=1) - intersection
        result = np.full(self.class_count, -1.0, dtype=np.float64)
        valid = union > 0
        result[valid] = intersection[valid] / union[valid]
        return result

    def avg_iou(self):
        values = self.iou()
        valid = values >= 0
        return float(values[valid].mean()) if np.any(valid) else 0.0
