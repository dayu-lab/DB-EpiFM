import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


class Evaluator:
    def __init__(self, params, data_loader):
        self.params = params
        self.data_loader = data_loader

    def _collect(self, model):
        labels, logits = [], []
        model.eval()
        for x, y in self.data_loader:
            output = model(x.cuda().float())
            labels.append(y.detach().cpu().numpy())
            logits.append(output.detach().cpu().numpy())
        return np.concatenate(labels).reshape(-1), np.concatenate(logits)

    def get_metrics_for_binaryclass(self, model):
        labels, logits = self._collect(model)
        scores = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        predictions = (scores >= 0.5).astype(int)
        cm = confusion_matrix(labels.astype(int), predictions, labels=[0, 1])
        balanced_accuracy = balanced_accuracy_score(labels, predictions)
        return (
            balanced_accuracy,
            average_precision_score(labels, scores),
            roc_auc_score(labels, scores),
            cm,
        )

    def get_metrics_for_chbmit(self, model):
        labels, logits = self._collect(model)
        predictions = (logits.reshape(-1) >= 0.0).astype(int)
        cm = confusion_matrix(labels.astype(int), predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced_accuracy = (sensitivity + specificity) / 2.0
        return balanced_accuracy, sensitivity, specificity, cm

    def get_metrics_for_multiclass(self, model):
        labels, logits = self._collect(model)
        predictions = np.argmax(logits, axis=1)
        cm = confusion_matrix(labels.astype(int), predictions)
        return (
            balanced_accuracy_score(labels, predictions),
            cohen_kappa_score(labels, predictions),
            f1_score(labels, predictions, average="weighted"),
            cm,
        )

    def get_metrics_for_regression(self, model):
        labels, predictions = self._collect(model)
        predictions = predictions.reshape(-1)
        correlation = np.corrcoef(labels, predictions)[0, 1]
        residual = np.sum((labels - predictions) ** 2)
        total = np.sum((labels - np.mean(labels)) ** 2)
        r2 = 1.0 - residual / total
        rmse = np.sqrt(np.mean((labels - predictions) ** 2))
        return correlation, r2, rmse
