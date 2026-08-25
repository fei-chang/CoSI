"""Metric helpers used by CoSI evaluation.

This module replaces the missing `metrics.py` dependency in `raw_codes`.
"""

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    roc_auc_score,
)


def multi_hot_targets(gaze_points, image_size):
    width, height = map(int, image_size)
    target = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(gaze_points.detach().cpu() if hasattr(gaze_points, "detach") else gaze_points)
    for x, y in points.reshape(-1, 2):
        if x >= 0 and y >= 0:
            px = min(int(x * width), width - 1)
            py = min(int(y * height), height - 1)
            target[py, px] = 1
    return target


def auc(target, prediction, is_im=False):
    target = np.asarray(target).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    return roc_auc_score(target, prediction)


def ap(target, prediction):
    return average_precision_score(np.asarray(target), np.asarray(prediction))


def acc(target, prediction):
    return accuracy_score(np.asarray(target), np.asarray(prediction))


def classwise_metrics(target, prediction):
    return classification_report(np.asarray(target), np.asarray(prediction), digits=4)


def L2_dist(point_a, point_b):
    a = np.asarray(point_a.detach().cpu() if hasattr(point_a, "detach") else point_a)
    b = np.asarray(point_b.detach().cpu() if hasattr(point_b, "detach") else point_b)
    return float(np.linalg.norm(a.astype(float) - b.astype(float)))


def argmax_pts(heatmap):
    heatmap = np.asarray(heatmap)
    row, col = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
    return float(col), float(row)


def dark_inference(heatmap):
    """Refine the heatmap maximum using the local log-gradient (DARK)."""
    heatmap = np.asarray(heatmap, dtype=np.float32)
    blurred = cv2.GaussianBlur(heatmap, (3, 3), 0)
    logged = np.log(np.maximum(blurred, 1e-10))
    x, y = argmax_pts(logged)
    x_i, y_i = int(x), int(y)
    if 1 <= x_i < logged.shape[1] - 1 and 1 <= y_i < logged.shape[0] - 1:
        dx = 0.5 * (logged[y_i, x_i + 1] - logged[y_i, x_i - 1])
        dy = 0.5 * (logged[y_i + 1, x_i] - logged[y_i - 1, x_i])
        dxx = logged[y_i, x_i + 1] - 2 * logged[y_i, x_i] + logged[y_i, x_i - 1]
        dyy = logged[y_i + 1, x_i] - 2 * logged[y_i, x_i] + logged[y_i - 1, x_i]
        dxy = 0.25 * (
            logged[y_i + 1, x_i + 1]
            - logged[y_i + 1, x_i - 1]
            - logged[y_i - 1, x_i + 1]
            + logged[y_i - 1, x_i - 1]
        )
        hessian = np.array([[dxx, dxy], [dxy, dyy]], dtype=np.float32)
        gradient = np.array([dx, dy], dtype=np.float32)
        if abs(np.linalg.det(hessian)) > 1e-6:
            offset = -np.linalg.solve(hessian, gradient)
            x += float(np.clip(offset[0], -1, 1))
            y += float(np.clip(offset[1], -1, 1))
    return x, y
