"""Utility functions to see if a goal has been reached."""

from typing import Tuple, Union
from scipy import stats
import numpy as np


class GaussianImageGoalChecker:
    def __init__(
        self,
        image_shape: Tuple[int, ...],
        sigma: float = 0.1,
        logpdf_frac: float = 0.99,
    ):
        self._image_shape = image_shape
        self._gaussian = stats.norm(0, sigma)
        self._threshold = (
            self._heuristic(np.zeros(image_shape), np.zeros(image_shape)) * logpdf_frac
        )

    def _batched_heuristic(self, imagesA, imagesB):
        batch_size = imagesA.shape[0]
        assert imagesA.shape == imagesB.shape

        imagesA = imagesA.reshape(batch_size, -1)
        imagesB = imagesB.reshape(batch_size, -1)

        return self._gaussian.logpdf(imagesA - imagesB).mean(axis=1)

    def _heuristic(self, imageA, imageB):
        return self._batched_heuristic(imageA[None], imageB[None])[0]

    def goal_reached_value(self, imagesA, imagesB) -> np.ndarray:
        if imagesA.shape != imagesB.shape:
            raise ValueError("Images must have the same shape")

        if imagesA.ndim == len(self._image_shape):
            return self._heuristic(imagesA, imagesB)

        return self._batched_heuristic(imagesA, imagesB)

    def goal_reached(self, imagesA, imagesB) -> Union[bool, np.ndarray]:
        return self.goal_reached_value(imagesA, imagesB) > self._threshold


class BinaryIOUGoalChecker(object):
    def __init__(self, threshold: float = 0.98):
        self._threshold = threshold

    @staticmethod
    def _batched_binary_iou_single(masks1, masks2):
        masks1 = masks1 > 0.5
        masks2 = masks2 > 0.5

        # masks is either (batch, w, h) or (batch, w, h, 1)
        if masks1.ndim == 4:
            masks1 = masks1.squeeze(-1)
        if masks2.ndim == 4:
            masks2 = masks2.squeeze(-1)

        masks1_area = np.count_nonzero(masks1, axis=(1, 2))
        masks2_area = np.count_nonzero(masks2, axis=(1, 2))
        intersection = np.count_nonzero(np.logical_and(masks1, masks2), axis=(1, 2))

        both_empty = np.logical_and(masks1_area == 0, masks2_area == 0)
        masks1_area[both_empty] = 1
        masks2_area[both_empty] = 1
        intersection[both_empty] = 1

        iou = intersection / (masks1_area + masks2_area - intersection)

        return iou

    @staticmethod
    def _batched_binary_iou(masks1, masks2):
        a = BinaryIOUGoalChecker._batched_binary_iou_single(masks1, masks2)
        b = BinaryIOUGoalChecker._batched_binary_iou_single(1 - masks1, 1 - masks2)
        return (a + b) / 2.0

    def _goal_reached_value(self, masks1, masks2) -> Union[bool, np.ndarray]:
        # Input is either (batch, w, h, 1) or (w, h, 1)
        if masks1.shape != masks2.shape:
            raise ValueError("Masks must have the same shape")

        if masks1.ndim == 3:
            return BinaryIOUGoalChecker._batched_binary_iou(masks1[None], masks2[None])[
                0
            ]

        return BinaryIOUGoalChecker._batched_binary_iou(masks1, masks2)

    def goal_reached_value(self, masks1, masks2) -> Union[bool, np.ndarray]:
        vals1 = self._goal_reached_value(masks1, masks2)
        vals2 = self._goal_reached_value(1 - masks1, masks2)
        return np.maximum(vals1, vals2)

    def goal_reached(self, masks1, masks2) -> Union[bool, np.ndarray]:
        return self.goal_reached_value(masks1, masks2) > self._threshold
