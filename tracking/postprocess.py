from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    score: float
    area: int


def build_score_map(
    x: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    height: int,
    width: int,
    score_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = scores >= score_thresh
    score_map = np.zeros((height, width), dtype=np.float32)
    binary_map = np.zeros((height, width), dtype=np.uint8)

    if not np.any(valid):
        return score_map, binary_map

    x_sel = x[valid].astype(np.int32)
    y_sel = y[valid].astype(np.int32)
    score_sel = scores[valid].astype(np.float32)

    np.add.at(score_map, (y_sel, x_sel), score_sel)
    binary_map[y_sel, x_sel] = 1
    return score_map, binary_map


def extract_detections(
    x: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    height: int,
    width: int,
    score_thresh: float = 0.35,
    min_area: int = 3,
) -> Tuple[List[Detection], np.ndarray]:
    score_map, binary_map = build_score_map(x, y, scores, height, width, score_thresh)
    if binary_map.max() == 0:
        return [], score_map

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_map, connectivity=8, ltype=cv2.CV_32S
    )

    detections: List[Detection] = []
    for label_idx in range(1, num_labels):
        x0, y0, w, h, area = stats[label_idx]
        if area < min_area:
            continue

        component_mask = labels == label_idx
        component_scores = score_map[component_mask]
        score = float(component_scores.mean()) if component_scores.size else 0.0
        cx, cy = centroids[label_idx]
        bbox = (int(x0), int(y0), int(x0 + w - 1), int(y0 + h - 1))
        detections.append(
            Detection(
                bbox=bbox,
                center=(float(cx), float(cy)),
                score=score,
                area=int(area),
            )
        )

    detections.sort(key=lambda det: det.score, reverse=True)
    return detections, score_map
