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


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1 + 1), max(0, iy2 - iy1 + 1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    area_b = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _nms_detections(
    detections: List[Detection],
    iou_thresh: float = 0.3,
    max_detections: int = 10,
) -> List[Detection]:
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[Detection] = []
    for det in sorted_dets:
        if all(_bbox_iou(det.bbox, k.bbox) <= iou_thresh for k in kept):
            kept.append(det)
        if len(kept) >= max_detections:
            break
    return kept


def extract_detections(
    x: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    height: int,
    width: int,
    score_thresh: float = 0.30,
    min_area: int = 5,
    morph_kernel: int = 3,
    dilate_iterations: int = 1,
    bbox_margin: int = 4,
    min_box_size: int = 3,
    max_box_area_ratio: float = 0.15,
    nms_iou: float = 0.3,
    max_detections: int = 10,
) -> Tuple[List[Detection], np.ndarray]:

    score_map, binary_map = build_score_map(x, y, scores, height, width, score_thresh)
    if binary_map.max() == 0:
        return [], score_map

    # Morphological close to bridge gaps in fragmented event clusters,
    # then slight dilation to expand the footprint of tiny targets.
    if morph_kernel > 1:
        kernel_size = morph_kernel + (1 - morph_kernel % 2)  # keep odd
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel)
        if dilate_iterations > 0:
            binary_map = cv2.dilate(binary_map, kernel, iterations=dilate_iterations)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_map, connectivity=8, ltype=cv2.CV_32S
    )

    frame_area = float(max(width * height, 1))
    detections: List[Detection] = []

    for label_idx in range(1, num_labels):
        x0, y0, w, h, area = stats[label_idx]
        if area < min_area or w < min_box_size or h < min_box_size:
            continue

        x1 = max(0, int(x0) - bbox_margin)
        y1 = max(0, int(y0) - bbox_margin)
        x2 = min(width - 1, int(x0 + w - 1) + bbox_margin)
        y2 = min(height - 1, int(y0 + h - 1) + bbox_margin)

        box_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        if box_area / frame_area > max_box_area_ratio:
            continue

        component_mask = labels == label_idx
        component_scores = score_map[component_mask]
        score = float(component_scores.mean()) if component_scores.size else 0.0
        cx, cy = centroids[label_idx]

        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                center=(float(cx), float(cy)),
                score=score,
                area=int(area),
            )
        )

    detections = _nms_detections(detections, nms_iou, max_detections)
    return detections, score_map
