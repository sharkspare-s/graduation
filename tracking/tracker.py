from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from tracking.postprocess import Detection


@dataclass
class Track:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    bbox: Tuple[int, int, int, int]
    score: float
    age: int = 1
    hits: int = 1
    missed: int = 0

    def predict(self) -> None:
        self.center = self.center + self.velocity
        self.age += 1
        self.missed += 1

    def update(self, detection: Detection, momentum: float = 0.7) -> None:
        new_center = np.array(detection.center, dtype=np.float32)
        measured_velocity = new_center - self.center
        self.velocity = momentum * self.velocity + (1.0 - momentum) * measured_velocity
        self.center = new_center
        self.bbox = detection.bbox
        self.score = detection.score
        self.hits += 1
        self.missed = 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2


class MultiObjectTracker:
    def __init__(self, max_distance: float = 20.0, max_missed: int = 5, min_hits: int = 2):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.next_id = 1
        self.tracks: List[Track] = []

    def _pairwise_distance(self, tracks: Sequence[Track], detections: Sequence[Detection]) -> np.ndarray:
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        track_centers = np.stack([track.center for track in tracks], axis=0)
        det_centers = np.stack([np.array(det.center, dtype=np.float32) for det in detections], axis=0)
        diff = track_centers[:, None, :] - det_centers[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _greedy_assign(self, detections: Sequence[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self.tracks))), []

        distance = self._pairwise_distance(self.tracks, detections)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        while distance.size:
            track_idx, det_idx = np.unravel_index(np.argmin(distance), distance.shape)
            if distance[track_idx, det_idx] > self.max_distance:
                break
            if track_idx in unmatched_tracks and det_idx in unmatched_detections:
                matches.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)
            distance[track_idx, :] = np.inf
            distance[:, det_idx] = np.inf
            if not np.isfinite(distance).any():
                break

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = self._greedy_assign(detections)

        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            self.tracks.append(
                Track(
                    track_id=self.next_id,
                    center=np.array(detection.center, dtype=np.float32),
                    velocity=np.zeros(2, dtype=np.float32),
                    bbox=detection.bbox,
                    score=detection.score,
                )
            )
            self.next_id += 1

        if unmatched_tracks:
            self.tracks = [
                track for idx, track in enumerate(self.tracks)
                if idx not in unmatched_tracks or track.missed <= self.max_missed
            ]
        else:
            self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]

        return [track for track in self.tracks if track.hits >= self.min_hits and track.missed == 0]
