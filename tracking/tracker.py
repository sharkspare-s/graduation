from dataclasses import dataclass, field
from typing import Deque, List, Sequence, Tuple

import numpy as np

from tracking.postprocess import Detection

# How many recent positions to remember for static-track detection
_STATIC_HISTORY_LEN = 8


class KalmanFilter:
    """4D constant-velocity Kalman filter: state = [cx, cy, vx, vy]."""

    def __init__(
        self,
        dt: float = 1.0,
        pos_std: float = 0.08,
        vel_std: float = 0.15,
        meas_std: float = 0.12,
    ):
        self.dt = dt
        self.x = np.zeros((4, 1), dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.5

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        q_pos = pos_std ** 2
        q_vel = vel_std ** 2
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

        r = meas_std ** 2
        self.R = np.diag([r, r]).astype(np.float32)

        self._initialized = False

    def init(self, cx: float, cy: float) -> None:
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32) * 0.5
        self._initialized = True

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        if not self._initialized:
            self.init(cx, cy)
            return cx, cy

        z = np.array([[cx], [cy]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def center(self) -> np.ndarray:
        return self.x[:2, 0].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:, 0].copy()


@dataclass
class Track:
    track_id: int
    kf: KalmanFilter = field(default_factory=KalmanFilter)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_w: float = 0.0
    bbox_h: float = 0.0
    score: float = 0.0
    age: int = 1
    hits: int = 1
    missed: int = 0
    # History of quantised centroids for static-track detection
    _pos_history: Deque[Tuple[int, int]] = field(default_factory=lambda: Deque(maxlen=_STATIC_HISTORY_LEN))

    @property
    def avg_score(self) -> float:
        """Running average score (approximate)."""
        return self.score  # score is already the latest; use in combination with hits

    @property
    def center(self) -> np.ndarray:
        return self.kf.center

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    def predict(self) -> None:
        self.kf.predict()
        vx, vy = self.velocity
        shift_x = int(np.rint(vx))
        shift_y = int(np.rint(vy))
        x1, y1, x2, y2 = self.bbox
        self.bbox = (x1 + shift_x, y1 + shift_y, x2 + shift_x, y2 + shift_y)
        self.age += 1
        self.missed += 1

    def update(self, detection: Detection, bbox_ema: float = 0.35) -> None:
        self.kf.update(detection.center[0], detection.center[1])

        x1, y1, x2, y2 = detection.bbox
        w, h = float(x2 - x1 + 1), float(y2 - y1 + 1)

        if self.bbox_w <= 0:
            self.bbox_w, self.bbox_h = w, h
        else:
            self.bbox_w = bbox_ema * w + (1.0 - bbox_ema) * self.bbox_w
            self.bbox_h = bbox_ema * h + (1.0 - bbox_ema) * self.bbox_h

        cx, cy = self.center
        hw, hh = self.bbox_w * 0.5, self.bbox_h * 0.5
        self.bbox = (
            int(cx - hw),
            int(cy - hh),
            int(cx + hw),
            int(cy + hh),
        )
        self.score = detection.score
        self.hits += 1
        self.missed = 0

        # Record quantised position
        self._pos_history.append((int(cx) // 4, int(cy) // 4))

    def is_static(self) -> bool:
        """True if the track has barely moved over the recent history window."""
        if len(self._pos_history) < 3:
            return False
        unique = set(self._pos_history)
        return len(unique) <= 2

    @property
    def confirmed(self) -> bool:
        return self.hits >= 2


class MultiObjectTracker:
    def __init__(
        self,
        max_distance: float = 30.0,
        max_missed: int = 8,
        min_hits: int = 2,
        min_track_score: float = 0.30,
        static_thresh: int = 4,
    ):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.min_hits = min_hits
        self.min_track_score = min_track_score
        self.static_thresh = static_thresh
        self.next_id = 1
        self.tracks: List[Track] = []

    def _pairwise_distance(
        self, tracks: Sequence[Track], detections: Sequence[Detection]
    ) -> np.ndarray:
        if not tracks or not detections:
            return np.empty((len(tracks), len(detections)), dtype=np.float32)

        track_centers = np.stack([t.center for t in tracks], axis=0)
        det_centers = np.stack([
            np.array(d.center, dtype=np.float32) for d in detections
        ], axis=0)
        diff = track_centers[:, None, :] - det_centers[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def _greedy_assign(
        self, detections: Sequence[Detection]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not self.tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(self.tracks))), []

        distance = self._pairwise_distance(self.tracks, detections)
        matches: List[Tuple[int, int]] = []
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        while distance.size:
            t_idx, d_idx = np.unravel_index(np.argmin(distance), distance.shape)
            if distance[t_idx, d_idx] > self.max_distance:
                break
            if t_idx in unmatched_tracks and d_idx in unmatched_detections:
                matches.append((t_idx, d_idx))
                unmatched_tracks.remove(t_idx)
                unmatched_detections.remove(d_idx)
            distance[t_idx, :] = np.inf
            distance[:, d_idx] = np.inf
            if not np.isfinite(distance).any():
                break

        return matches, sorted(unmatched_tracks), sorted(unmatched_detections)

    def _prune_tracks(self) -> None:
        self.tracks = [
            t for t in self.tracks
            if t.missed <= self.max_missed and not t.is_static()
        ]

    def visible_tracks(self, allow_missed: bool = False) -> List[Track]:
        return [
            t for t in self.tracks
            if t.hits >= self.min_hits
            and (allow_missed or t.missed == 0)
            and t.score >= self.min_track_score
        ]

    def predict_only(self) -> List[Track]:
        for track in self.tracks:
            track.predict()
        self._prune_tracks()
        return self.visible_tracks(allow_missed=True)

    def update(self, detections: Sequence[Detection]) -> List[Track]:
        for track in self.tracks:
            track.predict()

        matches, unmatched_tracks, unmatched_detections = self._greedy_assign(detections)

        for t_idx, d_idx in matches:
            self.tracks[t_idx].update(detections[d_idx])

        for d_idx in unmatched_detections:
            det = detections[d_idx]
            track = Track(track_id=self.next_id, score=det.score)
            track.kf.init(det.center[0], det.center[1])
            x1, y1, x2, y2 = det.bbox
            track.bbox = det.bbox
            track.bbox_w = float(x2 - x1 + 1)
            track.bbox_h = float(y2 - y1 + 1)
            self.tracks.append(track)
            self.next_id += 1

        if unmatched_tracks:
            self.tracks = [
                t for idx, t in enumerate(self.tracks)
                if idx not in unmatched_tracks or t.missed <= self.max_missed
            ]

        self._prune_tracks()
        return self.visible_tracks(allow_missed=False)
