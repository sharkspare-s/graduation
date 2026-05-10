import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from metavision_core.event_io import EventsIterator
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIAction, UIKeyEvent

from tracking.postprocess import (
    StaticFilter,
    accumulate_score_map,
    build_score_heatmap,
    extract_detections,
)
from tracking.tracker import MultiObjectTracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EV-UAV tracking v3 (adaptive threshold + hot-pixel filter)")
    parser.add_argument("--config", default="configs/evisseg_evuav.yaml", type=str)
    parser.add_argument("--checkpoint", default="", type=str, help="Model checkpoint override")
    parser.add_argument("--input-path", default="", type=str,
                        help="Optional raw event file; leave empty for live camera")

    # --- timing ------------------------------------------------------------
    parser.add_argument("--window-us", default=30000, type=int,
                        help="Event accumulation window [us]")
    parser.add_argument("--detect-every", default=2, type=int,
                        help="Run detection every N windows (others: predict-only)")
    parser.add_argument("--max-result-lag", default=2, type=int,
                        help="Drop results older than this many windows")
    parser.add_argument("--display-fps", default=30, type=int)

    # --- data --------------------------------------------------------------
    parser.add_argument("--max-events", default=50000, type=int,
                        help="Cap events per window")
    parser.add_argument("--target-res", default="346x260", type=str,
                        help="Model target resolution WxH (events scaled from camera res to this)")

    # --- noise filters (Metavision SDK CV) ----------------------------------
    parser.add_argument("--activity-filter-us", default=10000, type=int,
                        help="Activity noise filter window [us] (0 to disable)")
    parser.add_argument("--trail-filter-us", default=1000, type=int,
                        help="Trail/hot-pixel filter window [us] (0 to disable)")

    # --- detection thresholds (adaptive) -----------------------------------
    parser.add_argument("--score-thresh-low", default=0.25, type=float,
                        help="Lower threshold used when no visible track exists (search mode)")
    parser.add_argument("--score-thresh-high", default=0.45, type=float,
                        help="Higher threshold used when tracks are active (tracking mode)")
    parser.add_argument("--score-ema", default=0.0, type=float,
                        help="Temporal EMA for score_map (0=disabled, 0.2-0.4 recommended for dim targets)")

    # --- morphology ---------------------------------------------------------
    parser.add_argument("--min-area", default=5, type=int)
    parser.add_argument("--morph-kernel", default=3, type=int,
                        help="Odd morphology kernel size (0 to disable)")
    parser.add_argument("--dilate-iterations", default=0, type=int)
    parser.add_argument("--bbox-margin", default=4, type=int)
    parser.add_argument("--min-box-size", default=3, type=int)
    parser.add_argument("--max-box-area-ratio", default=0.15, type=float)
    parser.add_argument("--nms-iou", default=0.3, type=float)
    parser.add_argument("--max-detections", default=10, type=int)

    # --- tracker -----------------------------------------------------------
    parser.add_argument("--max-distance", default=30.0, type=float)
    parser.add_argument("--max-missed", default=8, type=int)
    parser.add_argument("--min-hits", default=3, type=int,
                        help="Track confirmation threshold (hits needed to become visible)")
    parser.add_argument("--min-track-score", default=0.30, type=float,
                        help="Minimum detection score for a track to remain visible")
    parser.add_argument("--static-thresh", default=4, type=int,
                        help="Consecutive frames a detection stays still before being suppressed")

    # --- optimisations -----------------------------------------------------
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 inference (Jetson: ~1.5-2x faster if spconv supports it)")

    # --- debug -------------------------------------------------------------
    parser.add_argument("--debug", action="store_true",
                        help="Show score heatmap instead of binary foreground overlay")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config & model
# ---------------------------------------------------------------------------

def load_cfg(config_path: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.load(f, Loader=yaml.CLoader)
    flat = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat)


def build_model(cfg, config_path: str, checkpoint_path: str, use_fp16: bool = False):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required (spconv/HAIS voxelization is GPU-only).")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], "--config", config_path]
        from model.evspsegnet import evspsegnet
    finally:
        sys.argv = old_argv

    model = evspsegnet(cfg).cuda().eval()
    if use_fp16:
        model = model.half()

    try:
        state_dict = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------

def empty_viz_events():
    return {"x": np.empty((0,), dtype=np.int32),
            "y": np.empty((0,), dtype=np.int32),
            "p": np.empty((0,), dtype=np.bool_)}


def filter_events(events, activity_filter, trail_filter, buf):
    """Apply Metavision SDK noise filters.  Pass-through if both disabled."""
    if activity_filter is None and trail_filter is None:
        return events
    if len(events) == 0:
        return events
    if activity_filter is not None:
        activity_filter.process_events(events, buf)
        if trail_filter is not None:
            trail_filter.process_events_(buf)
        return buf.numpy()
    trail_filter.process_events(events, buf)
    return buf.numpy()


def build_evs_norm(events, height, width):
    x = events["x"].astype(np.float32)
    y = events["y"].astype(np.float32)
    t = events["t"].astype(np.float32)
    p = events["p"].astype(np.float32)
    x_norm = x / max(width, 1)
    y_norm = y / max(height, 1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return np.stack([x_norm, y_norm, t_norm, p], axis=1)


def events_to_batch(events, height, width, max_events, target_w=346, target_h=260):
    x_raw = events["x"].astype(np.int32)
    y_raw = events["y"].astype(np.int32)
    t_raw = events["t"].astype(np.int64)
    p_raw = events["p"]

    n = x_raw.shape[0]
    if n > max_events:
        keep = np.random.default_rng().choice(n, max_events, replace=False)
        keep.sort()
        x_raw = x_raw[keep]; y_raw = y_raw[keep]
        t_raw = t_raw[keep]; p_raw = p_raw[keep]

    # Scale camera coords → model target resolution
    sx = (target_w - 1) / max(width - 1, 1)
    sy = (target_h - 1) / max(height - 1, 1)
    x_model = np.rint(x_raw.astype(np.float32) * sx).astype(np.int32)
    y_model = np.rint(y_raw.astype(np.float32) * sy).astype(np.int32)

    t_shift = t_raw - t_raw.min()
    t_max = int(t_shift.max())
    if t_max > 0:
        t_coord = np.rint(t_shift.astype(np.float32) / float(t_max) * 8191.0).astype(np.int32)
    else:
        t_coord = np.zeros_like(t_raw, dtype=np.int32)

    x_coord = np.clip(x_model, 1, max(1, target_w - 2))
    y_coord = np.clip(y_model, 1, max(1, target_h - 2))
    t_coord = np.clip(t_coord, 1, 8188)

    norm_events = {"x": x_model.astype(np.int32), "y": y_model.astype(np.int32),
                   "t": t_raw.astype(np.int64), "p": p_raw}

    viz_events = {"x": x_raw.astype(np.int32), "y": y_raw.astype(np.int32), "p": p_raw}

    sample = {
        "ev_loc": np.stack([x_coord, y_coord, t_coord], axis=1),
        "evs_norm": build_evs_norm(norm_events, target_h, target_w),
        "seg_label": np.zeros(len(x_raw), dtype=np.float32),
        "idx": np.arange(len(x_raw), dtype=np.int64),
    }
    return sample, viz_events, x_coord, y_coord


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(events, score_map, foreground_mask, tracks, height, width, score_thresh, debug=False):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(events["x"]):
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)
        p = events["p"].astype(np.bool_)
        canvas[y[~p], x[~p]] = (255, 80, 80)
        canvas[y[p], x[p]] = (80, 200, 255)

    if debug:
        heatmap = build_score_heatmap(score_map)
        canvas = cv2.addWeighted(canvas, 0.4, heatmap, 0.6, 0)
    else:
        canvas[foreground_mask.astype(bool)] = (0, 255, 0)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
        label = f"ID{track.track_id} {track.score:.2f}"
        cv2.putText(canvas, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_status(canvas, mode, dropped_jobs, stale_results, infer_ms, tracks, pending, thresh):
    pending_str = " pending" if pending else ""
    status = (f"{mode}{pending_str} tracks:{len(tracks)} "
              f"thr:{thresh:.2f} drop:{dropped_jobs} stale:{stale_results} infer:{infer_ms:.1f}ms")
    cv2.putText(canvas, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Q / ESC to quit", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    frame_idx: int
    detections: list
    score_map: np.ndarray
    foreground_mask: np.ndarray
    infer_ms: float
    dropped_jobs: int
    score_thresh_used: float


def _pick_score_thresh(tracker, args):
    """Adaptive threshold: low when searching, high when tracking."""
    if tracker.visible_tracks(allow_missed=True):
        return args.score_thresh_high
    return args.score_thresh_low


def infer_detections(model, sample, x_raw, y_raw, args, height, width,
                     tracker, static_filter, score_prev_container, target_w, target_h):
    from dataset.basedataset import BaseDataLoader

    batch = BaseDataLoader.custom_collate([sample])
    start = time.perf_counter()

    with torch.inference_mode():
        voxel_ev = batch["voxel_ev"]
        if args.fp16:
            voxel_ev = voxel_ev.replace_feature(voxel_ev.features.half())
        preds, _ = model(voxel_ev)

    scores = preds[batch["p2v_map"].long().cuda()].squeeze().detach().cpu().numpy()
    if scores.ndim == 0:
        scores = np.array([float(scores)], dtype=np.float32)

    score_thresh = _pick_score_thresh(tracker, args)

    # Detection in model-resolution space
    detections, score_map, foreground_mask = extract_detections(
        x=x_raw, y=y_raw, scores=scores, height=target_h, width=target_w,
        score_thresh=score_thresh, min_area=args.min_area,
        morph_kernel=args.morph_kernel, dilate_iterations=args.dilate_iterations,
        bbox_margin=args.bbox_margin, min_box_size=args.min_box_size,
        max_box_area_ratio=args.max_box_area_ratio,
        nms_iou=args.nms_iou, max_detections=args.max_detections,
        static_filter=static_filter,
    )

    # EMA accumulation at model resolution
    score_prev = score_prev_container[0]
    score_map = accumulate_score_map(score_map, score_prev, args.score_ema)
    score_prev_container[0] = score_map

    # Scale score_map and detections back to camera resolution
    if width != target_w or height != target_h:
        score_map = cv2.resize(score_map, (width, height), interpolation=cv2.INTER_LINEAR)
        foreground_mask = cv2.resize(foreground_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        sc_x = width / target_w
        sc_y = height / target_h
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            det.bbox = (
                int(x1 * sc_x), int(y1 * sc_y),
                int(x2 * sc_x), int(y2 * sc_y),
            )
            det.center = (det.center[0] * sc_x, det.center[1] * sc_y)
            det.pos_key = (int(det.center[0]) // 4, int(det.center[1]) // 4)

    infer_ms = (time.perf_counter() - start) * 1000.0
    return detections, score_map, foreground_mask, infer_ms, score_thresh


# ---------------------------------------------------------------------------
# Thread-safe job queue
# ---------------------------------------------------------------------------

class LatestDetectionJob:
    def __init__(self):
        self._cond = threading.Condition()
        self._request = None; self._request_seq = 0; self._dropped = 0
        self._latest_result = None; self._result_seq = 0
        self._closed = False; self._inflight = False

    def submit(self, request):
        with self._cond:
            if self._request is not None or self._inflight:
                self._dropped += 1
            self._request = request
            self._request_seq += 1
            self._cond.notify_all()

    def wait_request(self, last_seq, timeout=0.05):
        with self._cond:
            if self._request_seq <= last_seq and not self._closed:
                self._cond.wait(timeout=timeout)
            if self._request_seq <= last_seq:
                return last_seq, None, self._closed
            request = self._request; self._request = None; self._inflight = True
            dropped = self._dropped; self._dropped = 0
            return self._request_seq, (request, dropped), self._closed

    def publish_result(self, result):
        with self._cond:
            self._latest_result = result; self._result_seq += 1
            self._inflight = False; self._cond.notify_all()

    def get_latest_result(self, last_result_seq):
        with self._cond:
            if self._result_seq <= last_result_seq:
                return last_result_seq, None
            return self._result_seq, self._latest_result

    def has_pending_work(self):
        with self._cond:
            return self._request is not None or self._inflight

    def close(self):
        with self._cond:
            self._closed = True; self._cond.notify_all()


class LatestEventsBuffer:
    def __init__(self):
        self._cond = threading.Condition()
        self._events = None; self._seq = 0; self._closed = False

    def push(self, events):
        with self._cond:
            self._events = events.copy(); self._seq += 1; self._cond.notify_all()

    def get_latest(self, last_seq):
        with self._cond:
            if self._seq <= last_seq:
                return last_seq, None, self._closed
            return self._seq, self._events.copy(), self._closed

    def close(self):
        with self._cond:
            self._closed = True; self._cond.notify_all()


# ---------------------------------------------------------------------------
# Thread workers
# ---------------------------------------------------------------------------

def detection_worker(model, job_queue, stop_event, args, height, width, tracker, static_filter,
                     target_w, target_h):
    last_seq = 0
    score_prev_container = [None]
    try:
        while not stop_event.is_set():
            seq, payload, closed = job_queue.wait_request(last_seq, timeout=0.05)
            if payload is None:
                if closed: break
                continue
            last_seq = seq
            request, dropped = payload
            detections, score_map, foreground_mask, infer_ms, thresh = infer_detections(
                model, request["sample"], request["x_raw"], request["y_raw"],
                args, height, width, tracker, static_filter, score_prev_container,
                target_w, target_h)
            job_queue.publish_result(DetectionResult(
                frame_idx=request["frame_idx"], detections=detections,
                score_map=score_map, foreground_mask=foreground_mask,
                infer_ms=infer_ms, dropped_jobs=dropped,
                score_thresh_used=thresh))
    finally:
        job_queue.close()


def capture_worker(iterator, events_buffer, stop_event):
    try:
        for events in iterator:
            if stop_event.is_set(): break
            events_buffer.push(events)
    finally:
        events_buffer.close()


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

def install_keyboard_handler(window, stop_event, job_queue, events_buffer):
    def cb(key, scancode, action, mods):
        del scancode, mods
        if action != UIAction.RELEASE: return
        if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
            stop_event.set(); job_queue.close(); events_buffer.close()
    window.set_keyboard_callback(cb)


def install_stop_keyboard_handler(window, stop_event):
    def cb(key, scancode, action, mods):
        del scancode, mods
        if action != UIAction.RELEASE: return
        if key in (UIKeyEvent.KEY_ESCAPE, UIKeyEvent.KEY_Q):
            stop_event.set()
    window.set_keyboard_callback(cb)


# ---------------------------------------------------------------------------
# Resolution check
# ---------------------------------------------------------------------------

def check_resolution(camera_hw, cfg_res):
    """Warn if camera resolution differs from the config the model expects."""
    c_h, c_w = camera_hw
    cfg_w, cfg_h = cfg_res
    if c_w != cfg_w or c_h != cfg_h:
        print(f"[WARN] Camera resolution ({c_w}x{c_h}) != config res ({cfg_w}x{cfg_h}).")
        print(f"       Events will be clipped to config resolution. "
              f"Consider updating configs/evisseg_evuav.yaml DATA:res to [{c_w},{c_h}]")


# ---------------------------------------------------------------------------
# Offline (file) mode
# ---------------------------------------------------------------------------

def run_offline_tracking(args, model, tracker, static_filter):
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.window_us)
    height, width = mv_iterator.get_size()
    target_w, target_h = (int(x) for x in args.target_res.split("x"))
    check_resolution((height, width), cfg_res=(target_w, target_h))

    # Noise filters (Metavision SDK)
    act_filter = (ActivityNoiseFilterAlgorithm(width, height, args.activity_filter_us)
                  if args.activity_filter_us > 0 else None)
    trail_filter = (TrailFilterAlgorithm(width, height, args.trail_filter_us)
                    if args.trail_filter_us > 0 else None)
    filter_buf = (ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
                  if act_filter is not None or trail_filter is not None else None)

    blank_score_map = np.zeros((height, width), dtype=np.float32)
    frame_period = 1.0 / max(args.display_fps, 1)

    stop_event = threading.Event()
    latest_score_map = blank_score_map
    latest_foreground_mask = np.zeros((height, width), dtype=np.uint8)
    latest_infer_ms = 0.0
    latest_thresh = args.score_thresh_low
    frame_idx = 0
    current_tracks = []
    current_mode = "offline-idle"
    score_prev_container = [None]  # mutable, updated inside infer_detections

    try:
        with MTWindow(title="EV-UAV Tracking v3 (offline)", width=width,
                       height=height, mode=BaseWindow.RenderMode.BGR) as window:
            install_stop_keyboard_handler(window, stop_event)

            for events in mv_iterator:
                if stop_event.is_set(): break
                loop_start = time.perf_counter()
                EventLoop.poll_and_dispatch()
                if window.should_close():
                    stop_event.set(); break

                frame_idx += 1
                # Apply Metavision SDK noise filters before model
                events = filter_events(events, act_filter, trail_filter, filter_buf)
                if len(events) == 0:
                    canvas = render_frame(empty_viz_events(), blank_score_map, latest_foreground_mask,
                                          current_tracks, height, width,
                                          latest_thresh, debug=args.debug)
                    canvas = draw_status(canvas, "offline-empty", 0, 0,
                                         latest_infer_ms, current_tracks, False,
                                         latest_thresh)
                    window.show_async(canvas)
                else:
                    sample, viz_events, x_raw, y_raw = events_to_batch(
                        events, height, width, args.max_events, target_w, target_h)
                    should_detect = (
                        frame_idx % max(args.detect_every, 1) == 1
                        or not tracker.visible_tracks(allow_missed=True)
                    )
                    if should_detect:
                        detections, score_map, foreground_mask, latest_infer_ms, latest_thresh = infer_detections(
                            model, sample, x_raw, y_raw, args, height, width,
                            tracker, static_filter, score_prev_container, target_w, target_h)
                        latest_score_map = score_map
                        latest_foreground_mask = foreground_mask
                        current_tracks = tracker.update(detections)
                        current_mode = "offline-detect"
                    else:
                        current_tracks = tracker.predict_only()
                        current_mode = "offline-predict"

                    score_map = latest_score_map if current_mode == "offline-detect" else blank_score_map
                    foreground_mask = latest_foreground_mask if current_mode == "offline-detect" else np.zeros((height, width), dtype=np.uint8)
                    canvas = render_frame(viz_events, score_map, foreground_mask, current_tracks,
                                          height, width, latest_thresh,
                                          debug=args.debug)
                    canvas = draw_status(canvas, current_mode, 0, 0,
                                         latest_infer_ms, current_tracks, False,
                                         latest_thresh)
                    window.show_async(canvas)

                remaining = frame_period - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
                EventLoop.poll_and_dispatch()
                if window.should_close():
                    stop_event.set(); break
    except KeyboardInterrupt:
        stop_event.set()


# ---------------------------------------------------------------------------
# Live camera mode
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    checkpoint_path = args.checkpoint or cfg.model_path

    model = build_model(cfg, args.config, checkpoint_path, use_fp16=args.fp16)
    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
        min_track_score=args.min_track_score,
        static_thresh=args.static_thresh,
    )
    static_filter = StaticFilter(static_thresh=args.static_thresh)

    if args.input_path:
        run_offline_tracking(args, model, tracker, static_filter)
        return

    # --- live camera -------------------------------------------------------
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.window_us)
    height, width = mv_iterator.get_size()
    target_w, target_h = (int(x) for x in args.target_res.split("x"))
    check_resolution((height, width), cfg_res=(target_w, target_h))

    # Noise filters (Metavision SDK)
    act_filter = (ActivityNoiseFilterAlgorithm(width, height, args.activity_filter_us)
                  if args.activity_filter_us > 0 else None)
    trail_filter = (TrailFilterAlgorithm(width, height, args.trail_filter_us)
                    if args.trail_filter_us > 0 else None)
    filter_buf = (ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
                  if act_filter is not None or trail_filter is not None else None)

    print(f"Camera: {width}x{height} -> model: {target_w}x{target_h}  |  window: {args.window_us} us  |  "
          f"detect-every: {args.detect_every}  |  max-events: {args.max_events}  |  "
          f"FP16: {args.fp16}  |  debug: {args.debug}")
    print(f"Noise filter: activity={args.activity_filter_us}us trail={args.trail_filter_us}us")
    print(f"Adaptive threshold: search={args.score_thresh_low}  "
          f"track={args.score_thresh_high}  |  EMA: {args.score_ema}")
    print(f"Static filter: {args.static_thresh} frames  |  "
          f"min-hits: {args.min_hits}  |  min-track-score: {args.min_track_score}")

    blank_score_map = np.zeros((height, width), dtype=np.float32)
    stop_event = threading.Event()
    events_buffer = LatestEventsBuffer()
    job_queue = LatestDetectionJob()

    capture_thread = threading.Thread(
        target=capture_worker, args=(mv_iterator, events_buffer, stop_event),
        daemon=False)
    worker_thread = threading.Thread(
        target=detection_worker,
        args=(model, job_queue, stop_event, args, height, width, tracker, static_filter,
              target_w, target_h),
        daemon=False)
    capture_thread.start()
    worker_thread.start()

    title = "EV-UAV Tracking v3" if not args.debug else "EV-UAV Tracking v3 [DEBUG]"
    frame_period = 1.0 / max(args.display_fps, 1)
    last_result_seq = 0
    latest_score_map = blank_score_map
    latest_foreground_mask = np.zeros((height, width), dtype=np.uint8)
    latest_infer_ms = 0.0
    latest_dropped_jobs = 0
    stale_results = 0
    frame_idx = 0
    last_events_seq = 0
    current_viz_events = empty_viz_events()
    current_tracks = []
    current_mode = "idle"
    latest_thresh = args.score_thresh_low

    try:
        with MTWindow(title=title, width=width, height=height,
                       mode=BaseWindow.RenderMode.BGR) as window:
            install_keyboard_handler(window, stop_event, job_queue, events_buffer)

            while not stop_event.is_set():
                loop_start = time.perf_counter()
                EventLoop.poll_and_dispatch()
                if window.should_close():
                    stop_event.set(); break

                events_seq, events, closed = events_buffer.get_latest(last_events_seq)
                has_new_events = events is not None
                if has_new_events:
                    last_events_seq = events_seq
                    frame_idx += 1
                    events = filter_events(events, act_filter, trail_filter, filter_buf)
                    current_viz_events = empty_viz_events() if len(events) == 0 else None

                if closed and not has_new_events:
                    break

                # --- consume detection result -------------------------------
                result_seq, result = job_queue.get_latest_result(last_result_seq)
                if result is not None:
                    last_result_seq = result_seq
                    if frame_idx - result.frame_idx <= max(args.max_result_lag, 0):
                        latest_score_map = result.score_map
                        latest_foreground_mask = result.foreground_mask
                        latest_infer_ms = result.infer_ms
                        latest_dropped_jobs = result.dropped_jobs
                        latest_thresh = result.score_thresh_used
                        current_tracks = tracker.update(result.detections)
                        current_mode = "detect"
                    else:
                        stale_results += 1
                        if has_new_events:
                            current_tracks = tracker.predict_only()
                        current_mode = "predict-stale"
                elif has_new_events:
                    current_tracks = tracker.predict_only()
                    current_mode = "predict"

                # --- render -------------------------------------------------
                if not has_new_events:
                    score_map = latest_score_map if current_mode == "detect" else blank_score_map
                    foreground_mask = latest_foreground_mask if current_mode == "detect" else np.zeros((height, width), dtype=np.uint8)
                    canvas = render_frame(current_viz_events, score_map, foreground_mask,
                                          current_tracks, height, width,
                                          latest_thresh, debug=args.debug)
                    canvas = draw_status(canvas, current_mode + "-hold",
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work(),
                                         latest_thresh)
                    window.show_async(canvas)
                elif len(events) == 0:
                    current_viz_events = empty_viz_events()
                    canvas = render_frame(empty_viz_events(), blank_score_map, np.zeros((height, width), dtype=np.uint8),
                                          current_tracks, height, width,
                                          latest_thresh, debug=args.debug)
                    canvas = draw_status(canvas, current_mode + "-empty",
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work(),
                                         latest_thresh)
                    window.show_async(canvas)
                else:
                    sample, viz_events, x_raw, y_raw = events_to_batch(
                        events, height, width, args.max_events, target_w, target_h)
                    current_viz_events = viz_events
                    should_detect = (
                        frame_idx % max(args.detect_every, 1) == 1
                        or not tracker.visible_tracks(allow_missed=True)
                    )
                    if should_detect:
                        job_queue.submit({
                            "frame_idx": frame_idx,
                            "sample": sample,
                            "x_raw": x_raw,
                            "y_raw": y_raw,
                        })

                    score_map = latest_score_map if current_mode == "detect" else blank_score_map
                    foreground_mask = latest_foreground_mask if current_mode == "detect" else np.zeros((height, width), dtype=np.uint8)
                    canvas = render_frame(viz_events, score_map, foreground_mask, current_tracks,
                                          height, width, latest_thresh,
                                          debug=args.debug)
                    canvas = draw_status(canvas, current_mode,
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work(),
                                         latest_thresh)
                    window.show_async(canvas)

                remaining = frame_period - (time.perf_counter() - loop_start)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        job_queue.close()
        events_buffer.close()
        capture_thread.join(timeout=2.0)
        worker_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
