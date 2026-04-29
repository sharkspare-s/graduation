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
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow, UIAction, UIKeyEvent

from tracking.postprocess_v2 import extract_detections
from tracking.tracker_v2 import MultiObjectTracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="EV-UAV tracking v2 (Jetson-optimised)")
    parser.add_argument("--config", default="configs/evisseg_evuav.yaml", type=str)
    parser.add_argument("--checkpoint", default="", type=str, help="Model checkpoint override")
    parser.add_argument(
        "--input-path", default="", type=str,
        help="Optional raw event file; leave empty to open the first camera",
    )

    # --- timing ------------------------------------------------------------
    parser.add_argument("--window-us", default=30000, type=int,
                        help="Event accumulation window [us]")
    parser.add_argument("--detect-every", default=2, type=int,
                        help="Run detection every N windows (others: predict-only)")
    parser.add_argument("--max-result-lag", default=2, type=int,
                        help="Drop results older than this many windows")
    parser.add_argument("--display-fps", default=30, type=int)

    # --- data --------------------------------------------------------------
    parser.add_argument("--max-events", default=15000, type=int,
                        help="Cap events per window")

    # --- detection ---------------------------------------------------------
    parser.add_argument("--score-thresh", default=0.30, type=float)
    parser.add_argument("--min-area", default=5, type=int)
    parser.add_argument("--morph-kernel", default=3, type=int,
                        help="Odd morphology kernel size (0 to disable)")
    parser.add_argument("--dilate-iterations", default=1, type=int)
    parser.add_argument("--bbox-margin", default=4, type=int)
    parser.add_argument("--min-box-size", default=3, type=int)
    parser.add_argument("--max-box-area-ratio", default=0.15, type=float)
    parser.add_argument("--nms-iou", default=0.3, type=float)
    parser.add_argument("--max-detections", default=10, type=int)

    # --- tracker -----------------------------------------------------------
    parser.add_argument("--max-distance", default=30.0, type=float)
    parser.add_argument("--max-missed", default=8, type=int)
    parser.add_argument("--min-hits", default=2, type=int)

    # --- optimisations -----------------------------------------------------
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 inference (Jetson: ~1.5-2x faster if spconv supports it)")

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


def build_evs_norm(events, height, width):
    x = events["x"].astype(np.float32)
    y = events["y"].astype(np.float32)
    t = events["t"].astype(np.float32)
    p = events["p"].astype(np.float32)
    x_norm = x / max(width, 1)
    y_norm = y / max(height, 1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return np.stack([x_norm, y_norm, t_norm, p], axis=1)


def events_to_batch(events, height, width, max_events):
    x_raw = events["x"].astype(np.int32)
    y_raw = events["y"].astype(np.int32)
    t_raw = events["t"].astype(np.int64)
    p_raw = events["p"]

    n = x_raw.shape[0]
    if n > max_events:
        keep = np.linspace(0, n - 1, max_events, dtype=np.int64)
        x_raw = x_raw[keep]; y_raw = y_raw[keep]
        t_raw = t_raw[keep]; p_raw = p_raw[keep]

    t_shift = t_raw - t_raw.min()
    t_max = int(t_shift.max())
    if t_max > 0:
        t_coord = np.rint(t_shift.astype(np.float32) / float(t_max) * 8191.0).astype(np.int32)
    else:
        t_coord = np.zeros_like(t_raw, dtype=np.int32)

    x_coord = np.clip(x_raw, 1, max(1, width - 2))
    y_coord = np.clip(y_raw, 1, max(1, height - 2))
    t_coord = np.clip(t_coord, 1, 8188)

    norm_events = {"x": x_raw.astype(np.int32), "y": y_raw.astype(np.int32),
                   "t": t_raw.astype(np.int64), "p": p_raw}

    viz_events = {"x": x_raw.astype(np.int32), "y": y_raw.astype(np.int32), "p": p_raw}

    sample = {
        "ev_loc": np.stack([x_coord, y_coord, t_coord], axis=1),
        "evs_norm": build_evs_norm(norm_events, height, width),
        "seg_label": np.zeros(len(x_raw), dtype=np.float32),
        "idx": np.arange(len(x_raw), dtype=np.int64),
    }
    return sample, viz_events, x_coord, y_coord


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(events, score_map, tracks, height, width, score_thresh):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if len(events["x"]):
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)
        p = events["p"].astype(np.bool_)
        canvas[y[~p], x[~p]] = (255, 80, 80)   # OFF events – red
        canvas[y[p], x[p]] = (80, 200, 255)     # ON events – yellow

    active = score_map >= score_thresh
    canvas[active] = (0, 255, 0)                # foreground – green

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
        label = f"ID{track.track_id} {track.score:.2f}"
        cv2.putText(canvas, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def draw_status(canvas, mode, dropped_jobs, stale_results, infer_ms, tracks, pending):
    pending_str = " pending" if pending else ""
    status = (f"{mode}{pending_str} tracks:{len(tracks)} "
              f"drop:{dropped_jobs} stale:{stale_results} infer:{infer_ms:.1f}ms")
    cv2.putText(canvas, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Q / ESC to quit", (8, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    frame_idx: int
    detections: list
    score_map: np.ndarray
    infer_ms: float
    dropped_jobs: int


def infer_detections(model, sample, x_raw, y_raw, args, height, width):
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

    detections, score_map = extract_detections(
        x=x_raw, y=y_raw, scores=scores, height=height, width=width,
        score_thresh=args.score_thresh, min_area=args.min_area,
        morph_kernel=args.morph_kernel, dilate_iterations=args.dilate_iterations,
        bbox_margin=args.bbox_margin, min_box_size=args.min_box_size,
        max_box_area_ratio=args.max_box_area_ratio,
        nms_iou=args.nms_iou, max_detections=args.max_detections,
    )
    infer_ms = (time.perf_counter() - start) * 1000.0
    return detections, score_map, infer_ms


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

def detection_worker(model, job_queue, stop_event, args, height, width):
    last_seq = 0
    try:
        while not stop_event.is_set():
            seq, payload, closed = job_queue.wait_request(last_seq, timeout=0.05)
            if payload is None:
                if closed: break
                continue
            last_seq = seq
            request, dropped = payload
            detections, score_map, infer_ms = infer_detections(
                model, request["sample"], request["x_raw"], request["y_raw"],
                args, height, width)
            job_queue.publish_result(DetectionResult(
                frame_idx=request["frame_idx"], detections=detections,
                score_map=score_map, infer_ms=infer_ms, dropped_jobs=dropped))
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
# Offline (file) mode
# ---------------------------------------------------------------------------

def run_offline_tracking(args, model, tracker):
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.window_us)
    height, width = mv_iterator.get_size()
    blank_score_map = np.zeros((height, width), dtype=np.float32)
    frame_period = 1.0 / max(args.display_fps, 1)

    stop_event = threading.Event()
    latest_score_map = blank_score_map
    latest_infer_ms = 0.0
    frame_idx = 0
    current_tracks = []
    current_mode = "offline-idle"

    try:
        with MTWindow(title="EV-UAV Tracking v2 (offline)", width=width,
                       height=height, mode=BaseWindow.RenderMode.BGR) as window:
            install_stop_keyboard_handler(window, stop_event)

            for events in mv_iterator:
                if stop_event.is_set(): break
                loop_start = time.perf_counter()
                EventLoop.poll_and_dispatch()
                if window.should_close():
                    stop_event.set(); break

                frame_idx += 1
                if len(events) == 0:
                    canvas = render_frame(empty_viz_events(), blank_score_map,
                                          current_tracks, height, width, args.score_thresh)
                    canvas = draw_status(canvas, "offline-empty", 0, 0,
                                         latest_infer_ms, current_tracks, False)
                    window.show_async(canvas)
                else:
                    sample, viz_events, x_raw, y_raw = events_to_batch(
                        events, height, width, args.max_events)
                    should_detect = (
                        frame_idx % max(args.detect_every, 1) == 1
                        or not tracker.visible_tracks(allow_missed=True)
                    )
                    if should_detect:
                        detections, latest_score_map, latest_infer_ms = infer_detections(
                            model, sample, x_raw, y_raw, args, height, width)
                        current_tracks = tracker.update(detections)
                        current_mode = "offline-detect"
                    else:
                        current_tracks = tracker.predict_only()
                        current_mode = "offline-predict"

                    score_map = latest_score_map if current_mode == "offline-detect" else blank_score_map
                    canvas = render_frame(viz_events, score_map, current_tracks,
                                          height, width, args.score_thresh)
                    canvas = draw_status(canvas, current_mode, 0, 0,
                                         latest_infer_ms, current_tracks, False)
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
    )

    if args.input_path:
        run_offline_tracking(args, model, tracker)
        return

    # --- live camera -------------------------------------------------------
    mv_iterator = EventsIterator(input_path=args.input_path, delta_t=args.window_us)
    height, width = mv_iterator.get_size()
    print(f"Camera resolution: {width}x{height}")
    print(f"Window: {args.window_us} us  |  detect every: {args.detect_every}  |  "
          f"max events: {args.max_events}  |  FP16: {args.fp16}")

    blank_score_map = np.zeros((height, width), dtype=np.float32)
    stop_event = threading.Event()
    events_buffer = LatestEventsBuffer()
    job_queue = LatestDetectionJob()

    capture_thread = threading.Thread(
        target=capture_worker, args=(mv_iterator, events_buffer, stop_event),
        daemon=False)
    worker_thread = threading.Thread(
        target=detection_worker,
        args=(model, job_queue, stop_event, args, height, width),
        daemon=False)
    capture_thread.start()
    worker_thread.start()

    title = "EV-UAV Tracking v2"
    frame_period = 1.0 / max(args.display_fps, 1)
    last_result_seq = 0
    latest_score_map = blank_score_map
    latest_infer_ms = 0.0
    latest_dropped_jobs = 0
    stale_results = 0
    frame_idx = 0
    last_events_seq = 0
    current_viz_events = empty_viz_events()
    current_tracks = []
    current_mode = "idle"

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
                    current_viz_events = (empty_viz_events() if len(events) == 0
                                          else None)  # will be set below

                if closed and not has_new_events:
                    break

                # --- consume detection result ---------------------------------
                result_seq, result = job_queue.get_latest_result(last_result_seq)
                if result is not None:
                    last_result_seq = result_seq
                    if frame_idx - result.frame_idx <= max(args.max_result_lag, 0):
                        latest_score_map = result.score_map
                        latest_infer_ms = result.infer_ms
                        latest_dropped_jobs = result.dropped_jobs
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

                # --- render ----------------------------------------------------
                if not has_new_events:
                    score_map = latest_score_map if current_mode == "detect" else blank_score_map
                    canvas = render_frame(current_viz_events, score_map,
                                          current_tracks, height, width, args.score_thresh)
                    canvas = draw_status(canvas, current_mode + "-hold",
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work())
                    window.show_async(canvas)
                elif len(events) == 0:
                    current_viz_events = empty_viz_events()
                    canvas = render_frame(empty_viz_events(), blank_score_map,
                                          current_tracks, height, width, args.score_thresh)
                    canvas = draw_status(canvas, current_mode + "-empty",
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work())
                    window.show_async(canvas)
                else:
                    sample, viz_events, x_raw, y_raw = events_to_batch(
                        events, height, width, args.max_events)
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
                    canvas = render_frame(viz_events, score_map, current_tracks,
                                          height, width, args.score_thresh)
                    canvas = draw_status(canvas, current_mode,
                                         latest_dropped_jobs, stale_results,
                                         latest_infer_ms, current_tracks,
                                         job_queue.has_pending_work())
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
