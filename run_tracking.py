import argparse
import os
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml
from metavision_core.event_io import EventsIterator
from metavision_sdk_ui import BaseWindow, EventLoop, MTWindow

from dataset.basedataset import BaseDataLoader
from model.evspsegnet import evspsegnet
from tracking.postprocess import extract_detections
from tracking.tracker import MultiObjectTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Non-invasive EV-UAV tracking runner")
    parser.add_argument("--config", default="configs/evisseg_evuav.yaml", type=str, help="YAML config path")
    parser.add_argument("--checkpoint", default="", type=str, help="Model checkpoint override")
    parser.add_argument("--input-path", default="", type=str, help="Optional raw event file path")
    parser.add_argument("--window-us", default=1000, type=int, help="Event accumulation window in microseconds")
    parser.add_argument("--score-thresh", default=0.35, type=float, help="Foreground score threshold")
    parser.add_argument("--min-area", default=3, type=int, help="Minimum connected-component area")
    parser.add_argument("--max-distance", default=20.0, type=float, help="Association distance threshold")
    parser.add_argument("--max-missed", default=5, type=int, help="How many windows a track can disappear")
    parser.add_argument("--min-hits", default=2, type=int, help="Track confirmation threshold")
    return parser.parse_args()


def load_cfg(config_path: str) -> SimpleNamespace:
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.load(file, Loader=yaml.CLoader)

    flat = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat)


def build_evs_norm(events, height, width):
    x = events["x"].astype(np.float32)
    y = events["y"].astype(np.float32)
    t = events["t"].astype(np.float32)
    p = events["p"].astype(np.float32)

    x_norm = x / max(width, 1)
    y_norm = y / max(height, 1)
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)
    return np.stack([x_norm, y_norm, t_norm, p], axis=1)


def build_model(cfg, checkpoint_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("This tracking runner requires CUDA because spconv/HAIS voxelization is GPU-only here.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = evspsegnet(cfg).cuda().eval()
    state_dict = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state_dict)
    return model


def events_to_batch(events, height, width):
    x_raw = events["x"].astype(np.int32)
    y_raw = events["y"].astype(np.int32)
    t_raw = events["t"].astype(np.int64)
    z_raw = np.zeros_like(x_raw)

    sample = {
        "ev_loc": np.stack([x_raw, y_raw, z_raw, t_raw], axis=1),
        "evs_norm": build_evs_norm(events, height, width),
        "seg_label": np.zeros(len(x_raw), dtype=np.float32),
        "idx": np.arange(len(x_raw), dtype=np.int64),
    }
    return sample, x_raw, y_raw


def render_frame(events, score_map, tracks, height, width, score_thresh):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(events):
        x = events["x"].astype(np.int32)
        y = events["y"].astype(np.int32)
        p = events["p"].astype(np.bool_)

        canvas[y[~p], x[~p]] = (255, 80, 80)
        canvas[y[p], x[p]] = (80, 200, 255)

    active = score_map >= score_thresh
    canvas[active] = (0, 255, 0)

    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1)
        label = f"ID {track.track_id} {track.score:.2f}"
        text_origin = (x1, max(12, y1 - 4))
        cv2.putText(canvas, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main():
    args = parse_args()
    cfg = load_cfg(args.config)
    checkpoint_path = args.checkpoint or cfg.model_path

    model = build_model(cfg, checkpoint_path)
    tracker = MultiObjectTracker(
        max_distance=args.max_distance,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
    )

    iterator_kwargs = {"delta_t": args.window_us}
    if args.input_path:
        iterator_kwargs["input_path"] = args.input_path
    mv_iterator = EventsIterator(**iterator_kwargs)
    height, width = mv_iterator.get_size()

    title = "EV-UAV Tracking"
    with MTWindow(title=title, width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        for events in mv_iterator:
            EventLoop.poll_and_dispatch()

            if len(events) == 0:
                tracks = tracker.update([])
                window.show_async(np.zeros((height, width, 3), dtype=np.uint8))
                if window.should_close():
                    break
                continue

            sample, x_raw, y_raw = events_to_batch(events, height, width)
            batch = BaseDataLoader.custom_collate([sample])

            with torch.no_grad():
                preds, _ = model(batch["voxel_ev"])
            scores = preds[batch["p2v_map"].long().cuda()].squeeze().detach().cpu().numpy()

            detections, score_map = extract_detections(
                x=x_raw,
                y=y_raw,
                scores=scores,
                height=height,
                width=width,
                score_thresh=args.score_thresh,
                min_area=args.min_area,
            )
            tracks = tracker.update(detections)
            canvas = render_frame(events, score_map, tracks, height, width, args.score_thresh)
            window.show_async(canvas)

            if window.should_close():
                break


if __name__ == "__main__":
    main()
