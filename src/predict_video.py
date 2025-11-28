from pathlib import Path
import argparse
import sys
import math
import cv2
from ultralytics import YOLO
from tqdm import tqdm


VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}  # extend as needed


def ensure_writer(path: Path, fps: float, w: int, h: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {path}")
    return writer


def probe_video_dims_fps(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = int(frame_count_prop) if frame_count_prop and frame_count_prop > 0 else None
    cap.release()
    return fps, w, h, frame_count


def annotate_one_video(model: YOLO, inp: Path, out_path: Path, device: str, imgsz: int, conf: float, half: bool, vid_stride: int) -> Path:
    fps, w, h, frame_count = probe_video_dims_fps(inp)
    writer = ensure_writer(out_path, fps, w, h)

    results_iter = model.predict(
        source=str(inp),
        stream=True,
        device=device,
        imgsz=imgsz,
        conf=conf,
        half=half,
        vid_stride=vid_stride,
        save=False
    )

    total_frames = None
    if frame_count:
        total_frames = math.ceil(frame_count / max(vid_stride, 1))

    pbar = tqdm(results_iter, total=total_frames, desc=f"Annotating {inp.name}")
    frames_written = 0
    for res in pbar:
        frame_annotated = res.plot()
        if frame_annotated.shape[1] != w or frame_annotated.shape[0] != h:
            frame_annotated = cv2.resize(frame_annotated, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame_annotated)
        frames_written += 1

    writer.release()
    print(f"[OK] {inp.name} → {out_path} ({frames_written} frames)")
    return out_path


def gather_inputs(input_path: Path):
    # return a sorted list of video files from a file or directory input
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in VIDEO_EXTS else []
    if input_path.is_dir():
        return sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
    return []


def main():
    ap = argparse.ArgumentParser(description="Batch annotate videos with YOLO and save MP4 outputs.")
    ap.add_argument("--input", required=True, help="Path to a directory OR a single video file")
    ap.add_argument("--model", default="../runs/detect/bdd11s_base/weights/best.pt", help="Path to .pt weights")
    ap.add_argument("--out-dir", default="../outputs", help="Output directory for annotated videos")
    ap.add_argument("--device", default="cpu", help="'0' for first GPU or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame for speed (1 = all)")
    ap.add_argument("--overwrite", action="store_true", help="If set, redo outputs even if they already exist")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERR] --input does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.model)
    out_root = Path(args.out_dir)

    inputs = gather_inputs(input_path)
    if not inputs:
        print(f"[WARN] No video files found in: {input_path}")
        return

    successes = 0
    failures = []

    # For directory inputs, preserve relative structure under out_root.
    # For single-file inputs, write to out_root/<stem>.mp4
    for vid in inputs:
        try:
            if input_path.is_dir():
                rel = vid.relative_to(input_path)
                rel_no_ext = rel.with_suffix("")
                out_path = out_root / rel_no_ext.with_suffix(".mp4")
            else:
                out_path = out_root / (vid.stem + ".mp4")

            if out_path.exists() and not args.overwrite:
                print(f"[SKIP] Output exists: {out_path}")
                successes += 1  # treat as done
                continue

            annotate_one_video(
                model=model,
                inp=vid,
                out_path=out_path,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                half=args.half,
                vid_stride=args.vid_stride
            )
            successes += 1
        except Exception as e:
            failures.append((vid, str(e)))
            print(f"[ERR] {vid} failed: {e}", file=sys.stderr)

    print(f"[DONE] Processing complete. OK: {successes}, Failed: {len(failures)}")
    if failures:
        print("Failed files:")
        for f, msg in failures:
            print(f"  - {f}: {msg}")


if __name__ == "__main__":
    main()
