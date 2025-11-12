from pathlib import Path
import argparse
import sys
import cv2
from ultralytics import YOLO
from tqdm import tqdm


SOURCE_DIR = Path(r"../videos")  # <-- change me

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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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
    pbar = tqdm(results_iter, total=(frame_count // max(vid_stride,1)) if frame_count else None, desc=f"Annotating {inp.name}")
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../runs/detect/bdd11s_base/weights/best.pt", help="Path to .pt weights")
    ap.add_argument("--out-dir", default="../outputs", help="Output directory for annotated videos")
    ap.add_argument("--device", default="0", help="'0' for first GPU or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame for speed (1 = all)")
    args = ap.parse_args()

    if not SOURCE_DIR.exists() or not SOURCE_DIR.is_dir():
        print(f"[ERR] SOURCE_DIR does not exist or is not a directory: {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.model)

    movs = sorted([p for p in SOURCE_DIR.rglob("*") if p.is_file() and p.suffix.lower() == ".mov"])
    if not movs:
        print(f"[WARN] No .mov files found in: {SOURCE_DIR}")
        return

    out_dir = Path(args.out_dir)
    successes = 0
    failures = []

    for mov in movs:
        rel = mov.relative_to(SOURCE_DIR)
        rel_no_ext = rel.with_suffix("")
        out_path = out_dir / rel_no_ext.with_suffix(".mp4")
        try:
            annotate_one_video(model, mov, out_path, args.device, args.imgsz, args.conf, args.half, args.vid_stride)
            successes += 1
        except Exception as e:
            failures.append((mov, str(e)))
            print(f"[ERR] {mov} failed: {e}", file=sys.stderr)

    print(f"[DONE] Folder processing complete. OK: {successes}, Failed: {len(failures)}")
    if failures:
        print("Failed files:")
        for f, msg in failures:
            print(f"  - {f}: {msg}")

if __name__ == "__main__":
    main()
