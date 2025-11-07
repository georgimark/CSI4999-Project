import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt weights")
    ap.add_argument("--source", required=True, help="Path to input video (.mov/.mp4/...)")
    ap.add_argument("--out", default="outputs/annotated.mp4", help="Output video path (.mp4 recommended)")
    ap.add_argument("--device", default="0", help="'0' for first GPU or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame for speed (1 = all)")
    return ap.parse_args()

def main():
    args = parse_args()
    inp = Path(args.source)
    if not inp.exists():
        raise FileNotFoundError(f"Video not found: {inp}")

    cap = cv2.VideoCapture(str(inp))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {inp}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
    cap.release()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    model = YOLO(args.model)

    results_iter = model.predict(
        source=str(inp),
        stream=True,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=args.half,
        vid_stride=args.vid_stride,
        save=False
    )

    pbar = tqdm(results_iter, total=(frame_count // max(args.vid_stride,1)) if frame_count else None, desc="Annotating")
    for res in pbar:
        frame_annotated = res.plot()
        if frame_annotated.shape[1] != w or frame_annotated.shape[0] != h:
            frame_annotated = cv2.resize(frame_annotated, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(frame_annotated)

    writer.release()
    print(f"[OK] Wrote annotated video → {out_path}")

if __name__ == "__main__":
    main()
