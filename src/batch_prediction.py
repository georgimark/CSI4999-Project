#!/usr/bin/env python3
"""
Usage (example):
  python batch_prediction.py
    --model "runs/detect/bdd11s_base/weights/best.pt"
    --source "../data/sample_images"
    --out "../outputs/predictions" ^
    --device 0 --imgsz 1280 --conf 0.25 --save-txt --export-csv
"""

import argparse
from pathlib import Path
import csv

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics not found. Install with: pip install -U ultralytics") from e


def parse_args():
    ap = argparse.ArgumentParser(description="Run YOLOv11 .pt model on a folder of images and save outputs.")
    ap.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    ap.add_argument("--source", type=str, required=True, help="Folder with input images")
    ap.add_argument("--out", type=str, required=True, help="Output folder (will be created)")
    ap.add_argument("--device", type=str, default="0", help="Device: '0' for first GPU, or 'cpu'")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--half", action="store_true", help="Use FP16 (speeds up on supported GPUs)")
    ap.add_argument("--save-txt", action="store_true", help="Also save YOLO-format .txt labels")
    ap.add_argument("--save-conf", action="store_true", help="Include confidences in .txt labels")
    ap.add_argument("--export-csv", action="store_true", help="Write predictions.csv with all detections")
    ap.add_argument("--name", type=str, default="pred", help="Run name under the output folder")
    return ap.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    out_root = Path(args.out)
    run_name = args.name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found or not a directory: {source_dir}")

    out_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_path.as_posix())

    results = model.predict(
        source=source_dir.as_posix(),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        half=args.half,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=out_root.as_posix(),
        name=run_name,
        exist_ok=True,
        stream=False
    )

    out_dir = out_root / run_name
    print(f"[OK] Saved annotated outputs to: {out_dir}")

    if args.export_csv:
        csv_path = out_dir / "predictions.csv"
        class_names = model.names if hasattr(model, "names") else {}

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image",
                "class_id",
                "class_name",
                "confidence",
                "x1", "y1", "x2", "y2"
            ])

            for res in results:
                p = Path(res.path)
                if res.boxes is None or len(res.boxes) == 0:
                    continue
                boxes = res.boxes
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                    cname = class_names.get(k, str(k))
                    writer.writerow([p.name, k, cname, float(c), float(x1), float(y1), float(x2), float(y2)])

        print(f"[OK] Wrote CSV summary: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
