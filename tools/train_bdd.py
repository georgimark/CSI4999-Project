#!/usr/bin/env python3
"""
Run:
  python train_bdd11.py --root ../data/bdd100k_yolo_converted --model yolo11s.pt --epochs 80 --imgsz 1280
  - Start with yolo11s.pt or yolo11n.pt depending on GPU.
  - Increase --tune-iterations and --tune-epochs for deeper training, may overfit
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, List

try:
    from ultralytics import YOLO
except Exception as e:
    print("\n[ERROR] Ultralytics not found. Install it first:\n  pip install -U ultralytics\n")
    raise

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

DEFAULT_CLASS_NAMES = [
    "person",
    "car",
    "bus",
    "truck",
    "bicycle",
    "motorcycle",
    "traffic light",
    "traffic sign",
]

# ---------------------- Helpers -------------------------

def write_data_yaml(root: Path, names: List[str]) -> Path:
    """Create data.yaml if it doesn't exist."""
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        print(f"[INFO] Found existing {yaml_path}")
        return yaml_path

    yaml_text = (
        f"path: {root.as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
    )
    for i, n in enumerate(names):
        yaml_text += f"  {i}: {n}\n"

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"[OK] Wrote {yaml_path}")
    return yaml_path


def sanity_check_dataset(root: Path) -> None:
    """Basic checks for YOLO folder structure and label/image pairing."""
    img_train = root / "images" / "train"
    img_val = root / "images" / "val"
    lab_train = root / "labels" / "train"
    lab_val = root / "labels" / "val"

    for p in [img_train, img_val, lab_train, lab_val]:
        if not p.exists():
            raise FileNotFoundError(f"Expected path missing: {p}")

    def count_files(d: Path, exts={".jpg", ".jpeg", ".png"}):
        return [p for p in d.rglob("*") if p.suffix.lower() in exts]

    train_imgs = count_files(img_train)
    val_imgs = count_files(img_val)

    print(f"[CHECK] Train images: {len(train_imgs)} | Val images: {len(val_imgs)}")

    # quick label coverage check (sampled)
    def label_path_for(img: Path) -> Path:
        return (root / "labels" / img.parent.name / (img.stem + ".txt"))

    sample_size = min(50, len(train_imgs))
    missing = 0
    for i in range(sample_size):
        if not label_path_for(train_imgs[i]).exists():
            missing += 1
    if missing > 0:
        print(f"[WARN] {missing}/{sample_size} sampled train images had no matching label .txt")

    if len(train_imgs) == 0 or len(val_imgs) == 0:
        raise RuntimeError("No images found in one of the splits. Check dataset paths.")


def pick_device(device_flag: Optional[str]) -> str:
    """Decide which device string to pass to Ultralytics."""
    if device_flag is not None:
        return device_flag
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return "0"  # first GPU
    return "cpu"


def export_all(model_path: Path, half: bool = True, opset: int = 12):
    """Export best weights to ONNX and TensorRT (if env supports it)."""
    print(f"[EXPORT] Starting exports for: {model_path}")
    from ultralytics import YOLO
    model = YOLO(model_path.as_posix())

    # ONNX export (portable)
    onnx_out = model.export(format="onnx", opset=opset, half=half)
    print(f"[OK] ONNX export: {onnx_out}")

    # TensorRT export (requires TensorRT installed); failure is ok
    try:
        engine_out = model.export(format="engine", half=half)
        print(f"[OK] TensorRT export: {engine_out}")
    except Exception as e:
        print(f"[NOTE] TensorRT export skipped/failed: {e}")


# ---------------------- Main Train/Tune Pipeline -------------------------

def train_baseline(
    model_ckpt: str,
    data_yaml: Path,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    name: str,
    seed: int,
    optimizer: str,
    lr0: float,
    lrf: float,
    momentum: float,
    weight_decay: float,
    cos_lr,
):
    print(f"[TRAIN] Baseline training: {model_ckpt} → run name: {name}")
    model = YOLO(model_ckpt)
    results = model.train(
        data=data_yaml.as_posix(),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        name=name,
        seed=seed,
        verbose=True,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        momentum=momentum,
        weight_decay=weight_decay,
        cos_lr=cos_lr,
    )
    # best.pt path is stable under runs/detect/<name>/weights/best.pt
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    print(f"[DONE] Baseline best weights: {best}")
    return best, run_dir


def tune_hyperparams(
    base_model_ckpt: str,
    data_yaml: Path,
    imgsz: int,
    device: str,
    iterations: int,
    epochs: int,
    name: str,
):
    print(f"[TUNE] Starting hyperparameter search: {name}")
    model = YOLO(base_model_ckpt)
    tune_results = model.tune(
        data=data_yaml.as_posix(),
        imgsz=imgsz,
        device=device,
        iterations=iterations,
        epochs=epochs,
        optimizer="AdamW",
        plots=True,
        save=True,
        name=name,
    )
    # locate the tune directory and the best_hyperparameters.yaml if exists
    tune_dir = Path(tune_results.save_dir)
    hyp_path = tune_dir / "best_hyperparameters.yaml"
    if not hyp_path.exists():
        candidates = list(tune_dir.rglob("best_hyperparameters.yaml"))
        if candidates:
            hyp_path = candidates[0]
    if hyp_path.exists():
        print(f"[OK] Found tuned hyperparameters: {hyp_path}")
        return hyp_path, tune_dir
    else:
        raise FileNotFoundError("best_hyperparameters.yaml not found after tuning.")


def retrain_with_hyp(
    model_ckpt: str,
    data_yaml: Path,
    hyp_yaml: Path,
    imgsz: int,
    epochs: int,
    batch: int,
    device: str,
    workers: int,
    name: str,
    seed: int,
):
    print(f"[RE-TRAIN] Retraining with tuned hyperparameters: {hyp_yaml.name}")
    model = YOLO(model_ckpt)
    results = model.train(
        data=data_yaml.as_posix(),
        hyp=hyp_yaml.as_posix(),
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        workers=workers,
        name=name,
        seed=seed,
        verbose=True,
    )
    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    print(f"[DONE] Tuned best weights: {best}")
    return best, run_dir


def evaluate_model(model_path: Path, data_yaml: Path, device: str):
    print(f"[EVAL] Validating: {model_path.name}")
    model = YOLO(model_path.as_posix())
    results = model.val(data=data_yaml.as_posix(), device=device, verbose=True)
    # results dict is rich; we print key metrics briefly
    try:
        metrics = results.results_dict
        print("[METRICS]", json.dumps(metrics, indent=2))
    except Exception:
        print("[INFO] Validation finished; see runs for full metrics/plots.")


# ----------------------------- CLI --------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train, tune, and export YOLOv11 on BDD100K (YOLO format).")
    ap.add_argument("--root", type=str, default="data/bdd100k_yolo_converted", help="Dataset root folder")
    ap.add_argument("--model", type=str, default="yolo11s.pt", help="Base YOLOv11 checkpoint (e.g., yolo11n/s/m/l/x.pt)")
    ap.add_argument("--epochs", type=int, default=80, help="Baseline training epochs")
    ap.add_argument("--imgsz", type=int, default=1280, help="Training image size")
    ap.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    ap.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    ap.add_argument("--device", type=str, default=None, help="Device string (e.g., '0', 'cpu'); default: auto")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--no-tune", action="store_true", help="Skip hyperparameter tuning stage")

    # Tuning knobs
    ap.add_argument("--tune-iterations", type=int, default=120, help="Hyperparam search iterations")
    ap.add_argument("--tune-epochs", type=int, default=30, help="Epochs per tuning trial")

    ap.add_argument("--optimizer", type=str, default="SGD")
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.1)  # 0.01 -> 0.001
    ap.add_argument("--momentum", type=float, default=0.937)
    ap.add_argument("--weight-decay", type=float, default=0.0005, dest="weight_decay")
    ap.add_argument("--cos-lr", action="store_true", dest="cos_lr", default=True)

    # Names / run IDs
    ap.add_argument("--name-base", type=str, default="bdd11s_base", help="Run name for baseline training")
    ap.add_argument("--name-tune", type=str, default="bdd11s_tune", help="Run name for tuning stage")
    ap.add_argument("--name-final", type=str, default="bdd11s_tuned", help="Run name for final retrain with tuned hparams")

    # Class names (optional override)
    ap.add_argument("--classes", type=str, nargs="*", default=None, help="Override class names (space-separated)")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    device = pick_device(args.device)
    print(f"[DEVICE] Using: {device}")

    # 1) YAML + sanity check
    names = args.classes if args.classes else DEFAULT_CLASS_NAMES
    data_yaml = write_data_yaml(root, names)
    sanity_check_dataset(root)

    # 2) Baseline training
    best_base, base_run = train_baseline(
        model_ckpt=args.model,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        workers=args.workers,
        name=args.name_base,
        seed=args.seed,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        cos_lr=args.cos_lr,
    )

    # Baseline evaluation
    evaluate_model(best_base, data_yaml, device=device)

    # 3) Optional tuning
    if not args.no_tune:
        hyp_yaml, tune_dir = tune_hyperparams(
            base_model_ckpt=args.model,  # start tuner from the base checkpoint
            data_yaml=data_yaml,
            imgsz=args.imgsz,
            device=device,
            iterations=args.tune_iterations,
            epochs=args.tune_epochs,
            name=args.name_tune,
        )

        # 4) Retrain once with tuned hyperparams
        best_tuned, tuned_run = retrain_with_hyp(
            model_ckpt=args.model,
            data_yaml=data_yaml,
            hyp_yaml=hyp_yaml,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=device,
            workers=args.workers,
            name=args.name_final,
            seed=args.seed,
        )

        # Evaluate tuned model
        evaluate_model(best_tuned, data_yaml, device=device)

        # 5) Export tuned model to ONNX (+ TensorRT if available)
        export_all(best_tuned, half=True, opset=12)

        print("\n[SUMMARY]")
        print(f"Baseline run : {base_run}")
        print(f"Tune run     : {tune_dir}")
        print(f"Final run    : {tuned_run}")
        print(f"Best weights : {best_tuned}")
    else:
        # Export baseline if skipping tuning
        export_all(best_base, half=True, opset=12)
        print("\n[SUMMARY]")
        print(f"Baseline run : {base_run}")
        print(f"Best weights : {best_base}")


if __name__ == "__main__":
    main()
