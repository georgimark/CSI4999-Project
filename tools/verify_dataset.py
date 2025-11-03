import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def verify_split(split_path_images: Path, split_path_labels: Path, split_name: str, dest_dir: Path) -> bool:
    images = list(split_path_images.glob("*.jpg"))
    labels = list(split_path_labels.glob("*.txt"))

    print(f"\nVerifying split: {split_name}")
    print(f"Images: {len(images)} | Labels: {len(labels)}")

    images_map = {img.stem: img for img in images}
    labels_set = {lbl.stem for lbl in labels}

    missing_labels_stems = images_map.keys() - labels_set
    missing_images_stems = labels_set - images_map.keys()

    all_ok = True

    if missing_labels_stems:
        all_ok = False
        print(f"{len(missing_labels_stems)} images missing corresponding labels. Moving them to {dest_dir}...")

        for stem in tqdm(sorted(missing_labels_stems), desc=f"Moving {split_name} unlabeled"):
            src_path = images_map[stem]
            dest_path = dest_dir / src_path.name

            try:
                shutil.move(src_path, dest_path)
            except Exception as e:
                print(f"    - FAILED to move {src_path.name}: {e}")
        print(f"Move complete for {split_name}.")

    if missing_images_stems:
        all_ok = False
        print(f"{len(missing_images_stems)} labels missing corresponding images (no file to move):")
        for name in sorted(missing_images_stems)[:10]:
            print(f"    - {name}.txt -> [X] no {name}.jpg")

    if all_ok:
        print(f"{split_name} is consistent: all image-label pairs present.")

    return all_ok

def main():
    parser = argparse.ArgumentParser(description="Verify YOLO image-label consistency and move unlabeled images.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to YOLO dataset root")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy folder structure (e.g., train/images, train/labels)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    splits = ["train", "val"]

    all_verified = True
    for split in splits:
        if args.legacy:
            img_dir = dataset_root / split / "images"
            lbl_dir = dataset_root / split / "labels"
        else:
            img_dir = dataset_root / "images" / split
            lbl_dir = dataset_root / "labels" / split

        dest_dir = dataset_root / f"{split}_unlabeled"
        dest_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"Missing directories for split: {split} (img_dir: {img_dir}, lbl_dir: {lbl_dir})")
            all_verified = False
            continue

        if not verify_split(img_dir, lbl_dir, split, dest_dir):
            all_verified = False

    if all_verified:
        print("\nAll splits passed verification successfully!")
    else:
        print("\nSome issues found. Unlabeled images have been moved.")

if __name__ == "__main__":
    main()