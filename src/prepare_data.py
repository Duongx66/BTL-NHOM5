import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ZIP_PATH = r"D:\Dowload o D\archive.zip"


def find_dataset_root(raw_dir: Path) -> Path:
    candidates = []
    for path in raw_dir.rglob("*"):
        if not path.is_dir():
            continue
        class_dirs = [p for p in path.iterdir() if p.is_dir()]
        image_class_dirs = [
            p
            for p in class_dirs
            if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.iterdir() if f.is_file())
        ]
        if len(image_class_dirs) >= 2:
            candidates.append((len(image_class_dirs), path))
    if not candidates:
        raise FileNotFoundError(f"Could not find class folders under {raw_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def extract_zip(zip_path: Path, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if any(raw_dir.iterdir()):
        return
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(raw_dir)


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(image_path.resolve()), "label": label})
    if not rows:
        raise FileNotFoundError(f"No images found under {dataset_root}")
    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, seed: int, test_size: float, val_size: float):
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", default=DEFAULT_ZIP_PATH)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        fallback = Path(r"D:\Dowload ổ D\archive.zip")
        if fallback.exists():
            zip_path = fallback
        else:
            raise FileNotFoundError(f"Dataset zip not found: {args.zip_path}")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    extract_zip(zip_path, raw_dir)
    dataset_root = find_dataset_root(raw_dir)
    df = build_manifest(dataset_root)
    train, val, test = stratified_split(df, args.seed, args.test_size, args.val_size)

    train.to_csv(splits_dir / "train.csv", index=False)
    val.to_csv(splits_dir / "val.csv", index=False)
    test.to_csv(splits_dir / "test.csv", index=False)
    classes = sorted(df["label"].unique().tolist())
    (splits_dir / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")

    summary = df["label"].value_counts().sort_index().rename_axis("class").reset_index(name="count")
    summary.to_csv(splits_dir / "class_distribution.csv", index=False)
    shutil.copyfile(splits_dir / "class_distribution.csv", output_dir / "class_distribution.csv")
    print(f"Dataset root: {dataset_root}")
    print(f"Classes: {classes}")
    print(f"Train/val/test: {len(train)}/{len(val)}/{len(test)}")


if __name__ == "__main__":
    main()
