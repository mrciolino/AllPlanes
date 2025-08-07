from rich.console import Console
from pathlib import Path
import pandas as pd
import json


def validate_dataset_path(console: Console, path: Path) -> bool:
    """Validate dataset path structure and contents concisely."""
    if not path.exists() or not path.is_dir():
        console.print(f"❌ Invalid dataset path: {path}. Ensure it exists and is a directory.")
        return False
    if not any(path.iterdir()):
        console.print(f"❌ Dataset directory is empty: {path}. Check your download.")
        return False

    for split in ["train", "test", "val"]:
        split_dir = path / split
        ann_dir = split_dir / "annotations"
        img_dir = split_dir / "images"

        if not (split_dir.exists() and split_dir.is_dir() and ann_dir.exists() and ann_dir.is_dir() and img_dir.exists() and img_dir.is_dir()):
            console.print(f"❌ Missing or invalid directories in '{split}'. Check dataset structure.")
            return False

        ann_files = [f for f in ann_dir.iterdir() if f.is_file()]
        img_files = [f for f in img_dir.iterdir() if f.is_file()]
        if not ann_files or not img_files:
            console.print(f"❌ No annotation or image files in '{split}'. Check your download.")
            return False
        if len(ann_files) != len(img_files):
            console.print(f"❌ Mismatch: {len(ann_files)} annotations vs {len(img_files)} images in '{split}'.")
            return False

    return True


def load_classes(console: Console, path: Path) -> dict:
    """Load class names from a text file and return a mapping."""
    if not path.exists() or not path.is_file():
        console.print(f"❌ Class file not found: {path}")
        return {}
    try:
        with open(path, "r") as f:
            names = [line.strip() for line in f if line.strip()]
        return {name: idx for idx, name in enumerate(names)}
    except Exception as e:
        console.print(f"❌ Failed to load class names: {e}")
        return {}


def load_class_info(console: Console, path: Path) -> pd.DataFrame:
    """Load class information from a JSON file."""
    if not path.exists() or not path.is_file():
        console.print(f"❌ Class info file not found: {path}")
        return pd.DataFrame()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        console.print(f"❌ Failed to load class info: {e}")
        return pd.DataFrame()


def read_annotations(console: Console, root_dir: Path) -> pd.DataFrame:
    """Read all splits (train/val/test) and return a DataFrame."""
    records = []
    for split in ["train", "val", "test"]:
        ann_dir = root_dir / split / "annotations"
        if not ann_dir.exists() or not ann_dir.is_dir():
            console.print(f"⚠️ Skipping missing or invalid directory: {ann_dir}")
            continue
        ann_files = [f for f in ann_dir.glob("*.json")]
        if not ann_files:
            console.print(f"⚠️ No annotation files found in: {ann_dir}")
            continue
        for ann_file in ann_files:
            try:
                with open(ann_file, "r") as f:
                    data = json.load(f)
                labels = data.get("labels", [])
                bboxes = data.get("bboxes", [])
                keypoints = data.get("keypoints", [])
                for idx, label in enumerate(labels):
                    records.append(
                        {
                            "split": split,
                            "file": ann_file.stem + ".png",
                            "label": label,
                            "bbox": bboxes[idx] if idx < len(bboxes) else None,
                            "keypoints": keypoints[idx] if idx < len(keypoints) else None,
                        }
                    )
            except Exception as e:
                console.print(f"❌ Failed to read annotation {ann_file}: {e}")
    return pd.DataFrame(records)
