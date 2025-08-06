from src.utils import Poly2OBB
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import json
import os


class InstanceManager:
    """Manage single annotation instance and handle YOLO consversion."""

    def __init__(self, file_name: str, label: str, bbox: list, keypoints: list, classes_map: dict, img_w: int, img_h: int):
        self.file_name = file_name
        self.label = label
        self.bbox = bbox
        self.keypoints = keypoints
        self.classes_map = classes_map
        self.class_id = self.classes_map.get(self.label, -1)
        self.img_w = img_w
        self.img_h = img_h

    def yolo_hbb(self):
        if self.bbox is None or self.class_id == -1:
            return None
        x_min, y_min, x_max, y_max = self.bbox
        cx = ((x_min + x_max) / 2) / self.img_w
        cy = ((y_min + y_max) / 2) / self.img_h
        w = (x_max - x_min) / self.img_w
        h = (y_max - y_min) / self.img_h
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    def yolo_obb(self):
        if self.keypoints is None or self.class_id == -1 or len(self.keypoints) != 4:
            return None
        obb_pts = Poly2OBB(self.keypoints)
        obb_pts = np.array(obb_pts, dtype=float)
        obb_pts[:, 0] /= self.img_w
        obb_pts[:, 1] /= self.img_h
        flat = " ".join([f"{x:.6f} {y:.6f}" for (x, y) in obb_pts])
        return f"{self.class_id} {flat}"

    def yolo_pose(self):
        if self.keypoints is None or self.class_id == -1:
            return None
        x_min, y_min, x_max, y_max = self.bbox
        cx = ((x_min + x_max) / 2) / self.img_w
        cy = ((y_min + y_max) / 2) / self.img_h
        w = (x_max - x_min) / self.img_w
        h = (y_max - y_min) / self.img_h
        kp_str = " ".join([f"{kp[0]/self.img_w:.6f} {kp[1]/self.img_h:.6f} {kp[2]}" for kp in self.keypoints])
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}"


class ConversionManager:
    """Manage conversion of dataset annotations to YOLO format."""

    def __init__(
        self,
        dataset_dir: Path,
        output_dir: Path,
        to_format: str,
        to_framework: str,
        copy_images: bool = False,
        img_size=(512, 512),
    ):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.to_format = to_format.lower()
        self.to_framework = to_framework.lower()
        self.copy_images = copy_images
        self.img_w, self.img_h = img_size

        self.classes_map = self.load_classes(path=dataset_dir / "classes.txt")
        self.annotations = self.read_annotations(dataset_dir)

    def load_classes(self, path: Path) -> dict:
        with open(path, "r") as f:
            names = [line.strip() for line in f.readlines()]
        return {name: idx for idx, name in enumerate(names)}

    def read_annotations(self, root_dir: Path) -> pd.DataFrame:
        """Read all splits (train/val/test) and return a DataFrame."""
        records = []
        for split in ["train", "val", "test"]:
            ann_dir = root_dir / split / "annotations"
            if not ann_dir.exists():
                continue
            ann_files = [f for f in os.listdir(ann_dir) if f.endswith(".json")]
            for ann_file in ann_files:
                with open(ann_dir / ann_file, "r") as f:
                    data = json.load(f)
                labels = data.get("labels", [])
                bboxes = data.get("bboxes", [])
                keypoints = data.get("keypoints", [])
                for idx, label in enumerate(labels):
                    records.append(
                        {
                            "split": split,
                            "file": ann_file.replace(".json", ".png"),
                            "label": label,
                            "bbox": bboxes[idx] if idx < len(bboxes) else None,
                            "keypoints": keypoints[idx] if idx < len(keypoints) else None,
                        }
                    )
        return pd.DataFrame(records)

    def convert(self) -> pd.DataFrame:
        """Convert annotations to YOLO format, write txt, and return full DataFrame with yolo_line."""
        self.annotations[f"yolo_{self.to_format}"] = None  # add empty column first

        for split in ["train", "val", "test"]:
            split_df = self.annotations[self.annotations["split"] == split]
            if split_df.empty:
                continue

            label_dir = self.output_dir / split / "labels"
            img_dir = self.output_dir / split / "images"
            label_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

            for img_file in tqdm(split_df["file"].unique(), desc=f"Converting {split}"):
                inst_df = split_df[split_df["file"] == img_file]
                txt_path = label_dir / img_file.replace(".png", ".txt")
                lines = []

                for idx, row in inst_df.iterrows():
                    inst = InstanceManager(
                        file_name=row["file"],
                        label=row["label"],
                        bbox=row["bbox"],
                        keypoints=row["keypoints"],
                        classes_map=self.classes_map,
                        img_w=self.img_w,
                        img_h=self.img_h,
                    )

                    if self.to_format == "hbb":
                        line = inst.yolo_hbb()
                    elif self.to_format == "obb":
                        line = inst.yolo_obb()
                    elif self.to_format == "pose":
                        line = inst.yolo_pose()
                    else:
                        raise ValueError("Unsupported format")

                    if line:
                        lines.append(line)
                        self.annotations.at[idx, f"yolo_{self.to_format}"] = line  # store directly in DataFrame

                with open(txt_path, "w") as f:
                    f.write("\n".join(lines))

                if self.copy_images:
                    src_img = self.dataset_dir / split / "images" / img_file
                    dst_img = img_dir / img_file
                    if src_img.exists():
                        shutil.copy2(src_img, dst_img)

        return self.annotations

    def format_check(self, output_path: Path, to_format: str) -> bool:
        label_files = (
            list((output_path / "train" / "labels").glob("*.txt"))
            + list((output_path / "val" / "labels").glob("*.txt"))
            + list((output_path / "test" / "labels").glob("*.txt"))
        )
        return len(label_files) > 0 and all(os.path.getsize(f) > 0 for f in label_files)


if __name__ == "__main__":
    conv = ConversionManager(
        dataset_dir=Path("../data/example"),
        output_dir=Path("../output"),
        to_format="obb",  # or "hbb"/"pose"
        to_framework="yolo",
        copy_images=True,
    )
    converted_df = conv.convert()
    converted_df
