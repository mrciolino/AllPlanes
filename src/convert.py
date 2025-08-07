from scipy.spatial import distance
from rich.console import Console
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import shutil
import os 

from src.utils import read_annotations, load_classes


def distance_line_point(line, point):
    line = np.array(line)
    point = np.array(point)[:2]
    dist = np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) / np.linalg.norm(line[1] - line[0])
    return dist


def get_distance_sum(polygon, corners):
    distance_sum = 0
    for i in range(4):
        line = [corners[i], corners[(i + 1) % 4]]
        distance_sum += min([distance_line_point(line, point) for point in polygon])
    return distance_sum


def Poly2OBB(polygon):
    polygon = np.array(polygon)[:, :2]
    dist = distance.cdist(polygon, polygon, "euclidean")
    major_axis = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    minor_axis = tuple([x for x in range(4) if x not in major_axis])
    minor_vec = polygon[minor_axis[1]] - polygon[minor_axis[0]]
    major_axis_endpoints = [polygon[major_axis[0]], polygon[major_axis[1]]]
    corners = [
        major_axis_endpoints[0] + minor_vec // 2,
        major_axis_endpoints[1] + minor_vec // 2,
        major_axis_endpoints[1] - minor_vec // 2,
        major_axis_endpoints[0] - minor_vec // 2,
    ]

    # if our OBB edges dont pass very near to the polygon corners (20 pixels away or less total)
    # we need to swap major and minor axes, this is a heuristic to ensure the OBB is a good fit
    distance_sum = get_distance_sum(polygon, corners)
    if distance_sum > 20:
        major_axis, minor_axis = minor_axis, major_axis
        minor_vec = polygon[minor_axis[1]] - polygon[minor_axis[0]]
        major_axis_endpoints = [polygon[major_axis[0]], polygon[major_axis[1]]]
        corners = [
            major_axis_endpoints[0] + minor_vec // 2,
            major_axis_endpoints[1] + minor_vec // 2,
            major_axis_endpoints[1] - minor_vec // 2,
            major_axis_endpoints[0] - minor_vec // 2,
        ]
    return np.array(corners).tolist()


class Converter:
    """Manage single annotation instance and YOLO conversion."""

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
        console: Console,
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
        self.classes_map = load_classes(console, path=dataset_dir / "classes.txt")
        self.annotations = read_annotations(console, dataset_dir)

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
                    inst = Converter(
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
        dataset_dir=Path("data/example"),
        output_dir=Path("output"),
        to_format="obb",  # or "hbb"/"pose"
        to_framework="yolo",
        copy_images=True,
    )
    converted_df = conv.convert()
    converted_df
