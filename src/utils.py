from pathlib import Path
from rich.console import Console
from scipy.spatial import distance
import numpy as np


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
