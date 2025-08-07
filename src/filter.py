from rich.console import Console
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import shutil
import json
import re

from src.utils import read_annotations
from rich.panel import Panel
from rich.table import Table

# Load aircraft information from CSV
aircraft_df = pd.read_csv("data/AllPlanes.csv")


class FilterManager:
    """Manage filtering of dataset based on user-defined criteria."""

    def __init__(self, console: Console, input_path: Path, output_path: Path, criteria: str, copy_images: bool = False):
        self.input_path = input_path
        self.output_path = output_path
        self.criteria = criteria
        self.console = console
        self.copy_images = copy_images

    def copy_split_images(self, split: str):
        """Copy images for the given split to the output directory using shutil2."""
        src_img_dir = self.input_path / split / "images"
        dst_img_dir = self.output_path / split / "images"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        for img_file in src_img_dir.glob("*.png"):
            shutil.copy2(img_file, dst_img_dir / img_file.name)

    def create_annotations_file(self, split: str, img_file: str, labels, bboxes, keypoints):
        """Create a per-image annotation file in output directory following required format."""
        ann = {"labels": labels, "bboxes": bboxes, "keypoints": keypoints}
        ann_dir = self.output_path / split / "annotations"
        ann_dir.mkdir(parents=True, exist_ok=True)
        json_filename = Path(img_file).with_suffix(".json").name
        ann_path = ann_dir / json_filename
        with open(ann_path, "w") as f:
            json.dump(ann, f)

    def apply_filters(self):
        """Apply user-defined filters to the dataset and process splits."""
        ann_df = read_annotations(self.console, self.input_path)
        filters = self._parse_criteria(self.criteria)
        filtered_ann = ann_df.copy()
        total_steps = sum([filters.get("size") is not None, filters.get("classes") is not None])
        with tqdm(total=total_steps, desc="Filtering", unit="step") as pbar:
            if filters.get("size") is not None:
                ws_min, ws_max, len_min, len_max = filters["size"]
                merged = filtered_ann.merge(aircraft_df, left_on="label", right_on="ICAO", how="left")
                merged = merged[(merged["wingspan"].astype(float) >= ws_min) & (merged["wingspan"].astype(float) <= ws_max)]
                merged = merged[(merged["length"].astype(float) >= len_min) & (merged["length"].astype(float) <= len_max)]
                filtered_ann = merged[ann_df.columns]
                pbar.update(1)
            if filters.get("classes") is not None:
                filtered_ann = filtered_ann[filtered_ann["label"].isin(filters["classes"])]
                pbar.update(1)

        # Process each split
        self.output_path.mkdir(parents=True, exist_ok=True)
        for split in ["train", "val", "test"]:
            split_df = filtered_ann[filtered_ann["split"] == split]
            if split_df.empty:
                continue
            # Copy images if requested
            if self.copy_images:
                self.copy_split_images(split)
            # Write per-image annotation files
            for img_file in split_df["file"].unique():
                img_annots = split_df[split_df["file"] == img_file]
                labels = img_annots["label"].tolist()
                bboxes = img_annots["bbox"].tolist() if "bbox" in img_annots.columns else []
                keypoints = img_annots["keypoints"].tolist() if "keypoints" in img_annots.columns else []
                self.create_annotations_file(split, img_file, labels, bboxes, keypoints)

        # Print comparison summary
        self.df = filtered_ann
        self.filtered_printout(filtered_ann, original_df=ann_df)
        filtered_ann.to_csv(self.output_path / "filtered_annotations.csv", index=False)
        return filtered_ann

    def check_output(self, df: pd.DataFrame) -> bool:
        """Check if the filtered dataset meets the criteria."""
        # empty check
        if df.empty:
            self.console.print(f"[bold red]❌ Dataframe is empty after filtering at {self.output_path}[/bold red]")
            return False

        # check for presence of all splits
        required_splits = ["train", "val", "test"]
        missing_splits = [split for split in required_splits if df[df["split"] == split]["file"].nunique() == 0]
        if missing_splits:
            self.console.print(f"[bold yellow]⚠️ Warning: No images found in splits: {', '.join(missing_splits)}[/bold yellow]")
            return False

        # final success message
        self.console.print(f"[bold green]✅ Filter check passed: {self.output_path} contains {len(df)} rows.[/bold green]")
        return True

    def _parse_criteria(self, criteria: str):
        filters = {}
        size_match = re.search(r"size:\s*wingspan=(\d+|0|inf)-(\d+|inf),\s*length=(\d+|0|inf)-(\d+|inf)", criteria)
        if size_match:
            ws_min = float(size_match.group(1)) if size_match.group(1) != "inf" else float("inf")
            ws_max = float(size_match.group(2)) if size_match.group(2) != "inf" else float("inf")
            len_min = float(size_match.group(3)) if size_match.group(3) != "inf" else float("inf")
            len_max = float(size_match.group(4)) if size_match.group(4) != "inf" else float("inf")
            filters["size"] = (ws_min, ws_max, len_min, len_max)
        class_match = re.search(r"classes:\s*([\w,\s]+)", criteria)
        if class_match:
            classes = [c.strip() for c in class_match.group(1).split(",") if c.strip()]
            filters["classes"] = classes
        return filters

    def filtered_printout(self, filtered_df: pd.DataFrame, original_df: pd.DataFrame = None):
        """
        Print a comparison table of metrics before and after filtering.
        If original_df is None, only filtered metrics are shown.
        """

        def get_metrics(df):
            num_images = df["file"].nunique() if "file" in df.columns else 0
            num_annotations = len(df)
            num_classes = df["label"].nunique() if "label" in df.columns else 0
            instance_density = num_annotations / num_images if num_images > 0 else 0
            class_density = sum(df.groupby("split")["label"].nunique()) / len(df["split"].unique()) if len(df["split"].unique()) > 0 else 0
            classes = df["label"].unique() if "label" in df.columns else []
            split_counts = df["split"].value_counts().to_dict()
            split_counts_images = {split: df[df["split"] == split]["file"].nunique() for split in df["split"].unique()}
            return {
                "Annotations": num_annotations,
                "Images": num_images,
                "Classes": num_classes,
                "Instance Density": f"{instance_density:.2f}",
                "Class Density": f"{class_density:.2f}",
                "Classes List": ", ".join(classes) if len(classes) > 0 else "None",
                "Instance Counts": ", ".join([f"{split}: {count}" for split, count in split_counts.items()]),
                "Image Counts": ", ".join([f"{split}: {count}" for split, count in split_counts_images.items()]),
            }

        metrics_before = get_metrics(original_df) if original_df is not None else None
        metrics_after = get_metrics(filtered_df)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric")
        if metrics_before:
            table.add_column("Before", justify="right")
            table.add_column("→", justify="center")
        table.add_column("After", justify="right")

        for key in ["Annotations", "Images", "Classes", "Instance Density", "Class Density", "Classes List", "Instance Counts", "Image Counts"]:
            before_val = metrics_before[key] if metrics_before else ""
            after_val = metrics_after[key]
            if metrics_before:
                table.add_row(key, str(before_val), "→", str(after_val))
            else:
                table.add_row(key, str(after_val))

        self.console.clear()
        panel = Panel.fit(table, title="Dataset Metrics Comparison", border_style="bold blue")
        self.console.print(panel)


if __name__ == "__main__":
    input_path = Path("data/example")
    output_path = Path("data/output/filtered")
    criteria = "size: wingspan=20-80, length=20-80 | classes: B737, MG21"
    fm = FilterManager(input_path, output_path, criteria)
    fm.apply_filters()
    fm.check_filters()
