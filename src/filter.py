from rich.console import Console
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import re

from src.utils import read_annotations

# Load aircraft information from CSV
aircraft_df = pd.read_csv("data/AllPlanes.csv")


class FilterManager:
    """Manage filtering of dataset based on user-defined criteria."""

    def __init__(self, console: Console, input_path: Path, output_path: Path, criteria: str):
        self.input_path = input_path
        self.output_path = output_path
        self.criteria = criteria
        self.console = console

    def apply_filters(self):
        """Apply user-defined filters to the dataset."""
        # Load aircraft info and annotations
        ann_df = read_annotations(self.console, self.input_path)

        # Parse criteria string
        filters = self._parse_criteria(self.criteria)

        # Apply filters with tqdm progress bar
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

        # Save filtered annotations to output
        self.output_path.mkdir(parents=True, exist_ok=True)
        filtered_ann.to_csv(self.output_path / "filtered_annotations.csv", index=False)
        self.console.print(f"[bold green]✅ Filtered annotations saved to {self.output_path / 'filtered_annotations.csv'}[/bold green]")

    def check_filters(self) -> bool:
        """Check if the filtered dataset meets the criteria."""
        filtered_path = self.output_path / "filtered_annotations.csv"
        if not filtered_path.exists():
            self.console.print(f"[bold red]❌ Filtered file not found: {filtered_path}[/bold red]")
            return False
        df = pd.read_csv(filtered_path)
        if df.empty:
            self.console.print(f"[bold red]❌ Filtered file is empty: {filtered_path}[/bold red]")
            return False
        self.console.print(f"[bold green]✅ Filter check passed: {filtered_path} contains {len(df)} rows.[/bold green]")
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
        cat_match = re.search(r"categories:\s*([\w,\s]+)", criteria)
        return filters


if __name__ == "__main__":
    input_path = Path("data/example")
    output_path = Path("data/output/filtered")
    criteria = "size: wingspan=20-80, length=20-80 | classes: B737, MG21"
    fm = FilterManager(input_path, output_path, criteria)
    fm.apply_filters()
    fm.check_filters()
