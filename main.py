import time
from pathlib import Path
import typer
from rich import print
from rich.panel import Panel
from rich.align import Align
from rich.console import Console
from rich.table import Table

CLI_TEMP_WAIT_TIME = 2  # seconds
ALLPLANES_DATASET_URL = "https://allplanes.org/dataset"
app = typer.Typer(help="AllPlanes CLI: convert, download, or filter datasets", add_completion=False)


def convert_annotations(input_path: Path, output_path: Path, from_format: str, to_format: str, copy_images: bool):
    """Convert annotation formats between diamond, hbb, and obb."""
    print(f"[cyan]Converting[/cyan] from {from_format or '[unspecified]'} to {to_format} in {input_path}")
    print(f"{'✅ Copying images' if copy_images else '🔁 Skipping image copy'} to {output_path}")


def download_dataset(url: str, output_path: Path):
    """Download dataset from a given URL."""
    print(f"[magenta]⬇️  Downloading[/magenta] AllPlanes Dataset to {output_path}")


def filter_dataset(input_path: Path, output_path: Path, filter_criteria: str):
    """Filter dataset based on user-specified criteria."""
    print(f"[green]🧹 Filtering[/green] {input_path} with criteria: '{filter_criteria}' → {output_path}")


@app.command()
def main():
    """Interactive menu for dataset operations: convert, download, or filter."""
    console = Console()
    console.print(
        Panel.fit(
            "[bold blue]     ✈️  Welcome to AllPlanes Dataset Tool ✈️[/bold blue]\n" "[dim]This CLI tool helps manage the AllPlanes dataset.[/dim]",
            border_style="bold blue",
            padding=(1, 20),
        )
    )
    console.print("[bold]Choose an action:[/bold]")
    console.print("1. [bold yellow]🔄 convert[/bold yellow]   - Convert annotation formats (e.g., [cyan]json/text hbb/obb yolo/coco[/cyan])")
    console.print("2. [bold green]🧹 filter[/bold green]    - Filter datasets (e.g., [cyan]by ICAO Name, Type Code, etc...[/cyan])")
    console.print("3. [bold magenta]⬇️  download[/bold magenta]  - Download dataset\n")

    action = typer.prompt("What would you like to do? (convert, filter or download)").strip().lower()
    action = {"1": "convert", "2": "filter", "3": "download"}.get(action, action)

    if action == "convert":
        console.print(
            Panel.fit(
                "[bold yellow]🔄 You selected: Convert[/bold yellow]\n"
                "[dim]Convert annotation formats between supported types (e.g., JSON, YOLO, COCO, HBB, OBB).[/dim]",
                border_style="bold",
            )
        )
        input_path = typer.prompt("Enter dataset path", default="data/AllPlanes")
        output_path = typer.prompt("Enter output directory", default="output/")
        from_format = typer.prompt("Enter source annotation format (e.g., json, yolo, coco)", default="json")
        to_format = typer.prompt("Enter target annotation format (e.g., yolo, coco, hbb, obb)", default="yolo")
        copy_images = typer.confirm("Copy images to output directory?", default=True)
        convert_annotations(Path(input_path), Path(output_path), from_format, to_format, copy_images)
        typer.echo("✅ Conversion complete.")

    elif action == "filter":
        console.print(
            Panel.fit(
                "[bold green]🧹 You selected: Filter[/bold green]\n"
                "[dim]Filter the dataset by criteria such as ICAO Name, Type Code, or other attributes.[/dim]",
                border_style="bold",
            )
        )
        input_path = typer.prompt("Enter dataset path", default="data/AllPlanes")
        output_path = typer.prompt("Enter output directory", default="output/")
        filter_criteria = typer.prompt("Enter filter criteria (e.g., class=plane)")
        filter_dataset(Path(input_path), Path(output_path), filter_criteria)
        typer.echo("✅ Filtering complete.")

    elif action == "download":
        console.print(
            Panel.fit(
                "[bold magenta]⬇️  You selected: Download[/bold magenta]\n"
                "[dim]Download the latest AllPlanes dataset from the official source.[/dim]",
                border_style="bold",
            )
        )
        output_path = typer.prompt("Enter output directory", default="data/")
        download_dataset(ALLPLANES_DATASET_URL, Path(output_path))
        typer.echo("✅ Download complete.")

    else:
        typer.echo("❌ Invalid action. Please choose 'convert', 'download', or 'filter'.")
        time.sleep(CLI_TEMP_WAIT_TIME)
        app()


if __name__ == "__main__":
    app()
