from rich.console import Console
from rich.panel import Panel
from pathlib import Path
from rich import print
import typer
import time

from src.filter import filter, filter_check
from src.convert import convert, format_check, detect_format
from src.download import download, download_check


CLI_TEMP_WAIT_TIME = 1  # seconds
ALLPLANES_DATASET_URL = "https://allplanes.org/dataset"
app = typer.Typer(help="AllPlanes CLI: convert, download, or filter datasets", add_completion=False)


def convert_annotations(input_path: Path, output_path: Path, to_format: str, to_framework: str, copy_images: bool):
    """Convert annotation formats between diamond, hbb, and obb. Convert between frameworks like YOLO and COCO."""
    from_format, from_framework = detect_format(input_path)
    try:
        convert(from_format, to_format, from_framework, to_framework, input_path, output_path, copy_images)
        if not format_check(output_path, to_format):
            print(f"[bold red]❌ Conversion completed but format check failed at:[/bold red] {output_path}")
            return
        print(f"[bold green]✅ Converted {from_format} to {to_format} in {output_path}[/bold green]")
    except Exception as e:
        print(f"[bold red]❌ An error occurred during conversion:[/bold red] {e}")


def download_dataset(url: str, output_path: Path):
    """Download dataset from a given URL."""
    try:
        download(url, output_path)
        if not download_check(output_path):
            print(f"[bold red]❌ Download completed but dataset check failed at:[/bold red] {output_path}")
            return
        print(f"[bold magenta]✅ Downloaded dataset from {url} to {output_path}[/bold magenta]")
    except Exception as e:
        print(f"[bold red]❌ An error occurred during download:[/bold red] {e}")


def filter_dataset(input_path: Path, output_path: Path, filter_criteria: str):
    """Filter dataset based on user-specified criteria."""
    try:
        filter(input_path, output_path, filter_criteria)
        if not filter_check(output_path, filter_criteria):
            print(f"[bold red]❌ Filtering completed but check failed at:[/bold red] {output_path}")
            return
        print(f"[bold green]✅ Filtered dataset at {input_path} and saved to {output_path}[/bold green]")
    except Exception as e:
        print(f"[bold red]❌ An error occurred during filtering:[/bold red] {e}")


def visualize_dataset(input_path: Path):
    """Visualize dataset using FiftyOne."""
    import fiftyone as fo

    print(f"[blue]🔍 Visualizing[/blue] dataset at {input_path}")
    input_path = "data/example/train/annotations"
    dataset = fo.Dataset.from_dir(data_path=input_path, dataset_type=fo.types.FiftyOneImageDetectionDataset)
    session = fo.launch_app(dataset)
    session.wait()


def handle_convert(console: Console):
    console.print(
        Panel.fit(
            "[bold yellow]🔄 You selected: Convert[/bold yellow]\n"
            "[dim]Convert annotation formats between supported types (e.g., JSON, YOLO, COCO, HBB, OBB).[/dim]",
            border_style="bold",
        )
    )
    input_path = typer.prompt("Enter dataset path", default="data/AllPlanes")
    output_path = typer.prompt("Enter output directory", default="output/")
    to_framework = choice("Enter target framework [yolo, coco]", ["yolo", "coco"], default="yolo")
    to_format = choice("Enter target annotation format [hbb, obb, pose]", ["hbb", "obb", "pose"], default="hbb")
    copy_images = typer.confirm("Copy images to output directory?", default=True)
    convert_annotations(Path(input_path), Path(output_path), to_format, to_framework, copy_images)


def handle_filter(console: Console):
    console.print(
        Panel.fit(
            "[bold green]🧹 You selected: Filter[/bold green]\n"
            "[dim]Filter the dataset by criteria such as ICAO Name, Type Code, or other attributes.[/dim]",
            border_style="bold",
        )
    )
    input_path = typer.prompt("Enter dataset path", default="data/AllPlanes")
    output_path = typer.prompt("Enter output directory", default="output/")
    filter_criteria = choice("Enter filter criteria [size, class, category]", ["size", "class", "category"], default="size")

    console.print(
        Panel.fit(
            f"[bold yellow]🧹 Filtering dataset at {input_path} with criteria: {filter_criteria}[/bold yellow]\n",
            "[dim]This will filter the dataset based on the specified criteria and save the results to {output_path}.[/dim]",
            border_style="bold",
        )
    )


def handle_download(console: Console):
    console.print(
        Panel.fit(
            "[bold magenta]⬇️  You selected: Download[/bold magenta]\n" "[dim]Download the latest AllPlanes dataset from the official source.[/dim]",
            border_style="bold",
        )
    )
    output_path = typer.prompt("Enter output directory", default="data/")
    download_dataset(ALLPLANES_DATASET_URL, Path(output_path))


def handle_visualize(console: Console):
    console.print(
        Panel.fit(
            "[bold blue]🔍 You selected: Visualize[/bold blue]\n" "[dim]Visualize the dataset using FiftyOne for better insights.[/dim]",
            border_style="bold",
        )
    )
    input_path = typer.prompt("Enter dataset path", default="data/AllPlanes")
    visualize_dataset(Path(input_path))


def choice(prompt_text: str, choices: list, default: str = None) -> str:
    """
    Prompt the user for input and ensure the response is one of the allowed choices.
    """
    while True:
        choice = typer.prompt(prompt_text, default=default).strip().lower()
        if choice in choices:
            return choice
        print(f"[bold red]❌ Invalid choice: {choice}. Choose from {choices}.[/bold red]")


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
    console.print("3. [bold magenta]⬇️  download[/bold magenta]  - Download dataset")
    console.print("4. [bold blue]🔍 visualize[/bold blue]  - Visualize dataset\n")

    # select an action
    action = choice("What would you like to do?", ["convert", "filter", "download", "visualize", "1", "2", "3", "4"])
    action = {"1": "convert", "2": "filter", "3": "download", "4": "visualize"}.get(action, action)
    handlers = {"convert": handle_convert, "filter": handle_filter, "download": handle_download, "visualize": handle_visualize}.get(action)
    handlers(console)


if __name__ == "__main__":
    app()
