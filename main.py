"""AllPlanes CLI: Convert, download, filter, and visualize aircraft datasets."""

import os
from rich import print
from rich.panel import Panel
from rich.console import Console

console = Console()
console.clear()
console.print(
    Panel.fit(
        "[bold blue]     ✈️  Welcome to AllPlanes Dataset Tool ✈️[/bold blue]\n" "[dim]This CLI tool helps manage the AllPlanes dataset.[/dim]",
        border_style="bold blue",
        padding=(1, 20),
    )
)

import time
import typer
import prompt_toolkit
from pathlib import Path
from typing import Tuple, List

from src.filter import FilterManager
from src.convert import ConversionManager
from src.utils import validate_dataset_path
from src.download import download, download_check
from src.utils import load_classes

# Constants
CLI_TEMP_WAIT_TIME = 1  # seconds
ALLPLANES_DATASET_URL = "https://allplanes.org/dataset"
DEFAULT_INPUT_PATH = "data/example"
DEFAULT_OUTPUT_PATH = "output/"

# Initialize Typer app and global console
app = typer.Typer(help="AllPlanes CLI: convert, download, or filter datasets", add_completion=False)


class FilterCriteria:
    """Handle filter criteria generation and validation."""

    COLOR_MAP = {"size": "yellow", "class": "blue"}

    @staticmethod
    def prompt_size_criteria() -> str:
        """Prompt for size-based filter criteria."""
        min_ws = typer.prompt("Min wingspan (meters) [type 'all' for no minimum]", default="20")
        max_ws = typer.prompt("Max wingspan (meters) [type 'all' for no maximum]", default="80")
        min_len = typer.prompt("Min length (meters) [type 'all' for no minimum]", default="20")
        max_len = typer.prompt("Max length (meters) [type 'all' for no maximum]", default="80")
        # Convert 'all' to appropriate values
        min_ws = "0" if min_ws.lower() == "all" else min_ws
        max_ws = "inf" if max_ws.lower() == "all" else max_ws
        min_len = "0" if min_len.lower() == "all" else min_len
        max_len = "inf" if max_len.lower() == "all" else max_len
        return f"size: wingspan={min_ws}-{max_ws}, length={min_len}-{max_len}"

    @staticmethod
    def prompt_class_criteria() -> str:
        """Prompt for class-based filter criteria."""
        available_classes = load_classes(os.path.join(DEFAULT_INPUT_PATH, "classes.txt"))  # change this to load from dataset directory
        console.print(f"Available classes: [bold]{', '.join(available_classes)}[/bold]")
        while True:
            aircraft_class = typer.prompt("Enter comma separated classes (e.g., B737, MG21)")
            selected = [cls.strip() for cls in aircraft_class.split(",") if cls.strip()]
            invalid = [cls for cls in selected if cls not in available_classes]
            if invalid:
                print(f"[bold red]❌ Invalid class(es): {', '.join(invalid)}. Choose only from available classes.[/bold red]")
            else:
                return f"classes: {', '.join(selected)}"

    @classmethod
    def get_criteria(cls, criterion: str) -> str:
        """Get filter criteria based on type."""
        criteria_map = {
            "size": cls.prompt_size_criteria,
            "class": cls.prompt_class_criteria,
        }

        if criterion not in criteria_map:
            console.print(f"[bold red]❌ Unknown filter type: {criterion}[/bold red]")
            time.sleep(CLI_TEMP_WAIT_TIME)
            return cls.get_criteria(criterion)

        return criteria_map[criterion]()


class UIHelper:
    """Helper class for user interface operations."""

    @staticmethod
    def get_directory_io(default_input: str = DEFAULT_INPUT_PATH, default_output: str = DEFAULT_OUTPUT_PATH) -> Tuple[Path, Path]:
        console.print(
            Panel.fit(
                "[bold]📂 Directory Input/Output 📂[/bold]\n" "[dim]Use {tab} to autocomplete input and output directories.[/dim]",
                border_style="bold blue",
            )
        )
        input_path = prompt_toolkit.prompt(
            f"Enter dataset path [{default_input}]: ",
            default=default_input,
            completer=prompt_toolkit.completion.PathCompleter(only_directories=True, expanduser=True),
        )
        output_path = prompt_toolkit.prompt(
            f"Enter output directory [{default_output}]: ",
            default=default_output,
            completer=prompt_toolkit.completion.PathCompleter(only_directories=True, expanduser=True),
        )

        if not validate_dataset_path(console, Path(input_path)):
            raise typer.Exit()
        return Path(input_path), Path(output_path)

    @staticmethod
    def choice(prompt_text: str, choices: List[str], default: str = None) -> str:
        """Prompt user for input and ensure response is valid."""
        numerical_options = {str(i + 1): choice for i, choice in enumerate(choices)}

        while True:
            user_choice = typer.prompt(prompt_text, default=default).strip().lower()
            if user_choice in choices or user_choice in numerical_options:
                return numerical_options.get(user_choice, user_choice)
            print(f"[bold red]❌ Invalid choice: {user_choice}. Choose from {choices}.[/bold red]")

    @staticmethod
    def print_filter_menu() -> None:
        """Display filter menu options."""
        console.print("[bold]Choose a filter type:[/bold]")
        console.print("1. [bold yellow]size[/bold yellow]     - Filter Wingspan and Length in meters")
        console.print("2. [bold blue]class[/bold blue]    - Selected Aircraft Classes to include (e.g., [cyan]B737, A321[/cyan])")


class DatasetOperations:
    """Core dataset operations."""

    @staticmethod
    def convert_annotations(input_path: Path, output_path: Path, to_format: str, to_framework: str, copy_images: bool) -> None:
        """Convert annotation formats between diamond, hbb, and obb."""
        try:
            converter = ConversionManager(
                console=console,
                dataset_dir=input_path,
                output_dir=output_path,
                to_format=to_format,
                to_framework=to_framework,
                copy_images=copy_images,
            )
            converter.convert()

            if not converter.format_check(output_path, to_format):
                console.print(f"[bold red]❌ Conversion completed but format check failed at:[/bold red] {output_path}")
                return

            console.print(f"[bold green]✅ Converted to {to_format} in {output_path}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]❌ An error occurred during conversion:[/bold red] {e}")

    @staticmethod
    def filter_dataset(input_path: Path, output_path: Path, filter_criteria: str) -> None:
        """Filter dataset based on criteria."""
        try:
            filter = FilterManager(
                console=console,
                input_path=input_path,
                output_path=output_path,
                criteria=filter_criteria,
            )
            filter.apply_filters()

            if not filter.check_filters(output_path):
                console.print(f"[bold red]❌ Filtering completed but check failed at:[/bold red] {output_path}")
                return

            console.print(f"[bold green]✅ Filtered dataset at {input_path} and saved to {output_path}[/bold green]")

        except Exception as e:
            console.print(f"[bold red]❌ An error occurred during filtering:[/bold red] {e}")

    @staticmethod
    def download_dataset(url: str, output_path: Path) -> None:
        """Download dataset from URL."""
        try:
            download(console, url, output_path)

            if not download_check(output_path):
                console.print(f"[bold red]❌ Download completed but dataset check failed at:[/bold red] {output_path}")
                return

            console.print(f"[bold magenta]✅ Downloaded dataset from {url} to {output_path}[/bold magenta]")

        except Exception as e:
            console.print(f"[bold red]❌ An error occurred during download:[/bold red] {e}")

    @staticmethod
    def visualize_dataset(input_path: Path) -> None:
        """Visualize dataset using FiftyOne."""
        try:
            import fiftyone as fo

            print(f"[blue]🔍 Visualizing[/blue] dataset at {input_path}")
            # fiftyone expects a specific dataset type so we need to convert from here
            dataset = fo.Dataset.from_dir(data_path=input_path, dataset_type=fo.types.FiftyOneImageDetectionDataset)
            session = fo.launch_app(dataset)
            session.wait()

        except ImportError:
            print("❌ FiftyOne not installed. Please install with: [bold red]pip install fiftyone[/bold red]")
        except Exception as e:
            print(f"[bold red]❌ An error occurred during visualization:[/bold red] {e}")


class CommandHandlers:
    """Command handler methods for different operations."""

    def __init__(self):
        self.ui = UIHelper()
        self.ops = DatasetOperations()
        self.filter_criteria = FilterCriteria()

    def handle_convert(self) -> None:
        """Handle conversion workflow."""
        input_path, output_path = self.ui.get_directory_io()
        console.print(
            Panel.fit(
                "[bold yellow]🔄 You selected: Convert[/bold yellow]\n"
                "[dim]Convert annotation formats between supported types (e.g., JSON, YOLO, COCO, HBB, OBB).[/dim]",
                border_style="bold",
            )
        )
        to_framework = self.ui.choice("Enter target framework [(1) yolo, (2) coco]", ["yolo", "coco"], default="yolo")
        to_format = self.ui.choice("Enter target annotation format [(1) hbb, (2) obb, (3) pose]", ["hbb", "obb", "pose"], default="hbb")
        copy_images = typer.confirm("Copy images to output directory?", default=True)
        console.print(f"[bold green]🔄 Converting annotations:[/bold green] [bold blue]{to_format} to {to_framework} in {output_path}[/bold blue]")
        self.ops.convert_annotations(input_path, output_path, to_format, to_framework, copy_images)

    def handle_filter(self) -> None:
        """Handle filtering workflow."""
        input_path, output_path = self.ui.get_directory_io()

        console.print(
            Panel.fit(
                "[bold green]🧹 You selected: Filter[/bold green]\n" "[dim]Filter the dataset by criteria such as size or class.[/dim]",
                border_style="bold",
            )
        )

        # Collect filter criteria
        criteria_list = []
        filter_sequence = []
        while True:
            self.ui.print_filter_menu()
            filter_type = self.ui.choice("\nSelect filter type", list(self.filter_criteria.COLOR_MAP.keys()))
            color = self.filter_criteria.COLOR_MAP[filter_type]
            filter_sequence.append(f"[bold {color}]{filter_type}[/bold {color}]")
            criteria_list.append(self.filter_criteria.get_criteria(filter_type))
            console.print(Panel.fit(f"[bold green]🧹 Applying filters:[/bold green] " + ", ".join(filter_sequence), border_style="bold"))
            if not typer.confirm("Add another filter?", default=False):
                break
        full_criteria = " | ".join(criteria_list)

        # Confirm filtering
        if not typer.confirm(f"Confirm filtering with criteria: {full_criteria}?", default=True):
            console.print("[bold red]❌ Filtering cancelled by user.[/bold red]")
            return
        console.print(f"[bold green]🧹 Filtering with criteria:[/bold green] [bold blue]{full_criteria}[/bold blue]")
        self.ops.filter_dataset(input_path, output_path, full_criteria)

    def handle_download(self) -> None:
        """Handle download workflow."""
        console.print(
            Panel.fit(
                "[bold magenta]⬇️  You selected: Download[/bold magenta]\n"
                "[dim]Download the latest AllPlanes dataset from the official source.[/dim]",
                border_style="bold",
            )
        )

        output_path = typer.prompt("Enter output directory", default="data/")
        self.ops.download_dataset(ALLPLANES_DATASET_URL, Path(output_path))

    def handle_visualize(self) -> None:
        """Handle visualization workflow."""
        console.print(
            Panel.fit(
                "[bold blue]🔍 You selected: Visualize[/bold blue]\n" "[dim]Visualize the dataset using FiftyOne for better insights.[/dim]",
                border_style="bold",
            )
        )

        input_path = typer.prompt("Enter dataset path", default=DEFAULT_INPUT_PATH)
        self.ops.visualize_dataset(Path(input_path))


@app.command()
def main():
    """Interactive menu for dataset operations: convert, download, or filter."""
    ui = UIHelper()
    handlers = CommandHandlers()

    # Display menu options
    console.print("[bold]Choose an action:[/bold]")
    console.print("1. [bold yellow]♻️  convert[/bold yellow]   - Convert annotation formats (e.g., [cyan]json/text hbb/obb yolo/coco[/cyan])")
    console.print("2. [bold green]🧹 filter[/bold green]    - Filter datasets (e.g., [cyan]by ICAO Name, Type Code, etc...[/cyan])")
    console.print("3. [bold magenta]📦 download[/bold magenta]  - Download dataset")
    console.print("4. [bold blue]🔍 visualize[/bold blue] - Visualize dataset\n")

    # Get user choice and execute handler
    action = ui.choice("What would you like to do?", ["convert", "filter", "download", "visualize"])
    handler_map = {
        "convert": handlers.handle_convert,
        "filter": handlers.handle_filter,
        "download": handlers.handle_download,
        "visualize": handlers.handle_visualize,
    }
    handler_map.get(action)()


if __name__ == "__main__":
    app()
