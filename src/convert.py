from anyio import Path


def detect_format(file_path: Path) -> str:
    """Detect the format of the dataset annotations."""
    pass

def convert(from_format: str, to_format: str, input_path: Path, output_path: Path, copy_images: bool):
    """Convert annotations from one format to another."""
    pass

def format_check(output_path: Path, to_format: str) -> bool:
    """Check if the output annotations are in the correct format."""
    pass