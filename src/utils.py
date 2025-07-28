"""
Utility functions for AllPlanes dataset

Provides path management, logging, safe I/O, and dataset analysis utilities.
"""

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import cv2
from datetime import datetime


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"allplanes_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) if verbose else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger("AllPlanes")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists and is writable"""
    
    output_path = Path(output_path)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        test_file = output_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
    except PermissionError:
        raise PermissionError(f"No write permission for output directory: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_path}: {e}")
    
    return output_path


def safe_copy_file(src: Path, dst: Path, overwrite: bool = False) -> bool:
    """Safely copy a file with error handling"""
    
    try:
        if dst.exists() and not overwrite:
            logger = logging.getLogger("AllPlanes")
            logger.warning(f"Destination file exists, skipping: {dst}")
            return False
        
        # Ensure destination directory exists
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dst)
        return True
        
    except Exception as e:
        logger = logging.getLogger("AllPlanes")
        logger.error(f"Failed to copy {src} to {dst}: {e}")
        return False


def safe_load_json(file_path: Path) -> Optional[Dict]:
    """Safely load JSON file with error handling"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger = logging.getLogger("AllPlanes")
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None
    except Exception as e:
        logger = logging.getLogger("AllPlanes")
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def safe_save_json(data: Dict, file_path: Path, indent: int = 2) -> bool:
    """Safely save JSON file with error handling"""
    
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        logger = logging.getLogger("AllPlanes")
        logger.error(f"Failed to save JSON to {file_path}: {e}")
        return False


def find_files_by_extension(directory: Path, extensions: List[str], 
                           recursive: bool = True) -> List[Path]:
    """Find files by extension(s) in directory"""
    
    files = []
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = '.' + ext
        
        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"
        
        files.extend(directory.glob(pattern))
        # Also search for uppercase extensions
        files.extend(directory.glob(pattern.replace(ext, ext.upper())))
    
    return sorted(list(set(files)))  # Remove duplicates and sort


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes"""
    
    try:
        size_bytes = file_path.stat().st_size
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def create_directory_structure(base_path: Path, structure: Dict[str, Any]) -> bool:
    """Create directory structure from nested dictionary"""
    
    try:
        for name, content in structure.items():
            current_path = base_path / name
            
            if isinstance(content, dict):
                # It's a directory
                current_path.mkdir(parents=True, exist_ok=True)
                create_directory_structure(current_path, content)
            else:
                # It's a file
                current_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, str):
                    current_path.write_text(content)
                elif content is None:
                    current_path.touch()
        
        return True
        
    except Exception as e:
        logger = logging.getLogger("AllPlanes")
        logger.error(f"Failed to create directory structure: {e}")
        return False


class DatasetAnalyzer:
    """Analyzes dataset properties and statistics"""
    
    def __init__(self):
        self.logger = logging.getLogger("AllPlanes.Analyzer")
    
    def analyze_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """Comprehensive dataset analysis"""
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
        
        analysis = {
            "path": str(dataset_path),
            "total_images": 0,
            "total_annotations": 0,
            "format": "Unknown",
            "framework": "Unknown",
            "resolutions": [],
            "labels": [],
            "file_sizes": {
                "images": [],
                "annotations": []
            },
            "directory_structure": {},
            "issues": []
        }
        
        # Analyze directory structure
        analysis["directory_structure"] = self._analyze_directory_structure(dataset_path)
        
        # Find image and annotation files
        image_files = find_files_by_extension(dataset_path, ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'])
        annotation_files = find_files_by_extension(dataset_path, ['.json', '.txt', '.xml'])
        
        analysis["total_images"] = len(image_files)
        
        # Analyze images
        if image_files:
            self._analyze_images(image_files, analysis)
        
        # Analyze annotations
        if annotation_files:
            self._analyze_annotations(annotation_files, analysis)
        
        # Detect format and framework
        analysis["format"] = self._detect_annotation_format(annotation_files)
        analysis["framework"] = self._detect_annotation_framework(annotation_files)
        
        # Check for common issues
        analysis["issues"] = self._check_dataset_issues(dataset_path, image_files, annotation_files)
        
        return analysis
    
    def _analyze_directory_structure(self, path: Path) -> Dict[str, Any]:
        """Analyze directory structure"""
        
        structure = {}
        
        try:
            for item in path.iterdir():
                if item.is_dir():
                    structure[item.name] = {
                        "type": "directory",
                        "file_count": len(list(item.rglob("*")))
                    }
                else:
                    structure[item.name] = {
                        "type": "file",
                        "size_mb": get_file_size_mb(item)
                    }
        except PermissionError:
            structure["_error"] = "Permission denied"
        
        return structure
    
    def _analyze_images(self, image_files: List[Path], analysis: Dict):
        """Analyze image files"""
        
        resolutions = set()
        file_sizes = []
        
        for img_file in image_files:
            # Get file size
            size_mb = get_file_size_mb(img_file)
            file_sizes.append(size_mb)
            
            # Get resolution
            try:
                img = cv2.imread(str(img_file))
                if img is not None:
                    height, width = img.shape[:2]
                    resolutions.add(f"{width}x{height}")
            except:
                # Try with PIL as fallback
                try:
                    from PIL import Image
                    with Image.open(img_file) as img:
                        width, height = img.size
                        resolutions.add(f"{width}x{height}")
                except:
                    pass
        
        analysis["resolutions"] = sorted(list(resolutions))
        analysis["file_sizes"]["images"] = file_sizes
    
    def _analyze_annotations(self, annotation_files: List[Path], analysis: Dict):
        """Analyze annotation files"""
        
        labels = set()
        file_sizes = []
        annotation_count = 0
        
        for ann_file in annotation_files:
            # Get file size
            size_mb = get_file_size_mb(ann_file)
            file_sizes.append(size_mb)
            
            # Analyze content based on file extension
            if ann_file.suffix.lower() == '.json':
                annotation_data = safe_load_json(ann_file)
                if annotation_data:
                    annotation_count += 1
                    labels.update(self._extract_labels_from_annotation(annotation_data))
            
            elif ann_file.suffix.lower() == '.txt':
                # Assume YOLO format
                try:
                    with open(ann_file, 'r') as f:
                        lines = f.readlines()
                    if lines:
                        annotation_count += 1
                        # YOLO files typically don't contain label names, just class IDs
                        labels.add("yolo_class")
                except:
                    pass
        
        analysis["total_annotations"] = annotation_count
        analysis["labels"] = sorted(list(labels))
        analysis["file_sizes"]["annotations"] = file_sizes
    
    def _extract_labels_from_annotation(self, annotation: Dict) -> Set[str]:
        """Extract labels from a single annotation"""
        
        labels = set()
        
        # Check different annotation formats
        if 'annotations' in annotation:
            for ann in annotation['annotations']:
                if 'category_name' in ann:
                    labels.add(ann['category_name'])
                elif 'label' in ann:
                    labels.add(ann['label'])
        
        if 'categories' in annotation:
            for cat in annotation['categories']:
                if 'name' in cat:
                    labels.add(cat['name'])
        
        if 'labels' in annotation:
            if isinstance(annotation['labels'], list):
                labels.update(annotation['labels'])
        
        return labels
    
    def _detect_annotation_format(self, annotation_files: List[Path]) -> str:
        """Detect annotation format (diamond, HBB, OBB)"""
        
        for ann_file in annotation_files:
            if ann_file.suffix.lower() == '.json':
                annotation_data = safe_load_json(ann_file)
                if annotation_data and 'annotations' in annotation_data:
                    for ann in annotation_data['annotations']:
                        if 'bbox' in ann:
                            bbox = ann['bbox']
                            if len(bbox) == 4:
                                return "HBB"  # Horizontal Bounding Box
                            elif len(bbox) == 5:
                                return "Diamond"  # Diamond format with angle
                            elif len(bbox) == 8:
                                return "OBB"  # Oriented Bounding Box
        
        return "Unknown"
    
    def _detect_annotation_framework(self, annotation_files: List[Path]) -> str:
        """Detect annotation framework (COCO, YOLO)"""
        
        has_json = any(f.suffix.lower() == '.json' for f in annotation_files)
        has_txt = any(f.suffix.lower() == '.txt' for f in annotation_files)
        
        if has_json:
            # Check for COCO format indicators
            for ann_file in annotation_files:
                if ann_file.suffix.lower() == '.json':
                    annotation_data = safe_load_json(ann_file)
                    if annotation_data:
                        # COCO format indicators
                        if 'images' in annotation_data and 'annotations' in annotation_data:
                            return "COCO"
                        # Single image COCO-style format
                        elif 'annotations' in annotation_data:
                            return "COCO-style"
        
        if has_txt:
            # Check for YOLO format
            for ann_file in annotation_files:
                if ann_file.suffix.lower() == '.txt':
                    try:
                        with open(ann_file, 'r') as f:
                            line = f.readline().strip()
                        
                        # YOLO format: class_id x_center y_center width height
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                # Check if all parts are numbers
                                try:
                                    [float(p) for p in parts[:5]]
                                    return "YOLO"
                                except ValueError:
                                    pass
                    except:
                        pass
        
        return "Unknown"
    
    def _check_dataset_issues(self, dataset_path: Path, image_files: List[Path], 
                             annotation_files: List[Path]) -> List[str]:
        """Check for common dataset issues"""
        
        issues = []
        
        # Check for missing images or annotations
        if not image_files:
            issues.append("No image files found")
        
        if not annotation_files:
            issues.append("No annotation files found")
        
        # Check for mismatched counts
        if image_files and annotation_files:
            img_basenames = {f.stem for f in image_files}
            ann_basenames = {f.stem for f in annotation_files}
            
            missing_annotations = img_basenames - ann_basenames
            missing_images = ann_basenames - img_basenames
            
            if missing_annotations:
                issues.append(f"Images without annotations: {len(missing_annotations)}")
            
            if missing_images:
                issues.append(f"Annotations without images: {len(missing_images)}")
        
        # Check for empty annotation files
        empty_annotations = 0
        for ann_file in annotation_files:
            if get_file_size_mb(ann_file) < 0.001:  # Less than 1KB
                empty_annotations += 1
        
        if empty_annotations > 0:
            issues.append(f"Empty annotation files: {empty_annotations}")
        
        # Check for very large files
        large_images = [f for f in image_files if get_file_size_mb(f) > 50]  # > 50MB
        if large_images:
            issues.append(f"Very large images (>50MB): {len(large_images)}")
        
        # Check for permission issues
        try:
            test_file = dataset_path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
        except:
            issues.append("No write permission in dataset directory")
        
        return issues
    
    def generate_report(self, analysis: Dict, output_path: Optional[str] = None) -> str:
        """Generate a detailed analysis report"""
        
        report_lines = []
        
        # Header
        report_lines.append("# AllPlanes Dataset Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append(f"Dataset Path: {analysis['path']}")
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"Total Images: {analysis['total_images']}")
        report_lines.append(f"Total Annotations: {analysis['total_annotations']}")
        report_lines.append(f"Annotation Format: {analysis['format']}")
        report_lines.append(f"Annotation Framework: {analysis['framework']}")
        report_lines.append("")
        
        # Resolutions
        if analysis['resolutions']:
            report_lines.append("## Image Resolutions")
            for resolution in analysis['resolutions']:
                report_lines.append(f"  • {resolution}")
            report_lines.append("")
        
        # Labels
        if analysis['labels']:
            report_lines.append("## Labels/Categories")
            for label in analysis['labels']:
                report_lines.append(f"  • {label}")
            report_lines.append("")
        
        # File sizes
        if analysis['file_sizes']['images']:
            img_sizes = analysis['file_sizes']['images']
            report_lines.append("## Image File Statistics")
            report_lines.append(f"  Average size: {sum(img_sizes)/len(img_sizes):.2f} MB")
            report_lines.append(f"  Largest image: {max(img_sizes):.2f} MB")
            report_lines.append(f"  Smallest image: {min(img_sizes):.2f} MB")
            report_lines.append("")
        
        # Issues
        if analysis['issues']:
            report_lines.append("## Issues Found")
            for issue in analysis['issues']:
                report_lines.append(f"  ⚠️  {issue}")
            report_lines.append("")
        else:
            report_lines.append("## Issues Found")
            report_lines.append("  ✅ No issues detected")
            report_lines.append("")
        
        # Directory structure
        report_lines.append("## Directory Structure")
        for name, info in analysis['directory_structure'].items():
            if info.get('type') == 'directory':
                report_lines.append(f"  📁 {name}/ ({info.get('file_count', 0)} files)")
            else:
                report_lines.append(f"  📄 {name} ({info.get('size_mb', 0):.2f} MB)")
        
        report_text = "\\n".join(report_lines)
        
        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Analysis report saved to: {output_path}")
        
        return report_text


class PathValidator:
    """Validates and normalizes file paths"""
    
    @staticmethod
    def validate_input_path(path: str) -> Path:
        """Validate input path exists and is readable"""
        
        path = Path(path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        
        if not path.is_dir():
            raise ValueError(f"Input path must be a directory: {path}")
        
        try:
            # Test read permissions
            list(path.iterdir())
        except PermissionError:
            raise PermissionError(f"No read permission for input path: {path}")
        
        return path
    
    @staticmethod
    def validate_output_path(path: str) -> Path:
        """Validate output path can be created and is writable"""
        
        path = Path(path).resolve()
        
        # Create parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test write permissions
        try:
            if path.exists():
                if not path.is_dir():
                    raise ValueError(f"Output path exists but is not a directory: {path}")
                # Test write permission
                test_file = path / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            else:
                # Create the directory
                path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            raise PermissionError(f"No write permission for output path: {path}")
        
        return path


def cleanup_temp_files(directory: Path, pattern: str = ".tmp*"):
    """Clean up temporary files in directory"""
    
    try:
        temp_files = directory.glob(pattern)
        count = 0
        
        for temp_file in temp_files:
            try:
                temp_file.unlink()
                count += 1
            except:
                pass
        
        logger = logging.getLogger("AllPlanes")
        logger.info(f"Cleaned up {count} temporary files")
        
    except Exception as e:
        logger = logging.getLogger("AllPlanes")
        logger.warning(f"Failed to cleanup temp files: {e}")


def get_available_space_gb(path: Path) -> float:
    """Get available disk space in GB"""
    
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024**3)  # Convert to GB
    except:
        return 0.0
