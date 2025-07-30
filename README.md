# AllPlanes Dataset Utility

🛩️ **Comprehensive toolkit for AllPlanes aircraft detection datasets**

Supports format conversion, filtering, and data management with ICAO aircraft classification system integration.

## 📋 Contents

- [Features](#-features)
- [Installation](#-installation) 
- [Quick Start](#-quick-start)
- [AllPlanes Format](#-allplanes-format)
- [CLI Usage](#-cli-usage)
- [Project Structure](#-project-structure)
- [Documentation](#-documentation)

## ✨ Features

- **Format Conversion:** AllPlanes ⇄ COCO/YOLO, Diamond ⇄ HBB ⇄ OBB
- **Advanced Filtering:** ICAO codes, aircraft hierarchy, resolution-based
- **Aircraft Database:** Real ICAO data with type classification and metadata
- **AllPlanes Support:** Custom filename parsing, keypoints, multi-split handling
- **Developer Tools:** Interactive CLI, logging, dataset analysis, comprehensive tests

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

## 🏃 Quick Start

```bash
# Generate sample data
python download.py --dest ./data --samples 10

# Analyze dataset
python main.py info ./data

# Convert formats
python main.py convert format ./data/annotations ./output/diamond_format \
  --from-format hbb --to-format diamond

# Filter by aircraft type
python main.py filter hierarchy ./data ./output/fighters_only \
  --aircraft-type Fighter
```

## 🛩️ AllPlanes Format

### Dataset Structure
```
AllPlanes/
├── AllPlanes.csv          # Aircraft metadata
├── classes.txt            # ICAO codes (77 types)
└── data/example/
    ├── train/val/test/
    │   ├── images/        # PNG files
    │   └── annotations/   # JSON files
```

### Annotation Format
```json
{
  "labels": ["K35R", "F16"],
  "bboxes": [[242, 104, 433, 287], [212, 351, 402, 534]],
  "keypoints": [[[...], [...]], [[...], [...]]]
}
```

### Filename Convention
```
location_source_date_z_tilex_tiley_tilesx_tilesy_cropx1_cropy1_cropx2_cropy2.png
```

### Working with AllPlanes
```bash
# Convert to COCO/YOLO
python main.py convert allplanes ./data/example ./output/coco \
  --csv-path AllPlanes.csv --to-format coco --split train

# Filter by ICAO codes
python main.py filter labels ./data/example ./output/fighters \
  --csv-path AllPlanes.csv --labels F16 F18 F22
```

## 💻 CLI Usage

### Format Conversion
```bash
# Annotation formats
python main.py convert format INPUT OUTPUT \
  --from-format {diamond,hbb,obb} --to-format {diamond,hbb,obb}

# Framework conversion
python main.py convert framework INPUT OUTPUT \
  --from-framework {coco,yolo} --to-framework {coco,yolo}
```

### Filtering
```bash
# By labels
python main.py filter labels INPUT OUTPUT --labels aircraft fighter

# By hierarchy
python main.py filter hierarchy INPUT OUTPUT \
  --aircraft-type Fighter --subtype Single-seat

# By resolution
python main.py filter resolution INPUT OUTPUT \
  --min-resolution 1280x720 --max-resolution 1920x1080
```

### Analysis
```bash
# Dataset info
python main.py info INPUT_PATH
```

## 📁 Project Structure

```
AllPlanes/
├── main.py                 # CLI entry point
├── download.py             # Dataset download
├── requirements.txt        # Dependencies
│
├── src/                   # Core modules
│   ├── conversion.py      # Format conversion
│   ├── filtering.py       # Dataset filtering
│   ├── hierarchy.py       # Aircraft hierarchy
│   ├── resolution.py      # Resolution analysis
│   ├── interactive.py     # CLI utilities
│   └── utils.py          # Common utilities
│
├── data/                  # Sample data
├── output/               # Operation outputs
├── tests/                # Test suite
├── docs/                 # Documentation
└── logs/                 # Auto-generated logs
```

## 📖 Documentation

### Annotation Formats
- **Diamond:** `[center_x, center_y, width, height, angle]` - Rotated boxes
- **HBB:** `[x, y, width, height]` - Horizontal boxes
- **OBB:** `[x1, y1, x2, y2, x3, y3, x4, y4]` - Oriented boxes

### Aircraft Hierarchy
```
ICAO Code → Aircraft Type → Aircraft Subtype

Examples:
F16 → Fighter → Single-seat
B737 → Commercial → Narrow-body
UH60 → Helicopter → Utility
```

### Framework Support
- **COCO:** Standard object detection format with categories and metadata
- **YOLO:** Normalized coordinates with class IDs

### Resolution Categories
- **Quality:** Low (<0.3MP), Medium (0.3-2MP), High (2-8MP), Ultra-High (>8MP)
- **Aspect Ratios:** 4:3, 16:9, 21:9, Square, Custom
- **Size:** Very Small to Very Large categories

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test categories
pytest tests/ -m unit
pytest tests/ -m integration

# Coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔧 Configuration

### Dataset Config (`dataset_config.json`)
```json
{
  "name": "My Aircraft Dataset",
  "version": "1.0.0",
  "format": "HBB",
  "framework": "COCO",
  "categories": [{"id": 1, "name": "aircraft", "supercategory": "vehicle"}],
  "hierarchy": {
    "icao_codes": ["F16", "B737", "A320"],
    "aircraft_types": ["Fighter", "Commercial"],
    "subtypes": ["Single-seat", "Narrow-body"]
  }
}
```

## 🛠️ Development

### Setup
```bash
pip install -r requirements.txt pytest pytest-cov black flake8 isort
```

### Code Quality
```bash
black src/ tests/ *.py    # Format
isort src/ tests/ *.py    # Sort imports
flake8 src/ tests/ *.py   # Lint
```

## 🔧 Troubleshooting

### Common Issues
- **Permissions:** `chmod -R 755 ./output`
- **Dependencies:** `pip install --force-reinstall -r requirements.txt`
- **Validation:** `python main.py info ./dataset`

### Performance Tips
- Use SSD storage for large datasets
- Filter before conversion
- Process in batches
- Use `--no-confirm` for automation

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Add tests and update docs
4. Format code: `black src/ tests/`
5. Run tests: `pytest tests/`
6. Submit pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

COCO Dataset, YOLO framework, OpenCV, PIL, Click, pytest

---

## Schema Reference

### ICAO Aircraft Types

| Type | Examples | Subtypes |
|------|----------|----------|
| Commercial | B737, A320, B777 | Narrow-body, Wide-body, Regional |
| Fighter | F16, F22, F18 | Single-seat, Twin-seat, Multi-role |
| Transport | C130, C17, C5 | Tactical, Strategic, Cargo |
| Helicopter | UH60, AH64, CH47 | Utility, Attack, Heavy-lift |
| Business | Citation, Gulfstream | Light, Medium, Large |

### File Naming
- **Images:** `{type}_{id}_{resolution}.{ext}`
- **Annotations:** `{image_basename}.json`
- **Hard Negatives:** Include `hardneg` identifier

**Happy aircraft detection! 🛩️✨**