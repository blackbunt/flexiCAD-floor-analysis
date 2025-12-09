# Floor Analysis - Leveling Plan Generator

A Python tool for analyzing 3D floor measurement data and generating leveling plans for floating floor installations. This tool helps identify areas that need to be ground down or filled with leveling compound to achieve a flat surface.

<img src="ausgleichsplan_2.0mm.png" width="50%" alt="Example Leveling Plan">

## Overview

This application analyzes 3D measurement points from a DXF file and generates comprehensive leveling plans showing which areas need adjustment and by how much. It's particularly useful for floor installers working with floating floors that require a flat substrate.

The tool is designed to work with 3D measurement data from [FlexiJet 3D measurement systems](https://www.flexijet.info/) exported via FlexiCAD software.

## Features

- Imports 3D measurement data from DXF files
- Calculates local deviations using neighboring points
- Generates color-coded heatmaps showing problem areas
- Exports results in multiple formats (PNG, PDF, DXF, CSV)
- Configurable thresholds and search radius
- Automatic grid overlay (10cm and 50cm)
- Clear marking of areas requiring grinding or leveling compound

## Workflow

### Step 1: Data Preparation in FlexiCAD

The first step is to export your floor plan and all measured height points from FlexiCAD as a 3D DXF file. FlexiCAD is the software companion for [FlexiJet 3D measurement systems](https://www.flexijet.info/).

1. Open your project in FlexiCAD
2. Ensure the floor plan contour is closed (continuous line)
3. Include all measured 3D height points
4. Export as 3D DXF format
5. Save the file (e.g., `Boden.dxf`)

**Important:** The floor plan contour must be a closed polyline or connected lines for the boundary detection to work properly.

### Step 2: Python Analysis

With the exported DXF file, you can now analyze the floor data using Python:

1. Place the DXF file in the project directory
2. Configure the analysis parameters in `bodenanalyse.py`
3. Run the analysis script
4. Review the generated output files

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)

### Setup

1. Clone or download this repository
2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - Windows PowerShell: `.venv\Scripts\Activate.ps1`
   - Windows CMD: `.venv\Scripts\activate.bat`
   - Linux/Mac: `source .venv/bin/activate`

4. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install ezdxf matplotlib numpy scipy reportlab
```

## Quick Start

1. Place your DXF file in the same directory as `bodenanalyse.py`
2. Edit the filename in `bodenanalyse.py` (line ~29):
   ```python
   INPUT_DXF = "your_file.dxf"
   ```
3. Run the script:
   ```bash
   python bodenanalyse.py
   ```
4. Find results in the `ausgabe/` directory

## Configuration

Edit the configuration section at the top of `bodenanalyse.py`:

```python
# Input DXF file path
INPUT_DXF = "Schopfheimer_Str_ohne_Beschriftung.dxf"

# Output directory
OUTPUT_DIR = "ausgabe"

# Threshold in mm (deviations above this value will be marked)
SCHWELLE = 2.0  # 1.0 = very precise, 2.0 = standard, 3.0 = tolerant

# Search radius for local reference in mm
RADIUS = 400

# Enable/disable output formats
EXPORT_PNG = True
EXPORT_PDF = True
EXPORT_DXF = True
```

### Configuration Parameters

- **INPUT_DXF**: Path to your 3D DXF file
- **OUTPUT_DIR**: Directory for output files (created automatically)
- **SCHWELLE** (Threshold): Minimum deviation to flag (1.0 = strict, 2.0 = standard, 3.0 = tolerant)
- **RADIUS**: Search radius for neighboring points (400mm recommended for 10cm grid)
- **EXPORT_PNG/PDF/DXF**: Enable or disable specific output formats

## Output Files

| File | Description |
|------|-------------|
| `ausgleichsplan_Xmm.png` | High-resolution image (300 DPI) with color-coded heatmap |
| `ausgleichsplan_Xmm.pdf` | PDF format for printing |
| `ausgleichsplan_Xmm.dxf` | AutoCAD file with organized layers |
| `alle_punkte_Xmm.csv` | CSV with all measurement points and deviations |

## DXF Requirements

The input DXF file must contain:
- **POINT entities** with X, Y, Z coordinates (height measurements)
- **LINE entities** for floor plan boundary (optional but recommended)

## Coordinate System

- **Origin (0,0):** Northeast corner (NORDWAND reference)
- **X-axis:** Points west (negative from origin)
- **Y-axis:** Points south (negative from origin)

This system allows easy on-site measurements using a chalk line from the north wall.

## Algorithm

1. For each measurement point: Find neighbors within 400mm radius
2. Median of neighbor heights = local reference
3. Deviation = Measured height − Reference

**Positive deviation** → Bump → Grind down  
**Negative deviation** → Depression → Fill with leveling compound

This approach uses local references rather than global average, which accounts for intentional slopes and focuses on local flatness.

## DXF Layers

| Layer | Color | Content |
|-------|-------|---------|
| GRUNDRISS | White | Floor plan boundary |
| NORDWAND | Red | Reference edge + origin point |
| RASTER_500 | Gray | 50cm main grid |
| RASTER_100 | Light Gray | 10cm fine grid |
| ABTRAGEN_MARKER | Red | Circles (bumps to grind) |
| ABTRAGEN_WERT | Red | Millimeter values |
| SPACHTELN_MARKER | Blue | Squares (depressions to fill) |
| SPACHTELN_WERT | Blue | Millimeter values |
| BESCHRIFTUNG | Gray | Dimension labels |
| LEGENDE | White | Legend and statistics |

## Example Output

```
============================================================
BODENANALYSE - Ausgleichsplan Generator
============================================================

[1/4] Loading DXF file...
File: Schopfheimer_Str_ohne_Beschriftung.dxf
  → 2151 measurement points found
  → 10 floor plan lines found

Room size: 4.88 x 4.67 m
Height range: -24.2 to -7.9 mm (Δ 16.3 mm)

[2/4] Calculating local deviations (Radius: 400mm)...

Results (Threshold ±2.0mm):
  Grind down:  13 locations (max. 8.5mm)
  Fill:        68 locations (max. 6.6mm)

[3/4] Creating output files...
  ✓ ausgabe/ausgleichsplan_2.0mm.png
  ✓ ausgabe/ausgleichsplan_2.0mm.pdf
  ✓ ausgabe/ausgleichsplan_2.0mm.dxf

[4/4] Exporting CSV...
  ✓ ausgabe/alle_punkte_2.0mm.csv

============================================================
COMPLETE!
============================================================
```

## Tips

- **Test multiple thresholds:** Run the script with different SCHWELLE values (1.0, 2.0, 3.0 mm) to compare results
- **CSV in Excel:** Use semicolon (;) as delimiter when importing
- **DXF in AutoCAD:** Toggle individual layers on/off for better overview
- **On-site reference:** Print the PDF plan and use the 10cm grid to locate problem areas
- **Material planning:** Use CSV data to calculate total volume of leveling compound needed

## Troubleshooting

**"No POINT entities found"**  
→ DXF file contains no height points. Check FlexiCAD export settings.

**"Module not found"**  
→ Install dependencies: `pip install -r requirements.txt`

**Empty or strange graphics**  
→ Check coordinates. X/Y axes might be swapped or units incorrect.

**Too many problem areas**  
→ Increase SCHWELLE threshold or adjust RADIUS parameter.

## How It Works - Detailed

### 1. Data Loading
- Reads DXF file using ezdxf library
- Extracts all 3D POINT entities (measurement heights)
- Extracts LINE entities (floor plan boundary)
- Validates data completeness

### 2. Deviation Calculation
- Builds KD-tree for efficient spatial queries
- For each point, finds all neighbors within specified radius
- Calculates median height of neighbors as local reference
- Computes deviation from local reference
- Flags points exceeding threshold

### 3. Visualization
- Creates interpolated heatmap using scipy griddata
- Overlays floor plan boundary
- Adds 10cm and 50cm reference grids
- Marks problem areas with circles (grind) and squares (fill)
- Adds dimension lines and statistics

### 4. Export
- PNG/PDF: High-resolution images for printing and on-site use
- DXF: CAD file with organized layers for integration
- CSV: Structured data for analysis and documentation

## Best Practices

1. **Measurement Grid**: Use consistent spacing (10cm recommended)
2. **Measurement Density**: More points = better accuracy
3. **Closed Boundary**: Ensure floor plan outline is continuous
4. **Multiple Runs**: Test different thresholds to find optimal settings
5. **Documentation**: Save output files with project name and date
6. **Verification**: Cross-check critical areas with manual measurements

## Project Structure

```
bodenanalyse_lokal/
├── bodenanalyse.py              # Main analysis script
├── requirements.txt             # Python dependencies
├── README.md                    # Documentation
├── LICENSE                      # License information
├── .gitignore                   # Git ignore rules
├── .venv/                       # Virtual environment (created)
├── Boden.dxf                    # Input DXF file (your file)
└── ausgabe/                     # Output directory (auto-created)
    ├── ausgleichsplan_2.0mm.png
    ├── ausgleichsplan_2.0mm.pdf
    ├── ausgleichsplan_2.0mm.dxf
    └── alle_punkte_2.0mm.csv
```

## Requirements

All packages listed in `requirements.txt`:
- `ezdxf` - DXF file reading and writing
- `matplotlib` - Visualization and plotting
- `numpy` - Numerical computations
- `scipy` - Interpolation and spatial algorithms
- `reportlab` - PDF generation

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Feel free to submit issues or pull requests.

## Support

If you have problems or questions, check:
1. This README for solutions
2. Configuration parameters
3. Input DXF file format
4. Console output for error messages

## Acknowledgments

This tool was developed to help floor installers work more efficiently by automating floor leveling analysis from 3D measurement data. Special thanks to the community for sharing knowledge and helping others with similar challenges.
