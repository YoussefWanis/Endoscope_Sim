# C3VD Intelligent Endoscopic Assistance System

Endoscopic simulator with real colonoscopy video dataset (C3VD) and AI-based polyp detection (U-Net trained on Kvasir-SEG).

## Overview

This project implements an interactive endoscope simulator for research and education. It displays colonoscopy video frames, applies real‑time image processing, extracts quantitative features, and overlays A.I. polyp segmentation masks. The system supports both real dataset playback (C3VD) and a synthetic generation mode when no real data is available.

## Features

- **Real dataset support** – load C3VD (Colonoscopy 3D Video Dataset) sequences with RGB frames, depth maps, and camera poses.
- **Synthetic demo mode** – procedural generation of colon‑like anatomy when C3VD is not installed.
- **Interactive scope controls**  
  - Tip deflection (U/D, R/L)  
  - Zoom, focus, intensity, colour temperature  
  - LED modes: white, NBI (narrow‑band imaging), WLI+ (enhanced white light)
- **Image processing pipeline** (toggle per module)  
  - Noise reduction (Gaussian blur)  
  - CLAHE (contrast enhancement)  
  - Sobel edge detection  
  - Depth false‑colour overlay  
  - Inversion  
  - A.I. polyp segmentation (U‑Net)
- **Live metrics** – brightness, contrast, edge density, depth (if available), colour/texture features (dominant hue, saturation, redness index, LBP uniformity, GLCM contrast).
- **Sequence navigation** – next/previous frame, jump via filmstrip, next/previous sequence.
- **Capture frames** with flash effect.
- **Auto‑advance** (playback) mode.
- **Single‑image prediction** tab – load an external image and run A.I. segmentation.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

If you want GPU acceleration for A.I., install PyTorch with CUDA separately:  
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

### Optional: C3VD Dataset

The simulator works in **demo mode** without external data. To use real colonoscopy videos:

#### Automatic installation (recommended)

```bash
python install_dataset.py --small
```

This downloads the two smallest C3VD sequences (~1.6 GB total) and unpacks them into `data/c3vd/`.

#### Manual installation

1. Download one or more ZIP files from [C3VD dataset page](https://durrlab.github.io/C3VD/).
2. Extract each ZIP – a folder like `trans_t1_a/` is created.
3. Move this folder into `data/c3vd/` (create the directory if it does not exist).
4. Final structure example:

```
c3vd_scope/
└── data/
    └── c3vd/
        └── trans_t1_a/
            ├── 0001.png
            ├── 0002.png
            ├── ...
            ├── depth/
            │   ├── 0001_depth.tiff
            │   └── ...
            └── pose.txt
```

Run `python install_dataset.py --check` to verify what is already installed.

### Optional: Train U‑Net for polyp segmentation

The application looks for a pre‑trained model at `unet_kvasir_best.pth`. If not found, the A.I. button has no effect. To train your own model:

1. Download the [Kvasir‑SEG dataset](https://datasets.simula.no/kvasir-seg/) and place it as:

```
../dataset/kvasir-seg/
    images/
    masks/
```

2. Run training:

```bash
python Model_train.py
```

The script will save `unet_kvasir_best.pth` and `unet_kvasir_final.pth`. Copy the best model to the same folder as `endoscope_app.py`.

## Usage

Run the simulator:

```bash
python endoscope_app.py
```

If no C3VD data is found, a demo mode with synthetic frames starts automatically. A dialog shows instructions for installing the real dataset.

### Keyboard controls

| Key           | Action                     |
|---------------|----------------------------|
| ↑ / ↓ / ← / → | Pan camera tip (deflection)|
| , (comma)     | Previous frame             |
| . (period)    | Next frame                 |
| A / D         | Previous / next sequence   |
| [ / ]         | Previous / next sequence   |
| Space         | Capture current frame      |
| 1 … 6         | Toggle processing modules  |
| + / -         | Zoom in / out              |
| Escape        | Quit                       |

### Processing modules toggle matrix

| Key | Module            |
|-----|-------------------|
| 1   | Noise reduction   |
| 2   | CLAHE             |
| 3   | Edge detection    |
| 4   | Depth overlay     |
| 5   | Invert            |
| 6   | A.I. segmentation |

## Project structure

```
c3vd_scope/
├── endoscope_app.py            # Main simulator application
├── install_dataset.py          # C3VD downloader / installer
├── Model_train.py              # U‑Net training script (Kvasir dataset)
├── requirements.txt            # Python dependencies
├── data/
│   └── c3vd/                   # Unpacked C3VD sequences (created by installer)
│       └── <sequence_name>/
│           ├── *.png
│           ├── depth/
│           └── pose.txt
├── outputs/                    # Captured frames (auto‑created)
└── unet_kvasir_best.pth        # Optional pre‑trained segmentation model
```

## License

- **C3VD dataset**: [CC BY‑NC‑SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Johns Hopkins University).
- **Kvasir‑SEG dataset**: [CC BY‑NC‑SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (Simula Research Laboratory).
- **Code**: provided for educational purposes under the same terms as the datasets (CC BY‑NC‑SA 4.0) unless otherwise stated.

## Acknowledgments

- C3VD dataset by Durr Lab, Johns Hopkins University.
- Kvasir‑SEG dataset by Simula Research Laboratory.
- U‑Net architecture by Ronneberger et al.