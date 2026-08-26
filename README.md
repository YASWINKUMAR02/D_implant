# AI Dental Implant Planning System

A Streamlit web application for 3D dental CBCT segmentation using the official **OralSeg** model.

## Architecture

- **Model**: OralSeg (SwinUNETR + MambaEncoder hybrid)
- **Checkpoint**: `models/model_workstation39.pt`
- **Output**: 35-class segmentation (teeth, jaw bones, mandibular canal)
- **Inference**: MONAI sliding-window (96³ ROI, 0.5 overlap)

## Label Map

| Label | Structure         |
|-------|-------------------|
| 1     | Maxilla           |
| 2     | Mandible          |
| 3–10  | Teeth 11–18       |
| 11–18 | Teeth 21–28       |
| 19–26 | Teeth 31–38       |
| 27–34 | Teeth 41–48       |
| 35    | Mandibular Canal  |

## Setup

### 1. Prerequisites

- Python 3.10
- CUDA Toolkit 11.8 or 12.x (for GPU inference)
- NVIDIA GPU (RTX 2050 with 4 GB VRAM minimum)

### 2. Install Dependencies

```bash
# Step 1: Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install mamba-ssm (requires nvcc compiler)
pip install causal-conv1d==1.4.0 mamba-ssm==2.2.0

# Step 3: Install remaining dependencies
pip install -r requirements.txt
```

> **Windows Note**: `mamba-ssm` requires CUDA Toolkit (with `nvcc`) to compile.
> If compilation fails, consider using WSL2 (Ubuntu) for full compatibility.

### 3. Verify CUDA

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 4. Run the App

```bash
streamlit run app.py
```

## Project Structure

```
DentalImplantAI/
├── app.py                          # Main Streamlit application
├── models/
│   └── model_workstation39.pt      # OralSeg pretrained checkpoint
├── OralSeg/                        # Official OralSeg source (from GitHub)
│   ├── modified/
│   │   ├── monai/networks/nets/OralSeg.py    # Model architecture
│   │   ├── model_segmamba/segmamba.py        # Mamba components
│   │   └── btcv/utils/data_utils.py          # Data utilities
│   ├── main_dataset.py
│   └── trainer_dataset.py
├── utils/
│   ├── dicom_utils.py              # ZIP extraction, DICOM validation
│   ├── nifti_utils.py              # DICOM → NIfTI conversion
│   ├── oralseg_inference.py        # Model loading + sliding-window inference
│   └── visualization.py            # 2D slice views + FDI tooth chart
├── uploads/                        # Temporary DICOM extraction
├── converted/                      # NIfTI CBCT volumes
├── outputs/                        # Segmentation results
└── requirements.txt
```

## Features

- **End-to-End Pipeline**: CBCT DICOM (.zip) → SimpleITK NIfTI → MONAI preprocessing → OralSeg inference → 35-class segmentation.
- **3D Interactive Model Viewer**: Real-time 3D orbit, zoom, and layer toggles (Maxilla, Mandible, Teeth, Nerve Canal) in browser via Plotly 3D.
- **3D Printable STL Export**: Direct export of binary STL surface meshes for CAD software and 3D printing.
- **2D Multi-Planar Slice Reconstruction**: Synchronized Axial, Coronal, and Sagittal slice navigation with color-coded anatomical overlays.
- **FDI Tooth Chart**: 32-tooth dental arch chart mapping detected vs. missing teeth.
- **Dual Inference Modes**:
  - **⚡ Fast Mode (`overlap=0.25`)**: ~1 to 2 minutes on laptop GPU.
  - **🎯 High Precision Mode (`overlap=0.50`)**: Full 8-way patch blending.
- **Zero PII**: Patient identifying information is filtered and never exposed.

## Privacy

- All processing is **local** — no data is sent to external services.
- Patient-identifying DICOM tags are never displayed in the UI.

## References

- OralSeg GitHub: https://github.com/OttoYouZhou/oralseg
- OralSeg HuggingFace: https://huggingface.co/aiadir/OralSeg
- MONAI: https://monai.io
