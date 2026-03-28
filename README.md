# Essential Pipeline - Body Analysis System

Complete standalone system for body measurements, 3D DensePose visualization, and 3D mesh generation.

## Folder Structure

```
essential_pipeline/
├── generate_densepose_iuv.py      # 3D DensePose segmentation (24-part heat-map)
├── generate_3d_mesh.py             # 3D mesh generation (Python + OBJ export)
├── smpl_process_image_lite.py      # 2D measurements extraction
├── models/                         # Pre-trained models
│   ├── multiHMR_672_L.pt          # HMR model (1.2GB)
│   └── smpl_models/
│       ├── basicmodel_m_lbs_10_207_0_v1.1.0.pkl
│       └── basicmodel_f_lbs_10_207_0_v1.1.0.pkl
├── modules/                        # Core system modules
│   ├── hmr_estimator.py           # HMR neural network inference
│   ├── smpl_utils.py              # SMPL body model management
│   └── direct_measurement.py      # Measurement extraction engine
├── test_images/                    # Sample test images
├── output/                         # Generated outputs
└── requirements_a.txt              # Python dependencies
```

## Quick Start

### Setup (One-time):
```bash
# Navigate to this folder
cd essential_pipeline

# Install dependencies
pip install -r requirements_a.txt
```

### Usage

#### 1. Get Body Measurements (2D Skeleton + Text)
```bash
python smpl_process_image_lite.py test_images/full_body.jpg
```
**Output:**
- `output/annotated_full_body.jpg` - Image with pose skeleton
- `output/measurements_full_body.txt` - Body measurements (CM + INCHES)

#### 2. Generate 3D DensePose (Heat-map Segmentation)
```bash
python generate_densepose_iuv.py test_images/full_body.jpg
```
**Output:**
- `output/densepose_iuv_full_body.png` - 24-part colored segmentation

#### 3. Generate 3D Mesh (Solid Model + OBJ)
```bash
python generate_3d_mesh.py test_images/full_body.jpg
```
**Output:**
- `output/mesh_3d_full_body.png` - Multi-view 3D visualization
- `output/mesh_3d_full_body.obj` - Wavefront OBJ (import into Blender/MeshLab)

#### 4. Run All Together
```bash
python smpl_process_image_lite.py test_images/full_body.jpg && \
python generate_densepose_iuv.py test_images/full_body.jpg && \
python generate_3d_mesh.py test_images/full_body.jpg
```

## Output Files

| File | Purpose |
|------|---------|
| `annotated_*.jpg` | 2D image with pose skeleton overlay |
| `measurements_*.txt` | Body measurements table (CM, INCHES) |
| `densepose_iuv_*.png` | 24-part colored body segmentation |
| `mesh_3d_*.png` | 3D solid mesh (3 views: Front/Side/3D) |
| `mesh_3d_*.obj` | 3D model for Blender/3D software |

## Measurements Extracted

- Height (cm)
- Shoulder width (cm)
- Chest circumference (cm)
- Waist circumference (cm)
- Hip circumference (cm)
- Arm length (cm)
- Leg length (cm)
- Inseam (cm)
- Torso length (cm)
- Neck circumference (cm)

All measurements provided in **both CM and INCHES**.

## Device Options

```bash
# GPU (CUDA)
python smpl_process_image_lite.py test_images/full_body.jpg --device cuda

# CPU (slower)
python smpl_process_image_lite.py test_images/full_body.jpg --device cpu
```

## Dependencies

See `requirements_a.txt` for full list. Key packages:
- `torch` (PyTorch 2.1+)
- `opencv-python` (image processing)
- `matplotlib` (visualization)
- `smplx` (SMPL body models)
- `numpy`, `scipy`

## Models Used

- **HMR (multiHMR_672_L.pt)**: ViT-L backbone for pose/shape estimation
- **SMPL v1.1.0**: Male/Female body templates with 6,890 vertices

## System Architecture

```
Image Input
    ↓
[HMR Inference] → Pose + Shape Parameters
    ↓
[SMPL Body Model] → 3D Mesh (6,890 vertices)
    ↓
┌───────────────────────────────────────┐
│  3 Parallel Output Streams:           │
├───────────────────────────────────────┤
│ 1. Measurements (text)                │
│ 2. DensePose (24-part segmentation)   │
│ 3. 3D Mesh (OBJ + visualization)      │
└───────────────────────────────────────┘
```

## Troubleshooting

### "Models not found"
Ensure `models/` folder contains:
- `multiHMR_672_L.pt` (1.2GB)
- `smpl_models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl`
- `smpl_models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl`

### GPU Out of Memory
Use `--device cpu` or reduce batch size

### Slow inference
First run loads models (slow). Subsequent runs are cached and faster.

## Example Commands

```bash
# Process a single image
python smpl_process_image_lite.py path/to/image.jpg

# Batch process with CPU
for img in test_images/*.jpg; do
  python smpl_process_image_lite.py "$img" --device cpu
done

# Generate all outputs for one image
ls test_images/full_body.jpg | while read img; do
  python smpl_process_image_lite.py "$img"
  python generate_densepose_iuv.py "$img"
  python generate_3d_mesh.py "$img"
done
```

## Output Location

All outputs saved to `output/` directory with naming:
- `annotated_<image_name>.*`
- `measurements_<image_name>.*`
- `densepose_iuv_<image_name>.*`
- `mesh_3d_<image_name>.*`
