# phynizz

Part C implementation (Virtual Try-On) is now available as a hackathon MVP.

## What is implemented

- Mesh-based garment draping using local triangle warping for better shoulder and torso curvature.
- HMR2.0 landmarks support (optional) with MediaPipe fallback.
- Garment foreground extraction (alpha channel or white-background removal).
- Alpha blending with fallback preview mode when pose confidence is low.
- FastAPI endpoint for integration and a CLI demo runner.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run API (Part C service)

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Try-on endpoint (multipart form):

```bash
curl -X POST http://localhost:8000/part-c/try-on \
	-F "user_image=@samples/user.jpg" \
	-F "garment_image=@samples/garment.png" \
	-F "category=top" \
	-F "scale_adjust=1.0" \
	-F "y_offset=0" \
	-F 'hmr_landmarks_json={"left_shoulder":[0.36,0.24],"right_shoulder":[0.61,0.24],"left_hip":[0.42,0.49],"right_hip":[0.56,0.49],"score":0.94}'
```

The output image is saved to `outputs/part_c_result.jpg`.

## Run local demo script

```bash
python scripts/run_part_c_demo.py \
	--user samples/user.jpg \
	--garment samples/garment.png \
	--out outputs/part_c_demo.jpg \
	--scale 1.0 \
	--y-offset 0 \
	--hmr-json samples/hmr_landmarks.json
```

## Integration contract

Input:
- user image
- garment image (prefer PNG with transparent background)
- optional controls: `scale_adjust`, `y_offset`

Output:
- `mode`: `mesh_warp`, `anchored_overlay` (fallback), or `fallback_preview`
- `confidence`: 0 to 1
- `warnings`: list of non-blocking issues
- `output_path`: generated image path