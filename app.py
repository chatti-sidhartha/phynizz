from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from part_c import TryOnConfig, virtual_try_on

app = FastAPI(title="Phynizz Part C Service", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "part-c-virtual-try-on"}


@app.post("/part-c/try-on")
async def try_on(
    user_image: UploadFile = File(...),
    garment_image: UploadFile = File(...),
    category: str = Form("top"),
    scale_adjust: float = Form(1.0),
    y_offset: int = Form(0),
    hmr_landmarks_json: str | None = Form(None),
) -> JSONResponse:
    user_bytes = await user_image.read()
    garment_bytes = await garment_image.read()

    user_np = np.frombuffer(user_bytes, dtype=np.uint8)
    garment_np = np.frombuffer(garment_bytes, dtype=np.uint8)

    user_bgr = cv2.imdecode(user_np, cv2.IMREAD_COLOR)
    garment_img = cv2.imdecode(garment_np, cv2.IMREAD_UNCHANGED)

    if user_bgr is None or garment_img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image input"})

    hmr_landmarks = None
    if hmr_landmarks_json:
        try:
            hmr_landmarks = json.loads(hmr_landmarks_json)
        except json.JSONDecodeError:
            return JSONResponse(status_code=400, content={"error": "Invalid hmr_landmarks_json payload"})

    result = virtual_try_on(
        user_bgr,
        garment_img,
        TryOnConfig(
            category=category,
            scale_adjust=scale_adjust,
            y_offset=y_offset,
            hmr_landmarks=hmr_landmarks,
        ),
    )

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "part_c_result.jpg"
    cv2.imwrite(str(out_file), result.image_bgr)

    return JSONResponse(
        {
            "mode": result.mode,
            "confidence": result.confidence,
            "warnings": result.warnings,
            "output_path": str(out_file),
        }
    )
