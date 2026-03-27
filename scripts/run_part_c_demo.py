from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from part_c import TryOnConfig, virtual_try_on


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local Part C virtual try-on demo")
    parser.add_argument("--user", required=True, help="Path to user image")
    parser.add_argument("--garment", required=True, help="Path to garment image")
    parser.add_argument("--out", default="outputs/part_c_demo.jpg", help="Output path")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale adjustment")
    parser.add_argument("--y-offset", type=int, default=0, help="Vertical offset in pixels")
    parser.add_argument("--hmr-json", default=None, help="Path to HMR landmarks JSON")
    args = parser.parse_args()

    user = cv2.imread(args.user, cv2.IMREAD_COLOR)
    garment = cv2.imread(args.garment, cv2.IMREAD_UNCHANGED)

    if user is None:
        raise ValueError(f"Could not read user image: {args.user}")
    if garment is None:
        raise ValueError(f"Could not read garment image: {args.garment}")

    hmr_landmarks = None
    if args.hmr_json:
        with open(args.hmr_json, "r", encoding="utf-8") as f:
            hmr_landmarks = json.load(f)

    result = virtual_try_on(
        user,
        garment,
        TryOnConfig(scale_adjust=args.scale, y_offset=args.y_offset, hmr_landmarks=hmr_landmarks),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), result.image_bgr)

    print({
        "output_path": str(out_path),
        "mode": result.mode,
        "confidence": round(result.confidence, 4),
        "warnings": result.warnings,
    })


if __name__ == "__main__":
    main()
