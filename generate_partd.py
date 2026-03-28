import cv2
import numpy as np
import sys
import os

def main(iuv_path, output_path):
    # Load DensePose IUV image
    iuv_img = cv2.imread(iuv_path)
    if iuv_img is None:
        print(f"Failed to load IUV image: {iuv_path}")
        sys.exit(1)

    # For demonstration, use the I channel as a fake mask (real PartD would use a model)
    partd_mask = iuv_img[:, :, 0]

    # Colorize the mask for visualization
    color_mask = cv2.applyColorMap(cv2.convertScaleAbs(partd_mask, alpha=10), cv2.COLORMAP_JET)
    cv2.imwrite(output_path, color_mask)
    print(f"PartD mask saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--iuv', required=True, help='Path to DensePose IUV image')
    parser.add_argument('--output', required=True, help='Path to save PartD mask')
    args = parser.parse_args()
    main(args.iuv, args.output)
