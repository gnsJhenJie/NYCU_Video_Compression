import os
import argparse
from typing import Tuple

import numpy as np
import cv2


def load_image_rgb(image_path: str) -> np.ndarray:
	"""Load an image via OpenCV and return an RGB numpy array with shape (H, W, 3), dtype uint8.

	OpenCV loads images as BGR; we manually reorder channels to RGB without using cvtColor.
	"""
	bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
	if bgr is None:
		raise FileNotFoundError(f"Failed to read image with OpenCV: {os.path.abspath(image_path)}")
	# Convert BGR -> RGB by channel reordering (no color transform functions)
	rgb = bgr[..., ::-1].astype(np.uint8)
	return rgb


def save_grayscale(array_2d: np.ndarray, output_path: str) -> None:
	"""Save a 2D uint8 array as a grayscale image to output_path using OpenCV."""
	ok = cv2.imwrite(output_path, array_2d.astype(np.uint8))
	if not ok:
		raise RuntimeError(f"Failed to write image: {os.path.abspath(output_path)}")


def clip_round_u8(x: np.ndarray) -> np.ndarray:
	"""Round to nearest integer, clip to [0, 255], cast to uint8."""
	return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def compute_yuv_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute Y, U, V from R, G, B using provided formulas (BT.601-like, rounded coeffs)."""
	rf = r.astype(np.float64)
	gf = g.astype(np.float64)
	bf = b.astype(np.float64)

	y = 0.299 * rf + 0.587 * gf + 0.114 * bf
	u = -0.169 * rf - 0.331 * gf + 0.5 * bf + 128.0
	v = 0.5 * rf - 0.419 * gf - 0.081 * bf + 128.0

	return clip_round_u8(y), clip_round_u8(u), clip_round_u8(v)


def compute_ycbcr_from_rgb(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute Y, Cb, Cr from R, G, B using ITU-R BT.601 digital coefficients."""
	rf = r.astype(np.float64)
	gf = g.astype(np.float64)
	bf = b.astype(np.float64)

	y = 0.299 * rf + 0.587 * gf + 0.114 * bf
	cb = 128.0 -0.168736 * rf - 0.331264 * gf + 0.5 * bf
	cr = 128.0 + 0.5 * rf - 0.418688 * gf - 0.081312 * bf

	return clip_round_u8(y), clip_round_u8(cb), clip_round_u8(cr)



def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Compute and save RGB, YUV, and YCbCr channels as grayscale images.")
	parser.add_argument("input", nargs="?", default="lena.png", help="Path to input image (default: lena.png)")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = args.input
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Input image not found: {os.path.abspath(input_path)}")

	# Load RGB image
	rgb = load_image_rgb(input_path)
	r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

	# Save R, G, B as grayscale
	save_grayscale(r, "R.png")
	save_grayscale(g, "G.png")
	save_grayscale(b, "B.png")

	# Compute YUV (per user-provided formulas)
	y_yuv, u_yuv, v_yuv = compute_yuv_from_rgb(r, g, b)
	save_grayscale(y_yuv, "Y.png")
	save_grayscale(u_yuv, "U.png")
	save_grayscale(v_yuv, "V.png")

	# Compute YCbCr (BT.601 precise coefficients)
	y_ycbcr, cb, cr = compute_ycbcr_from_rgb(r, g, b)
	save_grayscale(cb, "Cb.png")
	save_grayscale(cr, "Cr.png")

	print("Saved grayscale channel images: R.png, G.png, B.png, Y.png, U.png, V.png, Cb.png, Cr.png")


if __name__ == "__main__":
	main()


