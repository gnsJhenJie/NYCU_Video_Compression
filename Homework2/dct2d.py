import os
import argparse
import time

import numpy as np
import cv2


def load_image_gray(image_path: str) -> np.ndarray:
	"""Load an image as grayscale (H, W), dtype uint8."""
	gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	if gray is None:
		raise FileNotFoundError(f"Failed to read image: {os.path.abspath(image_path)}")
	return gray.astype(np.uint8)


def save_grayscale(array_2d: np.ndarray, output_path: str) -> None:
	"""Save a 2D uint8 array as a grayscale image."""
	ok = cv2.imwrite(output_path, array_2d.astype(np.uint8))
	if not ok:
		raise RuntimeError(f"Failed to write image: {os.path.abspath(output_path)}")


def clip_round_u8(x: np.ndarray) -> np.ndarray:
	"""Round to nearest integer, clip to [0, 255], cast to uint8."""
	return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def dct_matrix(n: int) -> np.ndarray:
	"""Build an orthonormal DCT-II transform matrix of size (n, n).

	C[u, x] = alpha(u) * cos(pi * (2x + 1) * u / (2n))
	alpha(0) = sqrt(1/n), alpha(u>0) = sqrt(2/n)
	"""
	if n <= 0:
		raise ValueError("n must be positive")
	x = np.arange(n, dtype=np.float64)
	u = x.reshape(-1, 1)  # column vector for broadcasting
	alpha = np.sqrt(2.0 / n) * np.ones((n, 1), dtype=np.float64)
	alpha[0, 0] = np.sqrt(1.0 / n)
	C = alpha * np.cos((np.pi * (2.0 * x + 1.0) * u) / (2.0 * n))
	return C


def dct2d_matrix(x: np.ndarray) -> np.ndarray:
	"""2D-DCT using transform matrices: Y = C_H @ X @ C_W.T."""
	H, W = x.shape
	C_H = dct_matrix(H)
	C_W = dct_matrix(W)
	# Two matmuls, but conceptually a 2D transform via matrices
	return C_H @ x @ C_W.T


def idct2d_matrix(y: np.ndarray) -> np.ndarray:
	"""2D-IDCT using transform matrices: X = C_H.T @ Y @ C_W."""
	H, W = y.shape
	C_H = dct_matrix(H)
	C_W = dct_matrix(W)
	return C_H.T @ y @ C_W


def dct1d_rows_then_cols(x: np.ndarray) -> np.ndarray:
    """Two-pass 1D-DCT: rows then columns.

    Rows: Y1 = X @ C_W.T
    Cols: Y  = C_H @ Y1
    """
    H, W = x.shape
    C_H = dct_matrix(H)
    C_W = dct_matrix(W)
    Y_rows = x @ C_W.T
    Y = C_H @ Y_rows
    return Y


def idct1d_cols_then_rows(y: np.ndarray) -> np.ndarray:
	"""Two-pass 1D-IDCT: columns then rows.

	Cols: X1 = C_H.T @ Y
	Rows: X  = X1 @ C_W
	"""
	H, W = y.shape
	C_H = dct_matrix(H)
	C_W = dct_matrix(W)
	X_cols = C_H.T @ y
	X = X_cols @ C_W
	return X


def log_visualize_coeffs(dct: np.ndarray, clip_percentile: float = 99.0) -> np.ndarray:
	"""Create an 8-bit image visualizing |DCT| in log domain with percentile clipping."""
	mag = np.log1p(np.abs(dct).astype(np.float64))
	# Percentile-based scaling to avoid domination by a few large coefficients
	p_low = np.min(mag)
	p_high = np.percentile(mag, clip_percentile)
	if p_high <= p_low:
		p_high = np.max(mag)
	# Normalize to [0, 255]
	norm = (mag - p_low) / (p_high - p_low + 1e-12)
	img = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
	return img


def psnr(original_u8: np.ndarray, reconstructed_u8: np.ndarray) -> float:
	"""Compute PSNR in dB between two uint8 images."""
	orig = original_u8.astype(np.float64)
	rec = reconstructed_u8.astype(np.float64)
	mse = np.mean((orig - rec) ** 2)
	if mse == 0:
		return float('inf')
	max_i = 255.0
	return 10.0 * np.log10((max_i * max_i) / mse)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="2D DCT/IDCT on grayscale image, visualize log DCT, compare runtimes.")
	parser.add_argument("input", nargs="?", default="lena.png", help="Path to input image (default: lena.png)")
	parser.add_argument("--outdir", default=".", help="Output directory (default: current directory)")
	parser.add_argument("--validate", action="store_true", help="Validate with OpenCV's dct/idct (not used for results)")
	return parser.parse_args()


def ensure_dir(path: str) -> None:
	if not os.path.isdir(path):
		os.makedirs(path, exist_ok=True)


def main() -> None:
	args = parse_args()
	input_path = args.input
	out_dir = args.outdir
	ensure_dir(out_dir)

	# 1) Load grayscale
	img_u8 = load_image_gray(input_path)
	H, W = img_u8.shape
	save_grayscale(img_u8, os.path.join(out_dir, "lena_gray.png"))

	# Convert to float for transform
	img_f = img_u8.astype(np.float64)

	# 2) 2D-DCT via transform matrices
	start = time.perf_counter()
	dct2 = dct2d_matrix(img_f)
	time_2d = time.perf_counter() - start

	# 3) Two 1D-DCT (rows then columns)
	start = time.perf_counter()
	dct1 = dct1d_rows_then_cols(img_f)
	time_1d2 = time.perf_counter() - start

	# Confirm closeness
	diff_coeff = float(np.max(np.abs(dct2 - dct1)))

	# 4) Visualize coefficients in log domain (use 2D result)
	log_img = log_visualize_coeffs(dct2, clip_percentile=99.0)
	save_grayscale(log_img, os.path.join(out_dir, "dct_log.png"))

	# 5) IDCT reconstructions
	start = time.perf_counter()
	rec2 = idct2d_matrix(dct2)
	t_rec2 = time.perf_counter() - start

	start = time.perf_counter()
	rec1 = idct1d_cols_then_rows(dct1)
	t_rec1 = time.perf_counter() - start

	# Round/clip to 8-bit for comparison and saving
	rec2_u8 = clip_round_u8(rec2)
	rec1_u8 = clip_round_u8(rec1)
	save_grayscale(rec2_u8, os.path.join(out_dir, "recon_2d.png"))
	save_grayscale(rec1_u8, os.path.join(out_dir, "recon_1d.png"))

	# 6) PSNR versus original
	psnr_2d = psnr(img_u8, rec2_u8)
	psnr_1d = psnr(img_u8, rec1_u8)

	# Optional validation against OpenCV's dct/idct
	val_note = ""
	if args.validate:
		img_f32 = img_f.astype(np.float32)
		# OpenCV validation (two-pass 1D using DCT_ROWS to match separable 2D DCT)
		cv_rows = cv2.dct(img_f32, flags=cv2.DCT_ROWS)
		cv_dct = cv2.dct(cv_rows.T, flags=cv2.DCT_ROWS).T
		cv_diff = float(np.max(np.abs(cv_dct.astype(np.float64) - dct2)))
		val_note = f"\nOpenCV validation max |cv_dct - ours|: {cv_diff:.6g}"

	# 7) Print summary and also write a small metrics file
	metrics = [
		f"Image size: {H}x{W}",
		f"2D-DCT time (s): {time_2d:.6f}",
		f"Two 1D-DCT time (s): {time_1d2:.6f}",
		f"Max |DCT(2D) - DCT(1D)|: {diff_coeff:.6g}",
		f"IDCT(2D) time (s): {t_rec2:.6f}",
		f"IDCT(1D) time (s): {t_rec1:.6f}",
		f"PSNR (2D recon) dB: {psnr_2d:.6f}",
		f"PSNR (1D recon) dB: {psnr_1d:.6f}",
	]
	print("\n".join(metrics) + val_note)
	with open(os.path.join(out_dir, "metrics.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(metrics) + val_note + "\n")


if __name__ == "__main__":
	main()


