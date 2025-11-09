import os
import math
import time
from typing import Tuple, List, Dict

import cv2
import numpy as np


def compute_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
	"""Compute PSNR between two uint8 grayscale images."""
	assert original.shape == reconstructed.shape, "Shape mismatch for PSNR"
	orig = original.astype(np.float64)
	recon = reconstructed.astype(np.float64)
	mse = np.mean((orig - recon) ** 2)
	if mse == 0:
		return float('inf')
	max_i = 255.0
	return 10.0 * math.log10((max_i * max_i) / mse)


def full_search_sad(
	ref: np.ndarray,
	cur: np.ndarray,
	block_size: int,
	search_range: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Full search block matching using SAD.
	- ref: reference frame (previous), uint8, HxW
	- cur: current frame (to encode), uint8, HxW
	Returns:
	- recon: reconstructed frame from ref via MC
	- mv: motion vectors (dy, dx) per block as int16 array of shape (H/bs, W/bs, 2)
	"""
	H, W = cur.shape
	bs = block_size
	recon = np.zeros_like(cur)
	num_by = H // bs
	num_bx = W // bs
	mv = np.zeros((num_by, num_bx, 2), dtype=np.int16)

	for by in range(num_by):
		for bx in range(num_bx):
			y = by * bs
			x = bx * bs
			cur_blk = cur[y:y + bs, x:x + bs]

			# Valid dy/dx range considering image boundaries
			min_dy = max(-search_range, -y)
			max_dy = min(search_range, (H - bs) - y)
			min_dx = max(-search_range, -x)
			max_dx = min(search_range, (W - bs) - x)

			best_cost = float('inf')
			best_dy, best_dx = 0, 0
			for dy in range(min_dy, max_dy + 1):
				yy = y + dy
				for dx in range(min_dx, max_dx + 1):
					xx = x + dx
					ref_blk = ref[yy:yy + bs, xx:xx + bs]
					# SAD
					cost = np.abs(cur_blk.astype(np.int32) - ref_blk.astype(np.int32)).sum()
					if cost < best_cost:
						best_cost = cost
						best_dy, best_dx = dy, dx

			mv[by, bx, 0] = best_dy
			mv[by, bx, 1] = best_dx
			recon[y:y + bs, x:x + bs] = ref[y + best_dy:y + best_dy + bs, x + best_dx:x + best_dx + bs]

	return recon, mv


def three_step_search_sad(
	ref: np.ndarray,
	cur: np.ndarray,
	block_size: int,
	search_range: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Three-Step Search (TSS) using SAD.
	- Starts at (0,0), step = 2^floor(log2(R)), then step //= 2 until step == 0
	- Evaluates 9 points at each step (center + 8 neighbors), skipping out-of-bound candidates
	Returns recon and motion vectors per block (dy, dx)
	"""
	H, W = cur.shape
	bs = block_size
	recon = np.zeros_like(cur)
	num_by = H // bs
	num_bx = W // bs
	mv = np.zeros((num_by, num_bx, 2), dtype=np.int16)

	# Initial step size
	if search_range <= 0:
		step0 = 0
	else:
		step0 = 1 << int(math.floor(math.log2(search_range)))

	for by in range(num_by):
		for bx in range(num_bx):
			y = by * bs
			x = bx * bs
			cur_blk = cur[y:y + bs, x:x + bs]

			# Valid dy/dx considering boundaries
			min_dy = max(-search_range, -y)
			max_dy = min(search_range, (H - bs) - y)
			min_dx = max(-search_range, -x)
			max_dx = min(search_range, (W - bs) - x)

			best_dy, best_dx = 0, 0
			# Evaluate center first
			ref_blk_c = ref[y:y + bs, x:x + bs]
			best_cost = np.abs(cur_blk.astype(np.int32) - ref_blk_c.astype(np.int32)).sum()

			step = step0
			while step > 0:
				candidates = [
					(best_dy, best_dx),
					(best_dy - step, best_dx),
					(best_dy + step, best_dx),
					(best_dy, best_dx - step),
					(best_dy, best_dx + step),
					(best_dy - step, best_dx - step),
					(best_dy - step, best_dx + step),
					(best_dy + step, best_dx - step),
					(best_dy + step, best_dx + step),
				]
				for dy, dx in candidates:
					if dy < min_dy or dy > max_dy or dx < min_dx or dx > max_dx:
						continue
					yy = y + dy
					xx = x + dx
					ref_blk = ref[yy:yy + bs, xx:xx + bs]
					cost = np.abs(cur_blk.astype(np.int32) - ref_blk.astype(np.int32)).sum()
					if cost < best_cost:
						best_cost = cost
						best_dy, best_dx = dy, dx
				step //= 2

			mv[by, bx, 0] = best_dy
			mv[by, bx, 1] = best_dx
			recon[y:y + bs, x:x + bs] = ref[y + best_dy:y + best_dy + bs, x + best_dx:x + best_dx + bs]

	return recon, mv


def run_me_mc(
	ref_path: str,
	cur_path: str,
	outdir: str,
	block_size: int = 8,
	search_ranges: List[int] = (8, 16, 32),
) -> Dict[str, Dict[int, Dict[str, float]]]:
	"""
	Run ME/MC for Full Search and TSS over multiple search ranges.
	Saves reconstructed frames and residuals, and a metrics.txt summary.
	Returns nested metrics:
	  metrics[algo][R] = {'psnr': float, 'time_s': float}
	"""
	os.makedirs(outdir, exist_ok=True)

	ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
	cur = cv2.imread(cur_path, cv2.IMREAD_GRAYSCALE)
	assert ref is not None and cur is not None, "Failed to read images"
	assert ref.shape == cur.shape, "Input images must have same size"
	H, W = ref.shape

	metrics: Dict[str, Dict[int, Dict[str, float]]] = {'full': {}, 'tss': {}}

	for R in search_ranges:
		# Full search
		t0 = time.perf_counter()
		recon_full, _ = full_search_sad(ref, cur, block_size, R)
		time_full = time.perf_counter() - t0
		psnr_full = compute_psnr(cur, recon_full)
		metrics['full'][R] = {'psnr': float(psnr_full), 'time_s': float(time_full)}

		res_full = cv2.subtract(cur, recon_full)
		cv2.imwrite(os.path.join(outdir, f"recon_full_sr{R}.png"), recon_full)
		cv2.imwrite(os.path.join(outdir, f"residual_full_sr{R}.png"), res_full)

		# TSS
		t0 = time.perf_counter()
		recon_tss, _ = three_step_search_sad(ref, cur, block_size, R)
		time_tss = time.perf_counter() - t0
		psnr_tss = compute_psnr(cur, recon_tss)
		metrics['tss'][R] = {'psnr': float(psnr_tss), 'time_s': float(time_tss)}

		res_tss = cv2.subtract(cur, recon_tss)
		cv2.imwrite(os.path.join(outdir, f"recon_tss_sr{R}.png"), recon_tss)
		cv2.imwrite(os.path.join(outdir, f"residual_tss_sr{R}.png"), res_tss)

	# Save grayscale inputs for reference
	cv2.imwrite(os.path.join(outdir, "ref.png"), ref)
	cv2.imwrite(os.path.join(outdir, "cur.png"), cur)

	# Write metrics summary
	lines = []
	lines.append(f"Image size: {W}x{H}")
	lines.append(f"Block size: {block_size}x{block_size}")
	lines.append("")
	for R in search_ranges:
		fm = metrics['full'][R]
		tm = metrics['tss'][R]
		lines.append(f"Search range +- {R}:")
		lines.append(f"  Full  - PSNR: {fm['psnr']:.4f} dB, time: {fm['time_s']:.6f} s")
		lines.append(f"  TSS   - PSNR: {tm['psnr']:.4f} dB, time: {tm['time_s']:.6f} s")
		lines.append("")
	with open(os.path.join(outdir, "metrics.txt"), "w", encoding="utf-8") as f:
		f.write("\n".join(lines))

	return metrics


def main():
	# Defaults to Homework3 provided images and outputs inside the same directory
	this_dir = os.path.dirname(os.path.abspath(__file__))
	ref_path = os.path.join(this_dir, "one_gray.png")
	cur_path = os.path.join(this_dir, "two_gray.png")
	outdir = this_dir  # write outputs into Homework3 directory

	# Settings per homework
	block_size = 8
	search_ranges = [8, 16, 32]

	run_me_mc(ref_path, cur_path, outdir, block_size, search_ranges)


if __name__ == "__main__":
	main()
