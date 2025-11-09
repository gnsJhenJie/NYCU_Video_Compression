# VC HW3 – Motion Estimation & Compensation

## Overview
This project implements block-based motion estimation and compensation on two grayscale frames:
- Block size: `8×8`, integer-precision motion vectors
- Cost: SAD
- Algorithms:
  - Full search (FS) within ranges `±8`, `±16`, `±32`
  - Three-Step Search (TSS) with initial step `2^floor(log2(R))`
- Motion compensation: reconstruct current frame from reference via estimated MVs and save residuals
- Metrics: PSNR and runtime for FS vs TSS at each range

## Files
- `me_mc.py`: Main script (ME/MC, FS & TSS, metrics, outputs)
- Inputs (provided):
  - `one_gray.png` (reference)
  - `two_gray.png` (current)
- Outputs (written to `Homework3/`):
  - `recon_full_sr{R}.png`, `residual_full_sr{R}.png`
  - `recon_tss_sr{R}.png`, `residual_tss_sr{R}.png`
  - `metrics.txt` (PSNR and runtime summary)
- Report:
  - `VC_HW3_report.md` (simple report; convert to PDF for submission)

## Environment
- Python 3.8+
- Packages: `numpy`, `opencv-python`

Install packages:

```bash
pip install numpy opencv-python
```

## Usage
Run with default inputs (uses `one_gray.png` and `two_gray.png` in this folder):

```bash
python me_mc.py
```

This will generate reconstructions and residuals for FS and TSS at ranges ±8, ±16, ±32, and write a `metrics.txt` summary.


