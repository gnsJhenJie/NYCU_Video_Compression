# VC HW2 – 2D-DCT

## Overview
This project implements image-wide 2D-DCT/IDCT without using library DCT/IDCT functions. It also implements a fast two-pass approach using two 1D-DCTs (rows then columns), visualizes DCT magnitude in the log domain, reconstructs via IDCT, and reports PSNR and runtime comparisons.

- 2D-DCT: matrix-based transform `Y = C_H @ X @ C_W.T`
- Two 1D-DCT: rows then columns `Y = C_H @ (X @ C_W.T)`
- 2D-IDCT and two 1D-IDCT provided
- Log-domain visualization of |DCT| with percentile clipping
- PSNR evaluation and runtime comparison

## Files
- `dct2d.py`: Main script with implementations and CLI
- `lena.png`: Input image (color); converted to grayscale internally
- Outputs (in `--outdir`):
  - `lena_gray.png`: Grayscale input
  - `dct_log.png`: Log-magnitude visualization of DCT coefficients
  - `recon_2d.png`: Reconstruction via 2D-IDCT
  - `recon_1d.png`: Reconstruction via two 1D-IDCT
  - `metrics.txt`: Sizes, runtimes, max coefficient diff, PSNR

## Environment
- Python 3.8+
- Packages: `numpy`, `opencv-python`

Install packages:

```bash
pip install numpy opencv-python
```

## Usage
Default run (uses `lena.png` in the working dir, writes outputs to current dir):

```bash
python dct2d.py
```

Specify input and output directory:

```bash
python dct2d.py /path/to/lena.png --outdir ./outputs
```

Optional validation against OpenCV (not used for results; for sanity check only):

```bash
python dct2d.py --validate
```

## Notes
- The script converts the input to grayscale before processing and saves it as `lena_gray.png`.
- DCT/IDCT are implemented from scratch using cosine transform matrices; no calls to library DCT/IDCT are used for the main results. The `--validate` flag compares against OpenCV's `dct` only for verification, using the correct separable two-pass mode (`cv2.DCT_ROWS`) to match our transform.
- `metrics.txt` includes timing for 2D-DCT vs two 1D-DCT and PSNR for both reconstructions.

## Example Output (from metrics.txt)
```
Image size: 512x512
2D-DCT time (s): 0.023201
Two 1D-DCT time (s): 0.007323
Max |DCT(2D) - DCT(1D)|: 1.7053e-11
IDCT(2D) time (s): 0.007168
IDCT(1D) time (s): 0.007411
PSNR (2D recon) dB: inf
PSNR (1D recon) dB: inf
```

