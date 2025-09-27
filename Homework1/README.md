## Overview

This project reads `lena.png`, manually computes the RGB, YUV, and YCbCr channels (no ready-made color conversion APIs), and writes eight grayscale images: `R.png`, `G.png`, `B.png`, `Y.png`, `U.png`, `V.png`, `Cb.png`, `Cr.png`.

Implemented in Python using only image I/O (OpenCV `cv2`) and array math (NumPy). All color-space math is done explicitly in code.

## Requirements

- Python 3.8+
- Packages:
  - OpenCV (cv2)
  - NumPy

Install packages (one-time):

```bash
python3 -m pip install opencv-python numpy
```

## Files

- `convert_lena_channels.py`: script performing the conversions and saving outputs
- `lena.png`: input image (must be present in the same directory)

## How to run

Run from this directory (input path is an optional argument; default is `lena.png`):

```bash
# default input (lena.png in current directory)
python3 convert_lena_channels.py

# explicit input path
python3 convert_lena_channels.py /path/to/your/image.png
```

On success, the script prints a confirmation and writes:

- `R.png`, `G.png`, `B.png`
- `Y.png`, `U.png`, `V.png`
- `Cb.png`, `Cr.png`

Each output is a single-channel grayscale image visualizing that channel’s values in the 0–255 range.

## Formulas used

All transforms are computed manually per pixel with double precision, then rounded and clipped to [0, 255].

RGB → YUV (as provided):

\[
\begin{aligned}
Y &= 0.299 R + 0.587 G + 0.114 B \\
U &= -0.169 R - 0.331 G + 0.5 B + 128 \\
V &= 0.5 R - 0.419 G - 0.081 B + 128
\end{aligned}
\]

RGB → YCbCr (BT.601 digital coefficients):

\[
\begin{aligned}
Y &= 0.299 R + 0.587 G + 0.114 B \\
Cb &= -0.168736 R - 0.331264 G + 0.5 B + 128 \\
Cr &= 0.5 R - 0.418688 G - 0.081312 B + 128
\end{aligned}
\]
