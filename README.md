# SEM–EDX Co-location + Ratio Gating

Pipeline to detect phases from SEM–EDX by:
1) Logical co-location of elemental masks \(M_E\), and  
2) Pixel-wise ratio windows on co-located pixels,  
then overlay the detected pixels on the SEM image and report element-wise intensity sums and total ratios.

## Features
- Input: Bruker **.bcf** hypermaps (via HyperSpy) **or** pre-exported element maps (PNG/TIFF/NPY) + SEM image.
- Preprocess: percentile clipping normalization + optional denoise (bilateral or TV).
- Masks: threshold rules per element (percentile/Otsu/absolute).
- Co-location: AND over `include`, optional NOT via `exclude`.
- Ratios: pixel-wise \(A/B\) with inclusive windows.
- Outputs: overlay PNG, final mask (NPY), JSON summary, optional bar chart.

## Limitations
- Bruker **.spx** is a single spectrum (no pixel grid) and is not supported for mapping. Use `.bcf` or pre-exported maps.

## Install
```bash
pip install -r requirements.txt
