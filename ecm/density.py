"""T1: information density map.

iota(x) = -log2 p(pixel | its causal neighbours), estimated with a classic
lossless-image-coder model: the LOCO-I / JPEG-LS median edge detector (MED)
predicts each pixel from its left, upper and upper-left neighbours, and the
residual is coded with a discretised Laplacian whose scale adapts to the local
residual energy.

This gives real bits per pixel, not edge strength: a sharp but predictable
edge is cheap (the MED predictor follows edges), random texture is expensive.
A learned codec (e.g. CompressAI) can replace this later; the interface is the
same: frame in, bits-per-pixel map out.
"""
import numpy as np
import cv2


def med_residual(gray: np.ndarray) -> np.ndarray:
    """Residual of the MED predictor. gray: HxW float32 in [0, 255]."""
    x = gray.astype(np.float32)
    a = np.zeros_like(x); a[:, 1:] = x[:, :-1]          # left
    b = np.zeros_like(x); b[1:, :] = x[:-1, :]          # up
    c = np.zeros_like(x); c[1:, 1:] = x[:-1, :-1]       # up-left
    mx, mn = np.maximum(a, b), np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    r = x - pred
    # first row / column have no full context: code them against the left/up pixel
    r[0, 1:] = x[0, 1:] - x[0, :-1]
    r[1:, 0] = x[1:, 0] - x[:-1, 0]
    r[0, 0] = x[0, 0] - 128.0
    return r


def info_density(gray: np.ndarray, window: int = 7, min_scale: float = 0.3) -> np.ndarray:
    """Bits per pixel (>= 0) under an adaptive discretised Laplacian residual model."""
    r = np.rint(med_residual(gray))
    ar = np.abs(r)
    # local Laplacian scale = local mean |residual| (MLE of the Laplacian scale)
    scale = cv2.blur(ar, (window, window)) + min_scale
    # P(k) for integer residual k under Laplace(0, scale), discretised to unit bins
    p_zero = 1.0 - np.exp(-0.5 / scale)
    p_k = 0.5 * (np.exp(-(ar - 0.5) / scale) - np.exp(-(ar + 0.5) / scale))
    p = np.where(ar == 0, p_zero, p_k)
    return (-np.log2(np.clip(p, 1e-12, 1.0))).astype(np.float32)


def video_density(frames: np.ndarray, **kw) -> np.ndarray:
    """frames: T x H x W gray (0-255). Returns T x H x W bits per pixel."""
    return np.stack([info_density(f, **kw) for f in frames])
