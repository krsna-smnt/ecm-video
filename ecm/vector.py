"""Vector entropy: a per-pixel fingerprint of separate information channels.

Instead of one total (scalar density), each pixel carries a vector:
  channel 0      tone: MED-predictor information density (bits/px)
  channels 1-8   texture: local energy of oriented band-pass responses
                 (4 orientations x 2 scales), in bits via log2(1 + energy / noise)

A legal motion carries the whole fingerprint along the flow. A swap, vanish or melt
changes it even when the scalar total stays about the same. Because the channels are
local *statistics*, not raw pixels, they are less sensitive to lighting gain and
compression noise than the full-content (pixel surprise) source term.
"""
import numpy as np
import cv2
from .density import info_density

_BANK = None


def _bank():
    global _BANK
    if _BANK is None:
        _BANK = []
        for lam in (4.0, 8.0):
            for th in np.arange(4) * np.pi / 4:
                k = cv2.getGaborKernel((0, 0), sigma=0.56 * lam, theta=th, lambd=lam, gamma=0.7, psi=0)
                k -= k.mean()
                _BANK.append((k / np.abs(k).sum()).astype(np.float32))
    return _BANK


def fingerprint(gray: np.ndarray, smooth: float = 2.0, noise: float = 1.0) -> np.ndarray:
    """H x W x 9 channel fingerprint (all channels in bits-like units)."""
    g = gray.astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6) * 40.0          # gain-normalise: lighting changes cancel
    ch = [cv2.GaussianBlur(info_density(np.clip(g + 128, 0, 255)), (0, 0), smooth)]
    for k in _bank():
        r = cv2.filter2D(g, cv2.CV_32F, k)
        e = cv2.GaussianBlur(r * r, (0, 0), smooth)
        ch.append(np.log2(1.0 + e / noise))
    return np.stack(ch, -1)
