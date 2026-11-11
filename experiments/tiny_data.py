"""Training clips for the tiny video model: 8 frames, 32x32, static camera.

A natural-texture background (downsampled) with 1-2 textured sprites moving in
straight lines. Static camera on purpose: it is the case temporal token sharing
(B3) targets, and it makes "boring" vs "busy" regions unambiguous.
"""
import numpy as np
import cv2
from experiments.synth import textures, crop, sprite_mask, place


def tiny_clip(rng, T=8, S=32, pan_prob=0.0):
    # v2: smooth background (blurred texture + gradient) so "boring" regions really are
    # low-information, and crisp textured sprites. v1 used raw downsampled textures, which
    # at 32x32 are noise-like grain the tiny model could not learn in 3.5k steps.
    tex = textures()[rng.integers(len(textures()))]
    big = crop(tex, 2 * S, 2 * S, rng)
    bg = cv2.GaussianBlur(cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
    yy, xx = np.mgrid[0:S, 0:S] / S
    gx, gy = rng.uniform(-40, 40, 2)
    bg = (bg - bg.mean()) * rng.uniform(0.6, 1.2) + gx * (xx - 0.5) + gy * (yy - 0.5) + rng.uniform(80, 170)
    frames = np.repeat(bg[None], T, 0).astype(np.float32)
    if rng.random() < pan_prob:
        # R5: camera pan, 0.4-1.2 px/frame, rendered from a larger smooth background
        big2 = cv2.GaussianBlur(cv2.resize(crop(tex, 3 * S, 3 * S, rng), (2 * S, 2 * S), interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        big2 = (big2 - big2.mean()) * rng.uniform(0.6, 1.2) + rng.uniform(80, 170)
        ang = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0.4, 1.2)
        c0 = np.array([S, S], float) - np.array([np.cos(ang), np.sin(ang)]) * sp * T / 2
        for t in range(T):
            c = c0 + np.array([np.cos(ang), np.sin(ang)]) * sp * t
            frames[t] = cv2.getRectSubPix(big2.astype(np.float32), (S, S), (float(c[0]), float(c[1])))
    for _ in range(int(rng.integers(1, 3))):
        size = int(rng.integers(9, 14))
        t2 = textures()[rng.integers(len(textures()))]
        p = cv2.resize(crop(t2, 2 * size, 2 * size, rng), (size, size), interpolation=cv2.INTER_AREA)
        p = (p - p.mean()) * rng.uniform(0.8, 1.2) + bg.mean() + rng.choice([-1, 1]) * rng.uniform(50, 75)
        m = sprite_mask(size, rng.choice(["square", "ellipse"]), rng)
        v = rng.uniform(-1.6, 1.6, 2)
        p0 = rng.uniform(4, S - size - 4, 2) - v * T / 2
        for t in range(T):
            tl, ml = place((S, S), p, m, *(p0 + v * t))
            frames[t] = frames[t] * (1 - ml) + tl * ml
    return np.clip(frames, 0, 255) / 127.5 - 1.0      # [-1, 1]


def make_dataset(n, seed=0, pan_prob=0.0, T=None):
    import os
    T = int(os.environ.get("T", 8)) if T is None else T
    rng = np.random.default_rng(seed)
    return np.stack([tiny_clip(rng, T=T, pan_prob=pan_prob) for _ in range(n)]).astype(np.float32)
