"""Training clips for the tiny video model: 8 frames, 32x32, static camera.

A natural-texture background (downsampled) with 1-2 textured sprites moving in
straight lines. Static camera on purpose: it is the case temporal token sharing
(B3) targets, and it makes "boring" vs "busy" regions unambiguous.
"""
import numpy as np
import cv2
from experiments.synth import textures, crop, sprite_mask, place


def tiny_clip(rng, T=8, S=32):
    tex = textures()[rng.integers(len(textures()))]
    big = crop(tex, 4 * S, 4 * S, rng)
    bg = cv2.resize(big, (S, S), interpolation=cv2.INTER_AREA)
    bg = (bg - bg.mean()) * rng.uniform(0.3, 0.7) + rng.uniform(70, 180)   # calmer background
    frames = np.repeat(bg[None], T, 0).astype(np.float32)
    for _ in range(int(rng.integers(1, 3))):
        size = int(rng.integers(7, 12))
        t2 = textures()[rng.integers(len(textures()))]
        p = cv2.resize(crop(t2, 4 * size, 4 * size, rng), (size, size), interpolation=cv2.INTER_AREA)
        p = (p - p.mean()) * rng.uniform(0.8, 1.2) + bg.mean() + rng.choice([-1, 1]) * rng.uniform(50, 75)
        m = sprite_mask(size, rng.choice(["square", "ellipse"]), rng)
        v = rng.uniform(-1.6, 1.6, 2)
        p0 = rng.uniform(4, S - size - 4, 2) - v * T / 2
        for t in range(T):
            tl, ml = place((S, S), p, m, *(p0 + v * t))
            frames[t] = frames[t] * (1 - ml) + tl * ml
    return np.clip(frames, 0, 255) / 127.5 - 1.0      # [-1, 1]


def make_dataset(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([tiny_clip(rng) for _ in range(n)]).astype(np.float32)
