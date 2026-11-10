"""Synthetic video generator with known ground truth.

Natural textures (bundled scikit-image samples, no download needed) as background
and sprites; sprites move with sub-pixel motion. Events are either LEGAL
(motion, occlusion behind another object, leaving/entering the frame, camera pan)
or VIOLATIONS (vanish/appear in open view, melt into the background, identity
swap, texture boiling) of the kind AI video generators produce.
"""
import numpy as np
import cv2
import skimage.data as skd

_TEX_NAMES = ["camera", "astronaut", "coffee", "chelsea", "rocket", "coins", "brick",
              "grass", "gravel", "moon", "hubble_deep_field", "immunohistochemistry",
              "retina", "cell", "clock"]
_TEX = None


def textures():
    global _TEX
    if _TEX is None:
        _TEX = []
        for n in _TEX_NAMES:
            im = getattr(skd, n)()
            if im.ndim == 3:
                im = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2GRAY)
            _TEX.append(im.astype(np.float32))
    return _TEX


def crop(img, h, w, rng):
    H, W = img.shape
    y = rng.integers(0, H - h + 1); x = rng.integers(0, W - w + 1)
    return img[y:y + h, x:x + w].copy()


def sprite_mask(size, kind, rng):
    m = np.zeros((size, size), np.float32)
    c = size // 2
    if kind == "square":
        m[2:-2, 2:-2] = 1
    else:
        cv2.ellipse(m, (c, c), (c - 2, int((c - 2) * rng.uniform(0.6, 1.0))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    return cv2.GaussianBlur(m, (0, 0), 0.7)


def place(layer_size, patch, mask, x, y):
    """Paste patch/mask with top-left at sub-pixel (x, y) on an H x W canvas."""
    H, W = layer_size
    M = np.float32([[1, 0, x], [0, 1, y]])
    t = cv2.warpAffine(patch, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    m = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
    return t, m


def contrast_patch(size, bg_mean, rng):
    """Texture patch whose mean differs from the background so the sprite is visible."""
    tex = textures()[rng.integers(len(textures()))]
    p = crop(tex, size, size, rng)
    p = (p - p.mean()) * rng.uniform(0.8, 1.3)
    target = bg_mean + rng.choice([-1, 1]) * rng.uniform(45, 80)
    return np.clip(p + target, 0, 255)


def make_clip(event: str, rng, T: int = 16, H: int = 128, W: int = 128, pan: bool = False,
              noise: float = 1.5, event_frame: int = 8, fade: int = 1):
    """Returns frames (T,H,W) float32, label (1 = violation), event mask (T,H,W) bool."""
    tex = textures()[rng.integers(len(textures()))]
    pad = 40
    big = crop(tex, H + 2 * pad, W + 2 * pad, rng)
    bg_mean = float(big.mean())
    pan_v = rng.uniform(-1.5, 1.5, 2) if pan else np.zeros(2)

    size = int(rng.integers(22, 34))
    patch = contrast_patch(size, bg_mean, rng)
    mask = sprite_mask(size, rng.choice(["square", "ellipse"]), rng)
    speed = rng.uniform(1.0, 2.5)
    ang = rng.uniform(0, 2 * np.pi)
    v = np.array([np.cos(ang), np.sin(ang)]) * speed
    p0 = np.array([W / 2 - size / 2, H / 2 - size / 2]) - v * (T / 2)

    occluder = None
    if event == "occlude":
        # occluder sits on the sprite's path near the middle of the clip, drawn on top
        osz = int(rng.integers(38, 50))
        opatch = contrast_patch(osz, bg_mean, rng)
        omask = np.zeros((osz, osz), np.float32); omask[1:-1, 1:-1] = 1
        mid = p0 + v * (T / 2) + size / 2 - osz / 2
        occluder = (opatch, omask, mid)
        p0 = p0 - v * 2           # make sure it fully passes behind
    if event == "exit":
        p0 = np.array([W / 2 - size / 2, H / 2 - size / 2])
        v = v / np.linalg.norm(v) * 4.0     # fast enough to leave the frame
    if event == "enter":
        v = v / np.linalg.norm(v) * 4.0
        p0 = np.array([W / 2 - size / 2, H / 2 - size / 2]) - v * (T - 2)

    swap_patch = contrast_patch(size, bg_mean, rng)
    frames, emask = [], []
    for t in range(T):
        off = pad + pan_v * t
        bg = cv2.getRectSubPix(big, (W, H), (off[0] + W / 2, off[1] + H / 2))
        alpha = 1.0
        cur_patch, cur_mask = patch, mask
        e_on = False
        if event == "vanish" and t >= event_frame:
            alpha = max(0.0, 1.0 - (t - event_frame + 1) / fade); e_on = t < event_frame + fade + 1
        if event == "appear":
            alpha = 0.0 if t < event_frame else min(1.0, (t - event_frame + 1) / fade)
            e_on = event_frame - 1 <= t <= event_frame + fade
        if event == "melt" and t >= event_frame:
            k = min(t - event_frame + 1, 6)
            cur_patch = cv2.GaussianBlur(patch, (0, 0), 0.8 * k)
            cur_mask = cv2.GaussianBlur(mask, (0, 0), 1.0 * k)
            alpha = max(0.0, 1 - k / 6); e_on = t <= event_frame + 6
        if event == "swap" and t >= event_frame:
            cur_patch = swap_patch; e_on = t <= event_frame
        if event == "boil" and t >= event_frame:
            cur_patch = contrast_patch(size, bg_mean, rng); e_on = True

        pos = p0 + v * t + pan_v * 0  # sprite moves in screen space
        tl, ml = place((H, W), cur_patch, cur_mask, pos[0], pos[1])
        f = bg * (1 - alpha * ml) + tl * alpha * ml
        if occluder is not None:
            ot, om = place((H, W), occluder[0], occluder[1], occluder[2][0], occluder[2][1])
            f = f * (1 - om) + ot * om
        f = f + rng.normal(0, noise, f.shape)
        frames.append(np.clip(f, 0, 255))
        _, mfull = place((H, W), patch, mask, pos[0], pos[1])
        emask.append((mfull > 0.3) & e_on)
    label = int(event in ("vanish", "appear", "melt", "swap", "boil"))
    return np.stack(frames).astype(np.float32), label, np.stack(emask)


LEGAL = ["none", "occlude", "exit", "enter"]
VIOLATIONS = ["vanish", "appear", "melt", "swap", "boil"]
