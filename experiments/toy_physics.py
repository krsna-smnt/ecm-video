"""Ground-truth physics checker for the tiny model's toy world.

The training distribution obeys known laws: a perfectly static background, and 1-2 rigid
sprites that keep their size and texture and move at constant velocity. For any clip:

  bg_flicker    mean |frame - median background| on pixels that are background in every frame
  area_drift    relative std of the moving-object area across frames (melting / vanishing)
  vel_jerk      RMS deviation of the object centroid from a constant-velocity fit (px)
  app_drift     mean change of the object's pixel intensity distribution between frames
  valid         share of clips whose object stays detectable in every frame

Real clips should be ~0 on all of these (up to the sub-pixel rendering), so the numbers
are directly interpretable as violations of the toy world's laws.
"""
import numpy as np
from scipy import ndimage


def clip_physics(v, thr=0.12):
    """v: T x S x S in [-1, 1]."""
    bg = np.median(v, axis=0)
    dev = np.abs(v - bg[None])
    mov = dev > thr
    mov = np.stack([ndimage.binary_opening(m, iterations=1) for m in mov])
    always_bg = ~mov.any(0)
    flick = float(dev[:, always_bg].mean()) if always_bg.any() else np.nan
    areas = mov.sum((1, 2)).astype(float)
    if (areas == 0).all():
        return dict(bg_flicker=flick, area_drift=np.nan, vel_jerk=np.nan, app_drift=np.nan, valid=0.0)
    valid = float((areas > 0).all())
    area_drift = float(areas.std() / (areas.mean() + 1e-6))
    cents = []
    for m in mov:
        ys, xs = np.nonzero(m)
        cents.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
    c = np.array(cents); ok = ~np.isnan(c[:, 0]); t = np.arange(len(c))
    jerk = np.nan
    if ok.sum() >= 3:
        res = []
        for d in range(2):
            p = np.polyfit(t[ok], c[ok, d], 1); res.append(c[ok, d] - np.polyval(p, t[ok]))
        jerk = float(np.sqrt(np.mean(np.square(res))))
    hists = [np.histogram(f[m], bins=16, range=(-1, 1), density=True)[0] if m.any() else None for f, m in zip(v, mov)]
    diffs = [0.5 * np.abs(a - b).sum() * (2 / 16) for a, b in zip(hists[:-1], hists[1:]) if a is not None and b is not None]
    return dict(bg_flicker=flick, area_drift=area_drift, vel_jerk=jerk,
                app_drift=float(np.mean(diffs)) if diffs else np.nan, valid=valid)


def physics(vids):
    rows = [clip_physics(v) for v in vids]
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
