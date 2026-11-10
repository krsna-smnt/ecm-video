"""T3 + T4: source-term map and the Permanence Violation Score (PVS).

Balance law:  d(iota)/dt + div(iota v) = s.
Lagrangian form used here (follow each pixel along its flow):
    s(x, t) ~= iota(x + v(x), t+1) - iota(x, t)

In real footage s may be non-zero only where information can legally be created
or destroyed: along motion boundaries (occlusion / disocclusion) and at frame
edges (things entering / leaving). PVS sums |s| everywhere else, normalised by
the total information in the frame.

A warping-error baseline (the standard temporal-consistency metric) is included
for comparison.
"""
from dataclasses import dataclass, field
import numpy as np
import cv2

from .density import info_density
from .flow import farneback, warp, out_of_frame, fwd_bwd_inconsistent, motion_boundaries


@dataclass
class PVSResult:
    pvs_mean: float                 # mean per-frame violation (clip score)
    pvs_peak: float                 # worst frame
    frame_scores: np.ndarray        # (T-1,)
    heatmaps: np.ndarray            # (T-1, H, W) violation map in bits
    allowed: np.ndarray             # (T-1, H, W) bool, where sources are legal
    source: np.ndarray              # (T-1, H, W) signed source term s
    warp_err_mean: float = 0.0      # baseline
    warp_err_peak: float = 0.0
    warp_err_frames: np.ndarray = field(default_factory=lambda: np.zeros(0))


def conditional_surprise(f_t: np.ndarray, f_t1: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """Excess bits needed to explain frame t+1 given motion-compensated frame t.

    This is the source term in its content form: information *created* at x.
    The residual of the motion-compensated prediction is coded under a Laplacian
    whose scale is the frame's robust noise level (median |r| / ln 2), and the
    entropy of pure noise is subtracted, so legal motion costs ~0 excess bits.
    """
    r = warp(f_t1, fwd) - f_t
    b = max(np.median(np.abs(r)) / np.log(2), 0.5)          # Laplace median = b ln2
    bits = np.abs(r) / (b * np.log(2)) + np.log2(2 * b)     # -log2 density of Laplace(0,b)
    h_noise = np.log2(2 * np.e * b)                          # entropy of the noise itself
    return np.maximum(bits - h_noise, 0.0).astype(np.float32)


def score_video(frames: np.ndarray, mode: str = "conditional", smooth_sigma: float = 1.5,
                dead_zone: float = 0.35, mb_thresh: float = 1.2, mb_dilate: int = 3,
                edge_margin: int = 4, flow_win: int = 9, neighbour_boundaries: bool = True) -> PVSResult:
    """frames: T x H x W gray uint8/float (0-255). Returns PVS + baseline + maps.

    mode = "conditional": s = excess bits of the motion-compensated residual (default).
    mode = "scalar":      s = change in the scalar density along the flow (the original
                          formulation; kept for comparison, it misses content swaps).
    """
    frames = frames.astype(np.float32)
    T = len(frames)
    iota = np.stack([cv2.GaussianBlur(info_density(f), (0, 0), smooth_sigma) for f in frames])

    F = [farneback(frames[t], frames[t + 1], flow_win) for t in range(T - 1)]   # t -> t+1
    B = [farneback(frames[t + 1], frames[t], flow_win) for t in range(T - 1)]   # t+1 -> t

    def band(flow):
        """Motion-boundary band, widened by the local speed (a disoccluded strip is ~|v| wide)."""
        spd = np.sqrt((flow ** 2).sum(-1))
        d = int(min(6, np.ceil(np.percentile(spd, 99)))) + mb_dilate
        return motion_boundaries(flow, mb_thresh, d)

    scores, heat, allowed_all, src_all, werr = [], [], [], [], []
    for t in range(T - 1):
        fwd, bwd = F[t], B[t]

        # --- source term ---
        if mode == "scalar":
            s = warp(iota[t + 1], fwd) - iota[t]
        else:
            s = cv2.GaussianBlur(conditional_surprise(frames[t], frames[t + 1], fwd), (0, 0), smooth_sigma)

        # --- where creation/destruction is legal ---
        # Locate occlusion boundaries from the NEIGHBOURING frame pairs, not the pair being
        # judged: a violation corrupts the flow of its own pair (the estimator invents motion
        # to explain the vanish), which would otherwise mark the violation itself as legal.
        allowed = out_of_frame(fwd, edge_margin)
        if neighbour_boundaries:
            if t >= 1:
                allowed |= band(B[t - 1])          # frame t's motion, from the pair (t-1, t)
            if t + 1 < T - 1:
                allowed |= band(F[t + 1])          # frame t+1's motion, from the pair (t+1, t+2)
            if T == 2:
                allowed |= band(fwd)
        else:
            allowed |= band(fwd) | band(bwd)
        v = np.maximum(np.abs(s) - dead_zone, 0.0) * (~allowed)
        scores.append(v.sum() / (iota[t].sum() + 1e-6))
        heat.append(v); allowed_all.append(allowed); src_all.append(s)

        # --- baseline: photometric warping error on fwd-bwd-consistent pixels ---
        occ = fwd_bwd_inconsistent(fwd, bwd) | out_of_frame(fwd, edge_margin)
        e = np.abs(warp(frames[t + 1], fwd) - frames[t])
        werr.append(float(e[~occ].mean()) if (~occ).any() else 0.0)

    scores = np.array(scores); werr = np.array(werr)
    return PVSResult(float(scores.mean()), float(scores.max()), scores, np.stack(heat),
                     np.stack(allowed_all), np.stack(src_all),
                     float(werr.mean()), float(werr.max()), werr)


def load_video(path: str, max_side: int = 256, max_frames: int = 64, stride: int = 1) -> np.ndarray:
    """Read a video file as gray frames, resized so the longer side is <= max_side."""
    cap = cv2.VideoCapture(path)
    out, i = [], 0
    while len(out) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            h, w = g.shape
            s = max_side / max(h, w)
            if s < 1:
                g = cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            out.append(g)
        i += 1
    cap.release()
    if len(out) < 2:
        raise ValueError(f"could not read at least 2 frames from {path}")
    return np.stack(out)
