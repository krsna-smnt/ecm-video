"""T2: optical flow, warping, occlusion and motion-boundary masks.

Uses OpenCV Farneback flow so it runs on any CPU with no model weights.
On a GPU machine, RAFT (torchvision.models.optical_flow) is a drop-in upgrade:
it only has to return the same H x W x 2 (dx, dy) arrays.
"""
import numpy as np
import cv2


def farneback(prev: np.ndarray, nxt: np.ndarray, winsize: int = 9) -> np.ndarray:
    """Dense flow prev -> nxt. Inputs gray 0-255. Returns H x W x 2 float32 (dx, dy)."""
    return cv2.calcOpticalFlowFarneback(
        prev.astype(np.uint8), nxt.astype(np.uint8), None,
        pyr_scale=0.5, levels=3, winsize=winsize, iterations=5,
        poly_n=5, poly_sigma=1.1, flags=0)


def warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Sample img at x + flow(x) (bilinear). Used to pull frame t+1 back onto frame t's grid."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1],
                     interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def out_of_frame(flow: np.ndarray, margin: int = 4) -> np.ndarray:
    """Pixels whose flow target leaves the frame (or lies within `margin` of the edge)."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    tx, ty = gx + flow[..., 0], gy + flow[..., 1]
    edge = (tx < margin) | (tx > w - 1 - margin) | (ty < margin) | (ty > h - 1 - margin)
    edge |= (gx < margin) | (gx > w - 1 - margin) | (gy < margin) | (gy > h - 1 - margin)
    return edge


def fwd_bwd_inconsistent(fwd: np.ndarray, bwd: np.ndarray, alpha: float = 0.05, beta: float = 0.5) -> np.ndarray:
    """Standard forward-backward check (Sundaram et al. 2010 style)."""
    bwd_w = np.stack([warp(bwd[..., i], fwd) for i in range(2)], -1)
    diff = np.sum((fwd + bwd_w) ** 2, -1)
    mag = np.sum(fwd ** 2, -1) + np.sum(bwd_w ** 2, -1)
    return diff > alpha * mag + beta


def motion_boundaries(flow: np.ndarray, thresh: float = 0.25, dilate: int = 5) -> np.ndarray:
    """Where the flow field changes sharply: the only places occlusion/disocclusion can happen.

    A uniform camera pan has no boundaries; an object moving over a background does,
    along its outline. The band is dilated so leading/trailing edges are covered.
    """
    g = np.zeros(flow.shape[:2], np.float32)
    for i in range(2):
        gx = cv2.Sobel(flow[..., i], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(flow[..., i], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        g += gx ** 2 + gy ** 2
    mb = np.sqrt(g) > thresh
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        mb = cv2.dilate(mb.astype(np.uint8), k) > 0
    return mb
