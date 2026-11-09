"""Shared training/inference shoulder normalization; missing joints remain zero."""
import numpy as np

NORMALIZATION = 'shoulder_xyz_v1'


def normalize_features(array):
    array = np.asarray(array, dtype=np.float32)
    if array.shape[-1] != 180:
        raise ValueError('Expected 180 features')
    points = array.reshape(*array.shape[:-1], 60, 3).copy()
    active = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    points[~active] = 0
    left, right = points[..., 11, :], points[..., 12, :]
    scale = np.linalg.norm(left - right, axis=-1)
    valid = active[..., 11] & active[..., 12] & np.isfinite(scale) & (scale >= 1e-3)
    center = (left + right) * .5
    safe_scale = np.where(valid, scale, 1.)
    output = (points - center[..., None, :]) / safe_scale[..., None, None]
    output[~(active & valid[..., None])] = 0
    return output.reshape(array.shape).astype(np.float32)


def augment_features(x):
    """Training only; random transforms never mutate the cached tensor."""
    import torch
    points = x.clone().reshape(64, 60, 3)
    active = points.ne(0).any(dim=-1)
    scale = torch.empty(()).uniform_(.9, 1.1)
    translation = torch.empty(3).uniform_(-.05, .05)
    transformed = points * scale + translation + torch.randn_like(points) * .015
    points = torch.where(active.unsqueeze(-1), transformed, torch.zeros_like(points))
    # Mask 7–9 positions (10–15% of T=64); padding can also be selected.
    count = int(torch.randint(7, 10, ()).item())
    masked = torch.randperm(64)[:count]
    saved = points.clone()
    points[masked] = 0
    if not points.any() and saved.any():
        first = int(saved.ne(0).any(dim=-1).any(dim=-1).nonzero()[0])
        points[first] = saved[first]
    return points.reshape(64, 180)
