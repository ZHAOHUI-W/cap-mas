from __future__ import annotations


class CAPXDepthDecoder:
    """Decode CAP-X depth arrays from either memory or shared artifacts."""

    def __init__(self, *, subsample: int = 8, near_m: float = 0.015, far_m: float = 20.0) -> None:
        if subsample <= 0:
            raise ValueError("depth subsample must be positive")
        if not 0.0 < near_m < far_m:
            raise ValueError("depth clip range must satisfy 0 < near_m < far_m")
        self.subsample = subsample
        self.near_m = near_m
        self.far_m = far_m

    def decode(self, frame, depth, artifact_store):
        import numpy as np

        value = getattr(artifact_store, "get")(depth)
        depth_array = np.asarray(value).squeeze()
        if depth_array.ndim != 2:
            raise ValueError(f"CAP-X depth must be 2D, got shape={depth_array.shape}")
        intrinsics = np.asarray(frame.camera.intrinsics, dtype=float)
        if intrinsics.size != 9:
            raise ValueError("CAP-X camera intrinsics must contain 9 values")
        intrinsics = intrinsics.reshape(3, 3)
        sampled = depth_array[:: self.subsample, :: self.subsample].astype(float, copy=False)
        height, width = sampled.shape
        yy, xx = np.indices((height, width), dtype=float)
        scale = float(self.subsample)
        fx = intrinsics[0, 0] / scale
        fy = intrinsics[1, 1] / scale
        cx = intrinsics[0, 2] / scale
        cy = intrinsics[1, 2] / scale
        if fx == 0.0 or fy == 0.0:
            raise ValueError("CAP-X camera intrinsics contain zero focal length")
        valid = (
            np.isfinite(sampled)
            & (sampled >= self.near_m)
            & (sampled <= self.far_m)
        )
        z = sampled[valid]
        x = (xx[valid] - cx) * z / fx
        y = (yy[valid] - cy) * z / fy
        return tuple((float(px), float(py), float(pz)) for px, py, pz in zip(x, y, z))
