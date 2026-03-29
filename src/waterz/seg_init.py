"""2D slice-by-slice watershed for initial fragment generation.

Alternative to waterz's built-in C++ watershed, useful for anisotropic
EM volumes where z-resolution is much coarser than xy-resolution.

Uses mahotas.cwatershed (fast C implementation, float input).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mahotas
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["compute_fragments"]


def _parse_seed_method(seed_method: str) -> tuple[str, float]:
    """Parse ``"method-param"`` seed string.

    Returns ``(method_name, param)`` where *param* is:
      - grid spacing for ``"grid-N"``
      - binarization threshold for ``"maxima_distance-T"`` / ``"minima-T"``
      - depth threshold for ``"hminima-h"``
      - default when omitted

    Examples::

        "maxima_distance"      → ("maxima_distance", 0.5)
        "maxima_distance-0.3"  → ("maxima_distance", 0.3)
        "minima"               → ("minima", 0.0)
        "minima-0.4"           → ("minima", 0.4)
        "hminima"              → ("hminima", 0.1)
        "hminima-0.05"         → ("hminima", 0.05)
        "grid"                 → ("grid", 10)
        "grid-5"               → ("grid", 5)
    """
    parts = seed_method.split("-", 1)
    name = parts[0]
    defaults = {"grid": 10.0, "maxima_distance": 0.5, "minima": 0.0, "hminima": 0.1}
    if name not in defaults:
        raise ValueError(
            f"Unknown seed_method {seed_method!r}. "
            "Expected 'maxima_distance[-T]', 'minima[-T]', 'hminima[-h]', or 'grid[-N]'."
        )
    return name, float(parts[1]) if len(parts) > 1 else defaults[name]


def _get_seeds(
    boundary: NDArray[np.float64],
    method: str,
    param: float,
    next_id: int,
) -> tuple[NDArray[np.int32], int]:
    """Compute watershed seeds from a 2D boundary map."""
    if method == "grid":
        h, w = boundary.shape
        step = int(param)
        ys, xs = np.ogrid[0:h:step, 0:w:step]
        n = ys.size * xs.size
        seeds = np.zeros_like(boundary, dtype=np.int32)
        seeds[ys, xs] = np.arange(next_id, next_id + n).reshape(ys.size, xs.size)
        return seeds, n

    if method == "hminima":
        from skimage.morphology import h_minima
        markers = h_minima(boundary, param)
    elif method == "minima":
        if param > 0:
            markers = mahotas.regmin(boundary) & (boundary < param)
        else:
            markers = mahotas.regmin(boundary)
    else:  # maxima_distance
        markers = mahotas.regmax(mahotas.distance(boundary < param))

    seeds, n = mahotas.label(markers)
    seeds[seeds > 0] += next_id - 1
    return seeds, n


def compute_fragments(
    affs: NDArray,
    seed_method: str = "maxima_distance",
    aff_threshold_low: float = 0.0,
) -> NDArray[np.uint64]:
    """Compute initial over-segmentation via 2D slice-by-slice watershed.

    Parameters
    ----------
    affs : ndarray, shape ``(C, Z, Y, X)``
        Affinity predictions (C >= 3, channel order z, y, x).
        Only xy channels (indices 1, 2) are used for the boundary map.
    seed_method : str
        Seed placement strategy with optional parameter:

        - ``"maxima_distance"`` or ``"maxima_distance-T"`` — seeds at
          distance-transform maxima.  *T* is the boundary binarization
          threshold (default 0.5): ``boundary < T`` → interior.
        - ``"minima"`` or ``"minima-T"`` — seeds at boundary regional
          minima.  When *T* > 0, only minima where ``boundary < T``.
        - ``"hminima"`` or ``"hminima-h"`` — seeds at h-minima of the
          boundary map (suppress minima shallower than *h*, default 0.1).
          Reduces over-segmentation without a hard binarization threshold.
        - ``"grid"`` or ``"grid-N"`` — regular grid with spacing *N*
          (default 10).
    aff_threshold_low : float
        Zero out fragment voxels where mean xy affinity < this value
        (same semantics as the C++ watershed's low threshold).
        0 disables.  Default: 0.

    Returns
    -------
    fragments : ndarray, uint64, shape ``(Z, Y, X)``
    """
    affs = np.asarray(affs)
    if affs.ndim != 4 or affs.shape[0] < 3:
        raise ValueError(f"Expected affinities (C>=3, Z, Y, X), got {affs.shape}")

    method, param = _parse_seed_method(seed_method)

    is_uint8 = affs.dtype == np.uint8
    nz = affs.shape[1]
    fragments = np.zeros(affs.shape[1:], dtype=np.uint64)
    next_id = 1

    logger.info("compute_fragments: %d slices, seed_method=%s, aff_threshold_low=%s", nz, seed_method, aff_threshold_low)
    for z in tqdm(range(nz), desc="compute_fragments"):
        xy_z = affs[1:3, z].astype(np.float64)
        if is_uint8:
            xy_z /= 255.0
        boundary = 1.0 - xy_z.mean(axis=0)
        seeds, n = _get_seeds(boundary, method, param, next_id)
        fragments[z] = mahotas.cwatershed(boundary, seeds)
        if aff_threshold_low > 0:
            fragments[z][boundary > 1.0 - aff_threshold_low] = 0
        next_id += n

    return fragments
