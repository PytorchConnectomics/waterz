"""Uint8 affinity helpers shared by decode_waterz and LargeDecodeRunner."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np


def float_to_uint8(value: float) -> int:
    """Scale a float [0, 1] parameter to uint8 [0, 255].

    If the value is already > 1 (i.e. pre-scaled), return as int.
    """
    if isinstance(value, float) and value <= 1.0:
        return int(round(value * 255))
    return int(value)


def prepare_affinities(
    affs: np.ndarray,
    *,
    channel_order: str = "zyx",
    use_aff_uint8: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Normalise affinity array dtype, channel order, and memory layout.

    Parameters
    ----------
    affs : ndarray, shape ``(C, Z, Y, X)``
        Raw affinity predictions (first 3 channels used).
    channel_order : ``"zyx"`` or ``"xyz"``
        Channel ordering of the input.  Waterz C++ expects zyx.
    use_aff_uint8 : bool
        If True and affs are float, convert to uint8.

    Returns
    -------
    affs : ndarray, float32 or uint8, C-contiguous, zyx order
    is_uint8 : bool
    """
    affs = affs[:3]

    # Float → uint8 conversion if requested
    if use_aff_uint8 and affs.dtype != np.uint8:
        affs = np.clip(affs, 0, 1)
        affs = (affs * 255).astype(np.uint8)

    is_uint8 = affs.dtype == np.uint8
    if not is_uint8:
        affs = affs.astype(np.float32, copy=False)

    # Transpose to zyx order
    channel_order = channel_order.lower()
    if channel_order == "xyz":
        affs = affs[[2, 1, 0]]
    elif channel_order != "zyx":
        raise ValueError(f"Unknown channel_order '{channel_order}'. Expected 'xyz' or 'zyx'.")

    if not affs.flags["C_CONTIGUOUS"]:
        affs = np.ascontiguousarray(affs)

    return affs, is_uint8


def scale_thresholds(
    thresholds: Union[float, Sequence[float]],
    is_uint8: bool,
) -> list:
    """Normalise thresholds to a sorted list, scaled for uint8 if needed."""
    if isinstance(thresholds, (int, float)):
        out = [float(thresholds)]
    else:
        out = sorted(float(t) for t in thresholds)
    if is_uint8:
        out = [float_to_uint8(t) for t in out]
    return out


def scale_aff_threshold(
    aff_threshold: Tuple[float, float],
    is_uint8: bool,
) -> Tuple[float, float]:
    """Scale (low, high) affinity thresholds for uint8 if needed."""
    if is_uint8:
        return (float_to_uint8(aff_threshold[0]), float_to_uint8(aff_threshold[1]))
    return (float(aff_threshold[0]), float(aff_threshold[1]))
