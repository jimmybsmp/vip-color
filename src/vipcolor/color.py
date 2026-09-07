"""Colorimetry: sRGB transfer, XYZ/Lab conversion, and colour difference.

Everything in this project compares skin in CIELAB under a **D65** white
point, matching sRGB's native white so that the only chromatic adaptation
anywhere in the pipeline is the one the white-balance solver is explicitly
modelling.  sRGB values are for display and debug output only.

Conventions used throughout:

- "linear" means scene-linear sRGB primaries, floats nominally in [0, 1].
- "encoded" means the same values through the sRGB transfer function,
  still floats in [0, 1] (not 8-bit).
- Lab arrays carry L in [0, 100] and a/b roughly in [-128, 127].
"""

from __future__ import annotations

import numpy as np

# sRGB primaries -> CIE XYZ (D65), IEC 61966-2-1.
LINEAR_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
XYZ_TO_LINEAR_SRGB = np.linalg.inv(LINEAR_SRGB_TO_XYZ)

# CIE D65, 2 degree observer.
WHITEPOINT_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# CIE standard constants, in their exact rational form.
_LAB_EPSILON = 216.0 / 24389.0
_LAB_KAPPA = 24389.0 / 27.0


def srgb_encode(linear: np.ndarray) -> np.ndarray:
    """Scene-linear -> sRGB-encoded.  Operates elementwise on any shape."""
    linear = np.asarray(linear, dtype=np.float64)
    clipped = np.clip(linear, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def srgb_decode(encoded: np.ndarray) -> np.ndarray:
    """sRGB-encoded -> scene-linear.  Operates elementwise on any shape."""
    encoded = np.asarray(encoded, dtype=np.float64)
    clipped = np.clip(encoded, 0.0, 1.0)
    return np.where(
        clipped <= 0.04045,
        clipped / 12.92,
        np.power((clipped + 0.055) / 1.055, 2.4),
    )


def linear_srgb_to_xyz(linear: np.ndarray) -> np.ndarray:
    """(..., 3) linear sRGB -> (..., 3) XYZ."""
    linear = np.asarray(linear, dtype=np.float64)
    return linear @ LINEAR_SRGB_TO_XYZ.T


def xyz_to_linear_srgb(xyz: np.ndarray) -> np.ndarray:
    """(..., 3) XYZ -> (..., 3) linear sRGB.  May fall outside [0, 1]."""
    xyz = np.asarray(xyz, dtype=np.float64)
    return xyz @ XYZ_TO_LINEAR_SRGB.T


def xyz_to_lab(xyz: np.ndarray, whitepoint: np.ndarray = WHITEPOINT_D65) -> np.ndarray:
    """(..., 3) XYZ -> (..., 3) CIELAB."""
    xyz = np.asarray(xyz, dtype=np.float64)
    ratio = xyz / np.asarray(whitepoint, dtype=np.float64)
    f = np.where(
        ratio > _LAB_EPSILON,
        np.cbrt(np.maximum(ratio, 0.0)),
        (_LAB_KAPPA * ratio + 16.0) / 116.0,
    )
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack(
        [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)],
        axis=-1,
    )


def lab_to_xyz(lab: np.ndarray, whitepoint: np.ndarray = WHITEPOINT_D65) -> np.ndarray:
    """(..., 3) CIELAB -> (..., 3) XYZ."""
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t: np.ndarray) -> np.ndarray:
        t3 = t**3
        return np.where(t3 > _LAB_EPSILON, t3, (116.0 * t - 16.0) / _LAB_KAPPA)

    xr = finv(fx)
    # y has its own branch keyed on L, not on fy, per the CIE definition.
    yr = np.where(L > _LAB_KAPPA * _LAB_EPSILON, ((L + 16.0) / 116.0) ** 3, L / _LAB_KAPPA)
    zr = finv(fz)
    return np.stack([xr, yr, zr], axis=-1) * np.asarray(whitepoint, dtype=np.float64)


def linear_srgb_to_lab(linear: np.ndarray) -> np.ndarray:
    """(..., 3) linear sRGB -> (..., 3) CIELAB (D65)."""
    return xyz_to_lab(linear_srgb_to_xyz(linear))


def lab_to_linear_srgb(lab: np.ndarray) -> np.ndarray:
    """(..., 3) CIELAB (D65) -> (..., 3) linear sRGB.  May be out of gamut."""
    return xyz_to_linear_srgb(lab_to_xyz(lab))


def lab_to_srgb8(lab: np.ndarray) -> np.ndarray:
    """(..., 3) CIELAB -> (..., 3) uint8 sRGB, for human-readable output only."""
    encoded = srgb_encode(np.clip(lab_to_linear_srgb(lab), 0.0, 1.0))
    return np.clip(np.round(encoded * 255.0), 0, 255).astype(np.uint8)


def srgb8_to_lab(rgb8: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 sRGB -> (..., 3) CIELAB.  For hand-entered targets."""
    return linear_srgb_to_lab(srgb_decode(np.asarray(rgb8, dtype=np.float64) / 255.0))


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Plain Euclidean CIE76 difference.  Kept for debugging comparisons."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def delta_e_2000(
    lab1: np.ndarray,
    lab2: np.ndarray,
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
) -> np.ndarray:
    """CIEDE2000 colour difference, per Sharma, Wu & Dalal (2005).

    This is the difference metric used for every target/tolerance decision
    in the project; CIE76 exaggerates differences in the blue-violet and
    near-neutral regions that skin tones sit near.
    """
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2

    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    # atan2(0, 0) is defined as 0 here, per the standard.
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0
    h1p = np.where((np.abs(b1) < 1e-12) & (np.abs(a1p) < 1e-12), 0.0, h1p)
    h2p = np.where((np.abs(b2) < 1e-12) & (np.abs(a2p) < 1e-12), 0.0, h2p)

    dLp = L2 - L1
    dCp = C2p - C1p

    chroma_product = C1p * C2p
    dh = h2p - h1p
    dhp = np.where(
        chroma_product == 0.0,
        0.0,
        np.where(dh > 180.0, dh - 360.0, np.where(dh < -180.0, dh + 360.0, dh)),
    )
    dHp = 2.0 * np.sqrt(chroma_product) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_absdiff = np.abs(h1p - h2p)
    hp_bar = np.where(
        chroma_product == 0.0,
        h_sum,
        np.where(
            h_absdiff <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    d_theta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar**7
    R_C = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0**7))
    R_T = -np.sin(np.radians(2.0 * d_theta)) * R_C

    S_L = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    S_C = 1.0 + 0.045 * Cp_bar
    S_H = 1.0 + 0.015 * Cp_bar * T

    term_L = dLp / (k_L * S_L)
    term_C = dCp / (k_C * S_C)
    term_H = dHp / (k_H * S_H)

    return np.sqrt(term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H)
