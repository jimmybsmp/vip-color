"""Synthesise a valid uncompressed DNG from an RGB image.

The point is a raw file with a *known* answer.  Given a rendered image and a
chosen illuminant cast, this builds a Bayer mosaic that a real camera would
have produced under that light, so a test can decode it and check that the
pipeline recovers what went in.  It also exercises the LibRaw path in CI
without shipping anyone's photographs.

Only what LibRaw needs is written: uncompressed, single strip, one
calibration illuminant.
"""

from __future__ import annotations

import struct
from fractions import Fraction
from pathlib import Path

import numpy as np

# TIFF field types.
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10
_TYPE_SIZE = {_BYTE: 1, _ASCII: 1, _SHORT: 2, _LONG: 4, _RATIONAL: 8, _SRATIONAL: 8}

#: sRGB primaries (D65) -> XYZ, duplicated here so tests do not depend on the
#: module under test for their own ground truth.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ]
)

#: A plausible XYZ -> camera matrix, in the shape DNG stores as ColorMatrix1.
#: Loosely Nikon-like; the exact values matter only in that decode must invert
#: whatever is written here.
NIKONISH_XYZ_TO_CAM = np.array(
    [
        [0.8500, -0.2400, -0.0900],
        [-0.4200, 1.2400, 0.1900],
        [-0.0500, 0.1000, 0.6600],
    ]
)


def _rational(value: float, signed: bool = False) -> tuple[int, int]:
    frac = Fraction(float(value)).limit_denominator(100000)
    numerator, denominator = frac.numerator, frac.denominator
    if not signed and numerator < 0:
        raise ValueError("negative value in an unsigned rational")
    return numerator, denominator


def _encode(field_type: int, values) -> tuple[int, bytes]:
    """Return (count, payload) for one tag's values."""
    if field_type == _ASCII:
        data = values.encode("ascii") + b"\x00"
        return len(data), data
    if field_type == _BYTE:
        data = bytes(values)
        return len(data), data
    if field_type == _SHORT:
        return len(values), struct.pack(f"<{len(values)}H", *values)
    if field_type == _LONG:
        return len(values), struct.pack(f"<{len(values)}I", *values)
    if field_type in (_RATIONAL, _SRATIONAL):
        signed = field_type == _SRATIONAL
        parts = [_rational(v, signed) for v in values]
        code = "i" if signed else "I"
        payload = b"".join(struct.pack(f"<2{code}", n, d) for n, d in parts)
        return len(parts), payload
    raise ValueError(f"unsupported field type {field_type}")


def write_dng(
    path: str | Path,
    cfa: np.ndarray,
    *,
    cfa_pattern: tuple[int, int, int, int] = (0, 1, 1, 2),
    xyz_to_cam: np.ndarray = NIKONISH_XYZ_TO_CAM,
    as_shot_neutral: tuple[float, float, float] = (0.5, 1.0, 0.6),
    make: str = "NIKON CORPORATION",
    model: str = "NIKON Z 8",
    white_level: int = 65535,
    black_level: int = 0,
) -> Path:
    """Write ``cfa`` (HxW uint16 Bayer data) as an uncompressed DNG."""
    path = Path(path)
    if cfa.dtype != np.uint16 or cfa.ndim != 2:
        raise ValueError("cfa must be a 2-D uint16 array")
    height, width = cfa.shape

    tags: list[tuple[int, int, list | str]] = [
        (254, _LONG, [0]),                       # NewSubfileType: full resolution
        (256, _LONG, [width]),                   # ImageWidth
        (257, _LONG, [height]),                  # ImageLength
        (258, _SHORT, [16]),                     # BitsPerSample
        (259, _SHORT, [1]),                      # Compression: none
        (262, _SHORT, [32803]),                  # PhotometricInterpretation: CFA
        (271, _ASCII, make),                     # Make
        (272, _ASCII, model),                    # Model
        (273, _LONG, [0]),                       # StripOffsets, patched below
        (274, _SHORT, [1]),                      # Orientation: normal
        (277, _SHORT, [1]),                      # SamplesPerPixel
        (278, _LONG, [height]),                  # RowsPerStrip: one strip
        (279, _LONG, [cfa.nbytes]),              # StripByteCounts
        (339, _SHORT, [1]),                      # SampleFormat: unsigned int
        (33421, _SHORT, [2, 2]),                 # CFARepeatPatternDim
        (33422, _BYTE, list(cfa_pattern)),       # CFAPattern
        (50706, _BYTE, [1, 4, 0, 0]),            # DNGVersion
        (50707, _BYTE, [1, 1, 0, 0]),            # DNGBackwardVersion
        (50708, _ASCII, f"{make} {model}"),      # UniqueCameraModel
        (50714, _SHORT, [black_level]),          # BlackLevel
        (50717, _LONG, [white_level]),           # WhiteLevel
        (50721, _SRATIONAL, list(np.asarray(xyz_to_cam, dtype=float).ravel())),  # ColorMatrix1
        (50728, _RATIONAL, list(as_shot_neutral)),  # AsShotNeutral
        (50778, _SHORT, [21]),                   # CalibrationIlluminant1: D65
    ]
    tags.sort(key=lambda item: item[0])

    header_size = 8
    ifd_size = 2 + 12 * len(tags) + 4
    overflow_start = header_size + ifd_size

    entries: list[bytes] = []
    overflow = bytearray()
    strip_offset_positions: list[int] = []

    for tag, field_type, values in tags:
        count, payload = _encode(field_type, values)
        entry = struct.pack("<HHI", tag, field_type, count)
        if len(payload) <= 4:
            entry += payload.ljust(4, b"\x00")
        else:
            entry += struct.pack("<I", overflow_start + len(overflow))
            overflow += payload
            if len(payload) % 2:
                overflow += b"\x00"
        if tag == 273:
            # Remember where StripOffsets' inline value sits so it can be
            # patched once the pixel data's position is known.
            strip_offset_positions.append(header_size + 2 + 12 * len(entries) + 8)
        entries.append(entry)

    pixel_offset = overflow_start + len(overflow)
    blob = bytearray()
    blob += struct.pack("<2sHI", b"II", 42, header_size)
    blob += struct.pack("<H", len(entries))
    for entry in entries:
        blob += entry
    blob += struct.pack("<I", 0)  # no next IFD
    blob += overflow
    blob += cfa.astype("<u2").tobytes()

    for position in strip_offset_positions:
        blob[position : position + 4] = struct.pack("<I", pixel_offset)

    path.write_bytes(bytes(blob))
    return path


def mosaic(
    linear_srgb: np.ndarray,
    *,
    xyz_to_cam: np.ndarray = NIKONISH_XYZ_TO_CAM,
    illuminant_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
    cfa_pattern: tuple[int, int, int, int] = (0, 1, 1, 2),
    white_level: int = 65535,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Turn a linear sRGB image into Bayer sensor counts.

    ``illuminant_gain`` multiplies the camera-space channels, standing in for
    the scene being lit by something other than the calibration illuminant.
    Returns the mosaic and the AsShotNeutral that describes that light.
    """
    height, width = linear_srgb.shape[:2]
    if height % 2 or width % 2:
        linear_srgb = linear_srgb[: height - height % 2, : width - width % 2]
        height, width = linear_srgb.shape[:2]

    matrix = np.asarray(xyz_to_cam, dtype=float)
    gains = np.asarray(illuminant_gain, dtype=float)
    camera = linear_srgb @ _SRGB_TO_XYZ.T @ matrix.T * gains
    counts = np.clip(camera, 0.0, 1.0) * white_level

    # CFA pattern indices are 0=R, 1=G, 2=B in row-major 2x2 order.
    cfa = np.zeros((height, width), dtype=np.float64)
    for position, channel in enumerate(cfa_pattern):
        row, column = divmod(position, 2)
        cfa[row::2, column::2] = counts[row::2, column::2, channel]

    # AsShotNeutral is the camera's *own* response to the scene white, so it
    # depends on the colour matrix as well as the illuminant gain.  Deriving
    # it from the gain alone leaves a green cast that looks like a decode bug
    # but is really a malformed test file.
    white_xyz = np.array([0.95047, 1.00000, 1.08883])
    camera_white = (matrix @ white_xyz) * gains
    neutral = tuple(float(v / camera_white[1]) for v in camera_white)
    return np.clip(np.round(cfa), 0, white_level).astype(np.uint16), neutral
