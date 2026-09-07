"""Neutral RAW rendering.

The whole tool rests on one idea: two frames of the same face, rendered the
same way, should produce comparable numbers.  That only holds if the render
is defined by us rather than by whatever the camera was set to, so every
decode here goes through :func:`render_raw` with an explicit white balance
and no automatic anything.

The rendering choices, and why:

- ``use_camera_wb=False``, ``use_auto_wb=False``, explicit ``user_wb`` --
  the as-shot white balance is exactly the variable we are solving for, so
  it must not leak into the sample.
- The neutral anchor is the camera's own **daylight** preset rather than
  unity multipliers.  Unity means "raw sensor counts", which differ per body,
  so a Z8 and a D810 would land on different "neutral" renders and poison a
  profile built from both.  The daylight preset is a per-body constant that
  puts all three on the same nominal illuminant.
- ``gamma=(1, 1)`` and ``output_bps=16`` -- white balance is multiplicative,
  so the fit in :mod:`vipcolor.solve` is only linear if the pixels are.
  16 bits keeps the shadows of an underexposed cheek from quantising.
- ``no_auto_bright=True`` -- LibRaw's auto-brighten rescales per file, which
  would make exposure differences between frames invisible to us.  We want
  to see them.
- ``output_color=sRGB`` -- LibRaw applies each body's own colour matrix on
  the way out, so Z8/Z7/D810 all arrive in one known space.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rawpy

# LibRaw white balance is four multipliers: R, G1, B, G2.
WBMultipliers = tuple[float, float, float, float]

#: Sentinel values accepted by ``--neutral``.
NEUTRAL_SOURCES = ("daylight", "asshot", "unity")


@dataclass(frozen=True)
class RawInfo:
    """What we know about a RAW file without rendering it."""

    path: Path
    camera_make: str | None
    camera_model: str | None
    daylight_wb: WBMultipliers
    camera_wb: WBMultipliers
    raw_size: tuple[int, int]
    color_desc: str
    rgb_xyz_matrix: np.ndarray

    @property
    def camera(self) -> str:
        parts = [p for p in (self.camera_make, self.camera_model) if p]
        return " ".join(parts) if parts else "unknown camera"


def normalise_wb(wb) -> WBMultipliers:
    """Coerce LibRaw white balance into four positive multipliers, G1 == 1.

    LibRaw hands back a fourth (second green) multiplier that is zero on
    most Bayer sensors, meaning "same as G1".  Passing that zero straight
    back into ``user_wb`` renders a black channel, so it is fixed here once
    rather than at each call site.
    """
    values = [float(v) for v in wb]
    if len(values) != 4:
        raise ValueError(f"expected 4 white balance multipliers, got {len(values)}")
    r, g1, b, g2 = values
    if g2 <= 0.0:
        g2 = g1
    if min(r, g1, b, g2) <= 0.0:
        raise ValueError(f"white balance multipliers must be positive, got {values}")
    return (r / g1, 1.0, b / g1, g2 / g1)


def _read_tiff_make_model(path: Path) -> tuple[str | None, str | None]:
    """Pull Make/Model out of a TIFF-structured RAW's first IFD.

    rawpy exposes no camera identity, and NEF is TIFF underneath, so this
    reads the two tags directly.  Best effort: any malformed file just
    yields ``(None, None)`` rather than failing the render.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(8)
            if len(header) < 8:
                return (None, None)
            if header[:2] == b"II":
                endian = "<"
            elif header[:2] == b"MM":
                endian = ">"
            else:
                return (None, None)
            (magic,) = struct.unpack(endian + "H", header[2:4])
            if magic != 42:  # 42 is classic TIFF; BigTIFF (43) is not used by NEF.
                return (None, None)
            (ifd_offset,) = struct.unpack(endian + "I", header[4:8])

            handle.seek(ifd_offset)
            (count,) = struct.unpack(endian + "H", handle.read(2))
            wanted = {0x010F: "make", 0x0110: "model"}
            found: dict[str, str] = {}
            for _ in range(count):
                entry = handle.read(12)
                if len(entry) < 12:
                    break
                tag, field_type, length = struct.unpack(endian + "HHI", entry[:8])
                if tag not in wanted or field_type != 2:  # 2 == ASCII
                    continue
                if length <= 4:
                    raw_bytes = entry[8 : 8 + length]
                else:
                    (offset,) = struct.unpack(endian + "I", entry[8:12])
                    here = handle.tell()
                    handle.seek(offset)
                    raw_bytes = handle.read(length)
                    handle.seek(here)
                text = raw_bytes.split(b"\x00", 1)[0].decode("ascii", "replace").strip()
                if text:
                    found[wanted[tag]] = text
            return (found.get("make"), found.get("model"))
    except (OSError, struct.error):
        return (None, None)


def read_info(path: str | Path) -> RawInfo:
    """Open a RAW file and report its identity and white balance presets."""
    path = Path(path)
    make, model = _read_tiff_make_model(path)
    with rawpy.imread(str(path)) as raw:
        sizes = raw.sizes
        return RawInfo(
            path=path,
            camera_make=make,
            camera_model=model,
            daylight_wb=normalise_wb(raw.daylight_whitebalance),
            camera_wb=normalise_wb(raw.camera_whitebalance),
            raw_size=(int(sizes.width), int(sizes.height)),
            color_desc=raw.color_desc.decode("ascii", "replace"),
            rgb_xyz_matrix=np.array(raw.rgb_xyz_matrix, dtype=np.float64),
        )


def neutral_wb(info: RawInfo, source: str = "daylight") -> WBMultipliers:
    """The multipliers that define this body's neutral render."""
    if source == "daylight":
        return info.daylight_wb
    if source == "asshot":
        return info.camera_wb
    if source == "unity":
        return (1.0, 1.0, 1.0, 1.0)
    raise ValueError(f"unknown neutral source {source!r}; expected one of {NEUTRAL_SOURCES}")


def render_raw(
    path: str | Path,
    wb: WBMultipliers,
    *,
    half_size: bool = True,
) -> np.ndarray:
    """Render a RAW file to scene-linear sRGB-primaries float, HxWx3 in [0, 1].

    ``half_size`` takes one pixel per Bayer quad instead of demosaicing.  It
    is four times faster and introduces no interpolated colour, which is why
    it is the default for sampling and solving; the result is still 4128 px
    wide on a Z8.
    """
    wb = normalise_wb(wb)
    with rawpy.imread(str(path)) as raw:
        rendered = raw.postprocess(
            use_camera_wb=False,
            use_auto_wb=False,
            user_wb=list(wb),
            output_color=rawpy.ColorSpace.sRGB,
            output_bps=16,
            gamma=(1.0, 1.0),
            no_auto_bright=True,
            highlight_mode=rawpy.HighlightMode.Clip,
            half_size=half_size,
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    return rendered.astype(np.float64) / 65535.0
