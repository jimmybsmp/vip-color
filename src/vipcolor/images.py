"""Loading anything -- RAW or rendered -- into one scene-linear form.

A reference JPEG and a fresh NEF have to produce comparable Lab numbers, so
both arrive here and leave as scene-linear sRGB-primaries float.  The two
traps this module exists to avoid:

- **Untagged assumptions.** A Lightroom export is very often Adobe RGB or
  ProPhoto, not sRGB.  Read as sRGB it would report skin several delta-E
  off, silently, and that error would be baked into the target profile.
  Tagged files are converted through littleCMS; untagged files are assumed
  sRGB and say so, so the assumption is visible in the output.
- **Orientation.** A portrait frame stored with an EXIF rotation flag has to
  be uprighted before it reaches the face detector.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from . import raw as rawmod
from .color import srgb_decode, srgb_encode

#: Extensions we hand to LibRaw.  Nikon first; the rest cost nothing.
RAW_SUFFIXES = frozenset(
    {".nef", ".nrw", ".dng", ".arw", ".cr2", ".cr3", ".raf", ".orf", ".rw2", ".pef", ".srw"}
)
RGB_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})

#: Long edge the face detector sees.  Landmarks come back normalised, so
#: this only trades detector cost against detector accuracy, never sampling
#: precision -- patches are always measured on the full-resolution linear image.
DETECT_LONG_EDGE = 1280


@dataclass(frozen=True)
class LinearImage:
    """Scene-linear sRGB-primaries pixels plus how they got that way."""

    linear: np.ndarray
    path: Path
    kind: str
    colour_note: str
    camera: str | None = None
    wb: rawmod.WBMultipliers | None = None

    @property
    def size(self) -> tuple[int, int]:
        return (self.linear.shape[1], self.linear.shape[0])

    def encoded8(self) -> np.ndarray:
        """sRGB-encoded uint8, for the face detector and debug output."""
        return np.clip(np.round(srgb_encode(self.linear) * 255.0), 0, 255).astype(np.uint8)

    def detector_image(self) -> tuple[np.ndarray, float]:
        """A downscaled uint8 copy for detection, plus the scale applied."""
        import cv2

        encoded = self.encoded8()
        height, width = encoded.shape[:2]
        longest = max(height, width)
        if longest <= DETECT_LONG_EDGE:
            return encoded, 1.0
        scale = DETECT_LONG_EDGE / longest
        resized = cv2.resize(
            encoded,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale


def is_raw(path: str | Path) -> bool:
    return Path(path).suffix.lower() in RAW_SUFFIXES


def is_supported(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in RAW_SUFFIXES or suffix in RGB_SUFFIXES


def _load_rgb(path: Path) -> LinearImage:
    """Load a rendered image, honouring its ICC profile and EXIF rotation."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
        profile_bytes = image.info.get("icc_profile")

        if profile_bytes:
            source = ImageCms.getOpenProfile(io.BytesIO(profile_bytes))
            name = (ImageCms.getProfileDescription(source) or "").strip()
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            try:
                image = ImageCms.profileToProfile(
                    image,
                    source,
                    ImageCms.createProfile("sRGB"),
                    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                    outputMode="RGB",
                )
                note = f"converted from embedded profile {name!r} to sRGB"
            except ImageCms.PyCMSError as error:
                image = image.convert("RGB")
                note = f"embedded profile {name!r} unusable ({error}); read as sRGB"
        else:
            image = image.convert("RGB")
            note = "no embedded ICC profile; assumed sRGB"

        encoded = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0

    return LinearImage(
        linear=srgb_decode(encoded),
        path=path,
        kind="rgb",
        colour_note=note,
    )


def _load_raw(
    path: Path,
    *,
    neutral: str = "daylight",
    wb: rawmod.WBMultipliers | None = None,
    half_size: bool = True,
) -> LinearImage:
    info = rawmod.read_info(path)
    multipliers = rawmod.normalise_wb(wb) if wb is not None else rawmod.neutral_wb(info, neutral)
    linear = rawmod.render_raw(path, multipliers, half_size=half_size)
    described = (
        f"linear sRGB render, wb={'/'.join(f'{m:.4f}' for m in multipliers)}"
        f" ({neutral if wb is None else 'explicit'})"
    )
    return LinearImage(
        linear=linear,
        path=path,
        kind="raw",
        colour_note=described,
        camera=info.camera,
        wb=multipliers,
    )


def load_linear(
    path: str | Path,
    *,
    neutral: str = "daylight",
    wb: rawmod.WBMultipliers | None = None,
    half_size: bool = True,
) -> LinearImage:
    """Load any supported file as scene-linear sRGB-primaries float."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such file: {path}")
    if is_raw(path):
        return _load_raw(path, neutral=neutral, wb=wb, half_size=half_size)
    if path.suffix.lower() in RGB_SUFFIXES:
        return _load_rgb(path)
    raise ValueError(f"unsupported file type: {path.suffix or path.name}")


def iter_images(target: str | Path, *, raw_only: bool = False) -> list[Path]:
    """A single file, or every supported image directly inside a folder."""
    target = Path(target)
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"no such file or folder: {target}")
    predicate = is_raw if raw_only else is_supported
    return sorted(p for p in target.iterdir() if p.is_file() and predicate(p))
