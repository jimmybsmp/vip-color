"""The target: what this subject's skin is supposed to look like.

A profile is built by measuring a folder of frames whose colour is already
considered correct, and it is what a new frame gets solved towards.

Two things about it are less obvious than they look.

**A profile belongs to a rendering space.** Reference JPEGs have been
through Camera Raw -- baseline exposure, tone curve, a grade -- while a
neutral RAW render deliberately has not.  On real files that is about two
stops.  Comparing a target measured in one space against a sample measured
in the other is not a small error: it is tens of delta-E, and a solver
handed that difference will try to fix it by moving the white balance.  So
the space is recorded, and mixing spaces in one profile is refused.

**Hue is the part that survives.** Measured across four stops of exposure
on a real NEF, hue angle moves under a third of a degree while absolute Lab
moves 40 delta-E.  Hue is also most of what white balance actually
controls.  So a profile carries hue as a first-class circular statistic,
not as an afterthought derived from a* and b*.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .color import delta_e_2000, lab_to_srgb8, srgb8_to_lab
from .faces import SkinSample

#: Bumped when the on-disk shape changes in a way older files cannot satisfy.
PROFILE_VERSION = 1

#: Quality verdicts, worst to best, and what each contributes to the mean.
QUALITY_ORDER = ("poor", "marginal", "good")
QUALITY_WEIGHT = {"good": 1.0, "marginal": 0.4, "poor": 0.0}

#: A tolerance never tightens below this: it is roughly the point where a
#: colour difference stops being visible on skin under ordinary viewing.
MIN_TOLERANCE_DE = 2.0


class ProfileError(Exception):
    """Raised when a profile cannot be built or is being used wrongly."""


def render_space_of(sample: SkinSample) -> str:
    """Which rendering space a sample was measured in."""
    return "linear" if sample.path.suffix.lower() not in _RENDERED_SUFFIXES else "rendered"


_RENDERED_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})


def circular_mean_degrees(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted mean and spread of angles, in degrees.

    Hue is an angle, so averaging it arithmetically is wrong at the wrap
    point.  Skin never sits near 0/360, but a mean that is only correct for
    convenient inputs is a trap for whoever reuses this next.
    """
    radians = np.radians(np.asarray(angles, dtype=np.float64))
    total = float(np.sum(weights))
    if total <= 0:
        raise ProfileError("no weight to average")
    mean_cos = float(np.sum(np.cos(radians) * weights) / total)
    mean_sin = float(np.sum(np.sin(radians) * weights) / total)
    mean = math.degrees(math.atan2(mean_sin, mean_cos)) % 360.0

    resultant = math.hypot(mean_cos, mean_sin)
    if resultant >= 1.0:
        spread = 0.0
    elif resultant <= 1e-12:
        spread = 180.0
    else:
        spread = math.degrees(math.sqrt(-2.0 * math.log(resultant)))
    return mean, spread


def hue_of(lab) -> float:
    """CIELAB hue angle in degrees."""
    lab = np.asarray(lab, dtype=np.float64)
    return float(np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0)


def chroma_of(lab) -> float:
    lab = np.asarray(lab, dtype=np.float64)
    return float(np.hypot(lab[..., 1], lab[..., 2]))


def hue_difference(a: float, b: float) -> float:
    """Smallest angle between two hues, in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class ProfileSource:
    """One reference frame's contribution to the profile."""

    file: str
    lab: list[float]
    hue: float
    chroma: float
    quality: str
    weight: float
    face_scale: float
    pose_yaw: float
    chroma_spread: float
    used: bool
    note: str = ""


@dataclass
class SkinProfile:
    """The target skin tone, with the evidence it was built from."""

    version: int
    created: str
    render_space: str
    lab: list[float]
    lab_std: list[float]
    hue: float
    hue_std: float
    chroma: float
    chroma_std: float
    lightness: float
    lightness_std: float
    tolerance_de: float
    hue_tolerance_deg: float
    samples_used: int
    samples_seen: int
    sources: list[ProfileSource] = field(default_factory=list)
    origin: str = "references"
    note: str = ""

    @property
    def lab_array(self) -> np.ndarray:
        return np.array(self.lab, dtype=np.float64)

    @property
    def srgb8(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in lab_to_srgb8(self.lab_array))

    def distance(self, lab) -> float:
        """CIEDE2000 from this profile's centre."""
        return float(delta_e_2000(np.asarray(lab, dtype=np.float64), self.lab_array))

    def hue_distance(self, lab) -> float:
        """Hue-angle difference from this profile's centre, in degrees."""
        return hue_difference(hue_of(lab), self.hue)

    def within(self, lab) -> bool:
        return self.distance(lab) <= self.tolerance_de

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SkinProfile:
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ProfileError(f"{path} is not valid JSON: {error}") from error
        if raw.get("version") != PROFILE_VERSION:
            raise ProfileError(
                f"{path} is profile version {raw.get('version')}, this build reads "
                f"version {PROFILE_VERSION}. Rebuild it with 'vipcolor build-profile'."
            )
        sources = [ProfileSource(**entry) for entry in raw.pop("sources", [])]
        return cls(**raw, sources=sources)


def build_profile(
    samples: list[SkinSample],
    *,
    min_quality: str = "marginal",
    tolerance: float | None = None,
    notes: dict[Path, str] | None = None,
) -> SkinProfile:
    """Combine measured reference frames into one target.

    Frames below ``min_quality`` are recorded but contribute nothing.  That
    matters more than it sounds: on a real reference set, one
    three-quarter-profile frame sat 15 delta-E from the rest, and averaging
    it in would have moved the target towards a value no correctly graded
    frame ever had.
    """
    if not samples:
        raise ProfileError("no reference samples to build a profile from")
    if min_quality not in QUALITY_ORDER:
        raise ProfileError(f"unknown quality floor {min_quality!r}; expected {QUALITY_ORDER}")

    spaces = {render_space_of(sample) for sample in samples}
    if len(spaces) > 1:
        raise ProfileError(
            "reference set mixes rendered files (JPEG/PNG/TIFF) with RAW files. "
            "A rendered frame carries Camera Raw's exposure and tone curve and a "
            "neutral RAW render does not -- on real files that is about two stops, "
            "so one target cannot describe both. Build separate profiles."
        )
    render_space = spaces.pop()

    floor = QUALITY_ORDER.index(min_quality)
    notes = notes or {}
    sources: list[ProfileSource] = []
    used: list[SkinSample] = []
    weights: list[float] = []

    for sample in samples:
        keep = QUALITY_ORDER.index(sample.quality) >= floor
        weight = QUALITY_WEIGHT[sample.quality] if keep else 0.0
        if weight <= 0:
            keep = False
        sources.append(
            ProfileSource(
                file=str(sample.path),
                lab=[round(float(v), 4) for v in sample.lab],
                hue=round(hue_of(sample.lab), 3),
                chroma=round(chroma_of(sample.lab), 3),
                quality=sample.quality,
                weight=weight,
                face_scale=round(float(sample.face_scale), 1),
                pose_yaw=round(float(sample.pose_yaw), 4),
                chroma_spread=round(float(sample.chroma_spread), 3),
                used=keep,
                note=notes.get(sample.path, ""),
            )
        )
        if keep:
            used.append(sample)
            weights.append(weight)

    if not used:
        raise ProfileError(
            f"every reference frame scored below '{min_quality}'. "
            "Lower the floor with --min-quality, or check the overlays: a turned "
            "head or a face only a few dozen pixels across cannot anchor a target."
        )

    labs = np.array([sample.lab for sample in used], dtype=np.float64)
    weight_array = np.array(weights, dtype=np.float64)

    mean_lab = np.average(labs, axis=0, weights=weight_array)
    if len(used) > 1:
        variance = np.average((labs - mean_lab) ** 2, axis=0, weights=weight_array)
        std_lab = np.sqrt(variance)
    else:
        std_lab = np.zeros(3)

    # The centre is the mean Lab, and hue and chroma are read off it, so the
    # profile is self-consistent: hue_of(profile.lab) == profile.hue exactly.
    # The circular statistics are still used, but for the *spread* -- taking a
    # standard deviation of hue arithmetically would be wrong at the wrap
    # point, and the spread is what sets the tolerance.
    hues = np.array([hue_of(lab) for lab in labs])
    _, hue_spread = circular_mean_degrees(hues, weight_array)
    hue_mean = hue_of(mean_lab)
    chromas = np.array([chroma_of(lab) for lab in labs])
    chroma_mean = chroma_of(mean_lab)
    chroma_std = (
        float(np.sqrt(np.average((chromas - chroma_mean) ** 2, weights=weight_array)))
        if len(used) > 1
        else 0.0
    )
    lightness_mean = float(mean_lab[0])
    lightness_std = float(std_lab[0])

    distances = np.array([float(delta_e_2000(lab, mean_lab)) for lab in labs])
    if tolerance is None:
        if len(used) > 1:
            spread = float(distances.mean() + 2.0 * distances.std())
        else:
            spread = MIN_TOLERANCE_DE
        tolerance = max(MIN_TOLERANCE_DE, spread)

    return SkinProfile(
        version=PROFILE_VERSION,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        render_space=render_space,
        lab=[round(float(v), 4) for v in mean_lab],
        lab_std=[round(float(v), 4) for v in std_lab],
        hue=round(hue_mean, 3),
        hue_std=round(hue_spread, 3),
        chroma=round(chroma_mean, 3),
        chroma_std=round(chroma_std, 3),
        lightness=round(lightness_mean, 3),
        lightness_std=round(lightness_std, 3),
        tolerance_de=round(float(tolerance), 3),
        hue_tolerance_deg=round(max(1.0, float(hue_spread) * 2.0), 3),
        samples_used=len(used),
        samples_seen=len(samples),
        sources=sources,
        origin="references",
    )


def manual_profile(
    *,
    lab: tuple[float, float, float] | None = None,
    rgb: tuple[int, int, int] | None = None,
    rgb_range: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
    tolerance: float | None = None,
    render_space: str = "rendered",
) -> SkinProfile:
    """A target typed in by hand rather than measured.

    An sRGB range is accepted because that is how the eye-and-eyedropper
    version of this judgement is normally expressed.  Its midpoint becomes
    the target and half its diagonal becomes the tolerance, so "the face
    runs about 200-215 red" turns into a real delta-E bound instead of
    being silently treated as an exact value.
    """
    provided = [value is not None for value in (lab, rgb, rgb_range)]
    if sum(provided) != 1:
        raise ProfileError("give exactly one of a Lab target, an sRGB target, or an sRGB range")

    if lab is not None:
        centre = np.array(lab, dtype=np.float64)
        derived_tolerance = MIN_TOLERANCE_DE
        note = f"target given by hand as Lab {tuple(round(float(v), 2) for v in centre)}"
    elif rgb is not None:
        centre = srgb8_to_lab(np.array(rgb, dtype=np.float64))
        derived_tolerance = MIN_TOLERANCE_DE
        note = f"target given by hand as sRGB {tuple(int(v) for v in rgb)}"
    else:
        low, high = (np.array(bound, dtype=np.float64) for bound in rgb_range)
        if np.any(high < low):
            raise ProfileError("the upper sRGB bound must not be below the lower bound")
        low_lab = srgb8_to_lab(low)
        high_lab = srgb8_to_lab(high)
        centre = (low_lab + high_lab) / 2.0
        derived_tolerance = max(MIN_TOLERANCE_DE, float(delta_e_2000(low_lab, high_lab)) / 2.0)
        note = (
            f"target given by hand as sRGB range {tuple(int(v) for v in low)}"
            f" to {tuple(int(v) for v in high)}"
        )

    if tolerance is not None:
        derived_tolerance = tolerance

    return SkinProfile(
        version=PROFILE_VERSION,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        render_space=render_space,
        lab=[round(float(v), 4) for v in centre],
        lab_std=[0.0, 0.0, 0.0],
        hue=round(hue_of(centre), 3),
        hue_std=0.0,
        chroma=round(chroma_of(centre), 3),
        chroma_std=0.0,
        lightness=round(float(centre[0]), 3),
        lightness_std=0.0,
        tolerance_de=round(float(derived_tolerance), 3),
        hue_tolerance_deg=round(max(1.0, float(derived_tolerance)), 3),
        samples_used=0,
        samples_seen=0,
        sources=[],
        origin="manual",
        note=note,
    )


def merge_profiles(measured: SkinProfile, manual: SkinProfile, weight: float = 0.5) -> SkinProfile:
    """Blend a measured profile with a hand-supplied one.

    ``weight`` is how much the hand-supplied target counts, from 0 (ignore
    it) to 1 (replace the measured one).  Lightness, chroma and hue are
    blended in their own terms rather than by averaging a* and b*, so the
    result keeps a real hue rather than drifting towards grey when the two
    disagree.
    """
    if not 0.0 <= weight <= 1.0:
        raise ProfileError(f"merge weight must be between 0 and 1, got {weight}")
    if measured.render_space != manual.render_space:
        raise ProfileError(
            f"cannot merge a '{measured.render_space}' profile with a "
            f"'{manual.render_space}' one; they describe different renderings"
        )

    lightness = measured.lightness * (1 - weight) + manual.lightness * weight
    chroma = measured.chroma * (1 - weight) + manual.chroma * weight
    hue_mean, _ = circular_mean_degrees(
        np.array([measured.hue, manual.hue]), np.array([1.0 - weight, weight])
    )
    radians = math.radians(hue_mean)
    blended = [lightness, chroma * math.cos(radians), chroma * math.sin(radians)]

    return SkinProfile(
        version=PROFILE_VERSION,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        render_space=measured.render_space,
        lab=[round(float(v), 4) for v in blended],
        lab_std=measured.lab_std,
        hue=round(hue_mean, 3),
        hue_std=measured.hue_std,
        chroma=round(chroma, 3),
        chroma_std=measured.chroma_std,
        lightness=round(lightness, 3),
        lightness_std=measured.lightness_std,
        tolerance_de=round(
            measured.tolerance_de * (1 - weight) + manual.tolerance_de * weight, 3
        ),
        hue_tolerance_deg=measured.hue_tolerance_deg,
        samples_used=measured.samples_used,
        samples_seen=measured.samples_seen,
        sources=measured.sources,
        origin="merged",
        note=f"{measured.samples_used} reference frames merged with a hand target at {weight:.2f}",
    )
