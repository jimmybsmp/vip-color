"""Finding a face and measuring its skin.

The measurement is the whole product: every later number is a function of
what this module decides "the skin" is.  So it is deliberately conservative
-- it samples several small patches in places that are flat, matte and
unlikely to be anything but skin, throws away pixels that look like
specular highlight, shadow, hair or spectacle frame, and reports how much
the surviving patches disagree with each other.

If no face is found it says so.  It never falls back to sampling the middle
of the frame and hoping.
"""

from __future__ import annotations

import contextlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .color import delta_e_2000, lab_to_srgb8, linear_srgb_to_lab
from .images import LinearImage

# MediaPipe Face Mesh landmark indices.  Patch centres are blends of several
# landmarks rather than single points, so a small per-frame landmark wobble
# moves the patch by much less than its own radius.
# 151 sits on the hairline, so a forehead patch anchored there lands in a
# fringe on anyone who has one.  Weighting towards 9 (glabella) keeps the
# patch on skin for low and high hairlines alike.
_FOREHEAD = ((151, 0.4), (9, 0.6))
_CHEEK_LEFT = ((50, 0.35), (205, 0.35), (101, 0.2), (118, 0.1))
_CHEEK_RIGHT = ((280, 0.35), (425, 0.35), (330, 0.2), (347, 0.1))
_CHIN = ((200, 0.5), (199, 0.25), (175, 0.25))

#: Patches sampled by default: the three the eye judges skin tone by.
DEFAULT_PATCHES = ("forehead", "cheek_left", "cheek_right")
_PATCH_LANDMARKS = {
    "forehead": _FOREHEAD,
    "cheek_left": _CHEEK_LEFT,
    "cheek_right": _CHEEK_RIGHT,
    "chin": _CHIN,
}

# Face scale is the outer-eye-corner distance; patch radius is a fraction of it.
_EYE_OUTER_LEFT = 33
_EYE_OUTER_RIGHT = 263
_PATCH_RADIUS_FRACTION = 0.11
#: Per-patch size trim.  The forehead patch has eyebrows below it and hair
#: above, so it stays small; cheeks sit in open skin and can afford more.
_PATCH_RADIUS_SCALE = {
    "forehead": 0.75,
    "cheek_left": 1.0,
    "cheek_right": 1.0,
    "chin": 0.8,
}

# Pixel rejection thresholds, all in scene-linear terms except the last.
_CLIP_CEILING = 0.98          # any channel at or above this is blown
_FLOOR = 0.004                # below this, read noise dominates
_SPECULAR_PERCENTILE = 85.0   # drop the brightest tail: sheen, sweat, glare
_SHADOW_PERCENTILE = 10.0     # drop the darkest tail: pores, stubble, creases
_CHROMA_REJECT_DE = 12.0      # drop pixels this far from the patch's own colour
_MIN_USABLE_PIXELS = 40       # below this a patch is not worth trusting

#: How many faces to look for.  Group shots are the reason this is not 1:
#: silently measuring whichever face the detector happened to return first
#: would build a profile from the wrong person.
MAX_FACES = 8

#: Patches whose *chroma* disagrees by more than this are the ones that
#: threaten a white balance solve.  Luminance disagreement between forehead
#: and cheek is ordinary directional light and is reported, not warned about.
CHROMA_SPREAD_WARN_DE = 4.0


class NoFaceFound(Exception):
    """Raised when no face is detected confidently enough to sample."""


@dataclass(frozen=True)
class Patch:
    """One sampled skin region."""

    name: str
    center: tuple[float, float]
    radius: float
    lab: np.ndarray
    linear_rgb: np.ndarray
    pixels_total: int
    pixels_used: int
    clipped_fraction: float

    @property
    def usable_fraction(self) -> float:
        return self.pixels_used / self.pixels_total if self.pixels_total else 0.0

    @property
    def srgb8(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in lab_to_srgb8(self.lab))


@dataclass
class SkinSample:
    """The skin measurement for one image."""

    path: Path
    lab: np.ndarray
    patches: list[Patch]
    spread: float
    chroma_spread: float
    faces_found: int
    luminance_range: float
    colour_note: str
    camera: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def srgb8(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in lab_to_srgb8(self.lab))

    @property
    def pixels_used(self) -> int:
        return sum(p.pixels_used for p in self.patches)


@contextlib.contextmanager
def _quiet():
    """Silence MediaPipe/absl's startup chatter on stderr."""
    previous = os.environ.get("GLOG_minloglevel")
    os.environ["GLOG_minloglevel"] = "3"
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stderr(buffer):
            yield
    finally:
        if previous is None:
            os.environ.pop("GLOG_minloglevel", None)
        else:
            os.environ["GLOG_minloglevel"] = previous


def detect_faces(
    image: LinearImage,
    *,
    min_confidence: float = 0.5,
    max_faces: int = MAX_FACES,
) -> list[np.ndarray]:
    """Every face in the frame, as (N, 2) pixel-coordinate landmark arrays.

    Detection runs on an sRGB-encoded, downscaled copy because that is what
    the model was trained on; the returned coordinates are scaled back to
    the full-resolution linear image, which is where sampling happens.
    Faces come back largest first.
    """
    with _quiet():
        import mediapipe as mp

        detector_input, _scale = image.detector_image()
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
        ) as mesh:
            result = mesh.process(detector_input)

    if not result.multi_face_landmarks:
        raise NoFaceFound(f"no face detected in {image.path.name}")

    height, width = image.linear.shape[:2]
    # Landmarks are normalised to the detector input, so they map onto the
    # full-resolution image by scaling alone.
    faces = [
        np.array([[lm.x * width, lm.y * height] for lm in face.landmark], dtype=np.float64)
        for face in result.multi_face_landmarks
    ]
    return sorted(faces, key=_face_size, reverse=True)


def _face_size(landmarks: np.ndarray) -> float:
    """Bounding-box diagonal, used to rank faces by how much frame they fill."""
    extent = landmarks.max(axis=0) - landmarks.min(axis=0)
    return float(np.hypot(*extent))


def select_face(
    faces: list[np.ndarray],
    image: LinearImage,
    *,
    select: str = "largest",
) -> tuple[np.ndarray, str]:
    """Choose which detected face to measure, and say how it was chosen."""
    if not faces:
        raise NoFaceFound("no faces to select from")
    if select == "largest":
        return faces[0], "largest face"
    if select == "center":
        height, width = image.linear.shape[:2]
        middle = np.array([width / 2.0, height / 2.0])
        ranked = min(faces, key=lambda f: float(np.linalg.norm(f.mean(axis=0) - middle)))
        return ranked, "most central face"
    if select.isdigit():
        index = int(select)
        if index >= len(faces):
            raise NoFaceFound(f"asked for face {index} but only {len(faces)} were detected")
        return faces[index], f"face {index} by size"
    raise ValueError(f"unknown face selection {select!r}; expected largest, center, or an index")


def detect_landmarks(
    image: LinearImage,
    *,
    min_confidence: float = 0.5,
    select: str = "largest",
) -> np.ndarray:
    """The landmarks of the one face this image should be measured on."""
    faces = detect_faces(image, min_confidence=min_confidence)
    chosen, _how = select_face(faces, image, select=select)
    return chosen


def _blend(landmarks: np.ndarray, weighted: tuple) -> np.ndarray:
    total = sum(weight for _, weight in weighted)
    return sum(landmarks[index] * weight for index, weight in weighted) / total


def _disc_pixels(linear: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Every pixel inside a disc, as (N, 3) linear RGB."""
    height, width = linear.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = max(1.0, radius)
    x0, x1 = max(0, int(np.floor(cx - r))), min(width, int(np.ceil(cx + r)) + 1)
    y0, y1 = max(0, int(np.floor(cy - r))), min(height, int(np.ceil(cy + r)) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 3), dtype=np.float64)

    window = linear[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
    return window[inside].reshape(-1, 3)


def _measure(name: str, linear: np.ndarray, center: np.ndarray, radius: float) -> Patch | None:
    """Robustly reduce one disc of pixels to a single Lab value."""
    pixels = _disc_pixels(linear, center, radius)
    total = len(pixels)
    if total == 0:
        return None

    clipped = np.any(pixels >= _CLIP_CEILING, axis=1)
    clipped_fraction = float(clipped.mean())
    luminance = pixels @ np.array([0.2126729, 0.7151522, 0.0721750])
    keep = ~clipped & (luminance > _FLOOR)
    if keep.sum() < _MIN_USABLE_PIXELS:
        return None

    # Trim the specular and shadow tails by luminance.
    candidate = pixels[keep]
    candidate_luma = luminance[keep]
    low, high = np.percentile(candidate_luma, [_SHADOW_PERCENTILE, _SPECULAR_PERCENTILE])
    band = (candidate_luma >= low) & (candidate_luma <= high)
    if band.sum() >= _MIN_USABLE_PIXELS:
        candidate = candidate[band]

    # Then reject by colour: whatever is left should agree on hue, so hair,
    # spectacle frames and lip edge fall out here even at matching brightness.
    provisional = np.median(candidate, axis=0)
    provisional_lab = linear_srgb_to_lab(provisional)
    distances = delta_e_2000(linear_srgb_to_lab(candidate), provisional_lab)
    agreeing = distances <= _CHROMA_REJECT_DE
    if agreeing.sum() >= _MIN_USABLE_PIXELS:
        candidate = candidate[agreeing]

    median_rgb = np.median(candidate, axis=0)
    return Patch(
        name=name,
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        lab=linear_srgb_to_lab(median_rgb),
        linear_rgb=median_rgb,
        pixels_total=total,
        pixels_used=int(len(candidate)),
        clipped_fraction=clipped_fraction,
    )


def sample_skin(
    image: LinearImage,
    *,
    patches: tuple[str, ...] = DEFAULT_PATCHES,
    min_confidence: float = 0.5,
    landmarks: np.ndarray | None = None,
    select: str = "largest",
) -> SkinSample:
    """Detect the face in ``image`` and measure its skin tone in Lab."""
    warnings: list[str] = []
    faces_found = 1
    if landmarks is None:
        detected = detect_faces(image, min_confidence=min_confidence)
        faces_found = len(detected)
        landmarks, how = select_face(detected, image, select=select)
        if faces_found > 1:
            warnings.append(
                f"{faces_found} faces detected; measured the {how}. "
                "Check the overlay, and use --face to pick a different one."
            )

    scale = float(np.linalg.norm(landmarks[_EYE_OUTER_LEFT] - landmarks[_EYE_OUTER_RIGHT]))
    if not np.isfinite(scale) or scale <= 0:
        raise NoFaceFound(f"degenerate face geometry in {image.path.name}")
    radius = scale * _PATCH_RADIUS_FRACTION

    measured: list[Patch] = []
    for name in patches:
        if name not in _PATCH_LANDMARKS:
            raise ValueError(f"unknown patch {name!r}; expected {sorted(_PATCH_LANDMARKS)}")
        patch = _measure(
            name,
            image.linear,
            _blend(landmarks, _PATCH_LANDMARKS[name]),
            radius * _PATCH_RADIUS_SCALE.get(name, 1.0),
        )
        if patch is None:
            warnings.append(f"{name}: too few usable pixels, patch dropped")
            continue
        if patch.clipped_fraction > 0.25:
            warnings.append(f"{name}: {patch.clipped_fraction:.0%} of pixels clipped")
        measured.append(patch)

    if not measured:
        raise NoFaceFound(f"face found in {image.path.name} but no patch was usable")

    labs = np.array([p.lab for p in measured])
    weights = np.array([p.pixels_used for p in measured], dtype=np.float64)
    combined = np.average(labs, axis=0, weights=weights)

    spread = 0.0
    chroma_spread = 0.0
    luminance_range = 0.0
    if len(measured) > 1:
        # Chroma spread holds L constant, isolating the disagreement that
        # actually matters to a white balance solve from the brightness
        # gradient that any directional light produces.
        flattened = labs.copy()
        flattened[:, 0] = labs[:, 0].mean()
        pairs = [(i, j) for i in range(len(labs)) for j in range(i + 1, len(labs))]
        spread = max(float(delta_e_2000(labs[i], labs[j])) for i, j in pairs)
        chroma_spread = max(float(delta_e_2000(flattened[i], flattened[j])) for i, j in pairs)
        luminance_range = float(labs[:, 0].max() - labs[:, 0].min())
        if chroma_spread > CHROMA_SPREAD_WARN_DE:
            warnings.append(
                f"patch colours disagree by {chroma_spread:.1f} dE2000 with brightness held "
                "constant -- mixed light sources, or a patch is off-skin"
            )

    return SkinSample(
        path=image.path,
        lab=combined,
        patches=measured,
        spread=spread,
        chroma_spread=chroma_spread,
        faces_found=faces_found,
        luminance_range=luminance_range,
        colour_note=image.colour_note,
        camera=image.camera,
        warnings=warnings,
    )


def sample_rect(image: LinearImage, rect: tuple[int, int, int, int], name: str = "manual") -> SkinSample:
    """Measure an explicit x,y,w,h rectangle, bypassing face detection.

    This exists for frames with no face in them -- proving the decode and
    measurement path on a test file, or sampling a grey card.
    """
    x, y, width, height = rect
    center = np.array([x + width / 2.0, y + height / 2.0])
    radius = min(width, height) / 2.0
    patch = _measure(name, image.linear, center, radius)
    if patch is None:
        raise NoFaceFound(f"no usable pixels in rectangle {rect} of {image.path.name}")
    return SkinSample(
        path=image.path,
        lab=patch.lab,
        patches=[patch],
        spread=0.0,
        chroma_spread=0.0,
        faces_found=0,
        luminance_range=0.0,
        colour_note=image.colour_note,
        camera=image.camera,
        warnings=["measured from an explicit rectangle, not a detected face"],
    )
