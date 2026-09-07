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
_MIN_USABLE_PIXELS = 12       # below this a patch is not worth trusting
_LOW_PIXEL_WARN = 60          # above this, the patch median is comfortably stable

#: Eye-corner separation below which a face is small enough that JPEG chroma
#: subsampling starts to matter -- the colour resolution is half the luma
#: resolution, so a small face's skin tone is measured from very few real
#: chroma samples.
_SMALL_FACE_EYE_PX = 45.0

#: How many faces to look for.  Group shots are the reason this is not 1:
#: silently measuring whichever face the detector happened to return first
#: would build a profile from the wrong person.
MAX_FACES = 8

#: How far past the detector's box to crop before landmarking, and the size
#: band the crop is scaled into.  Face Mesh needs context and enough pixels.
_CROP_EXPANSION = 2.0
_CROP_MIN_EDGE = 256
_CROP_MAX_EDGE = 512

#: Head turn, as the nose tip's offset along the eye-corner axis: 0 is
#: square to camera, 1 is full profile.  A turned head puts one cheek in
#: different light from the other, which is measurable long before it is
#: obvious, so it is scored rather than eyeballed.
POSE_WARN_YAW = 0.25
POSE_REJECT_YAW = 0.60

#: Eye separation below which landmark placement error is a large fraction
#: of a patch radius.
MIN_RELIABLE_EYE_PX = 20.0

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
    #: The surviving pixels, linear RGB, kept only when the caller asks.
    #: The white balance solver re-measures this patch under trial white
    #: balances, and it must use the same pixels the rejection chose, not a
    #: fresh rejection pass whose membership would shift as colour moves.
    pixels: np.ndarray | None = None

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
    detector_score: float
    pose_yaw: float
    face_scale: float

    @property
    def quality(self) -> str:
        """A one-word verdict, for filtering reference images and batch runs."""
        if (
            self.pose_yaw > POSE_REJECT_YAW
            or self.chroma_spread > 2 * CHROMA_SPREAD_WARN_DE
            or self.face_scale < MIN_RELIABLE_EYE_PX
        ):
            return "poor"
        if (
            self.pose_yaw > POSE_WARN_YAW
            or self.chroma_spread > CHROMA_SPREAD_WARN_DE
            or self.face_scale < _SMALL_FACE_EYE_PX
            or min((p.pixels_used for p in self.patches), default=0) < _LOW_PIXEL_WARN
        ):
            return "marginal"
        return "good"
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


@dataclass(frozen=True)
class DetectedFace:
    """One face located in the frame, with landmarks in full-image pixels."""

    landmarks: np.ndarray
    score: float
    box: tuple[int, int, int, int]

    @property
    def size(self) -> float:
        """Bounding-box diagonal: how much of the frame this face fills."""
        extent = self.landmarks.max(axis=0) - self.landmarks.min(axis=0)
        return float(np.hypot(*extent))


def _detection_boxes(
    detector_input: np.ndarray, min_confidence: float, max_faces: int
) -> list[tuple[tuple[float, float, float, float], float]]:
    """Locate faces with the full-range detector, largest first.

    Face Mesh's own detector is BlazeFace *short range*, which assumes the
    face fills much of the frame.  Event photography does the opposite: a
    head a couple of hundred pixels wide in a 3000 px frame, which that
    model simply does not see.  So detection is done separately with the
    full-range model and Face Mesh is handed a crop.
    """
    import mediapipe as mp

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=min_confidence
    ) as detector:
        result = detector.process(detector_input)

    if not result.detections:
        return []

    height, width = detector_input.shape[:2]
    boxes = []
    for detection in result.detections:
        relative = detection.location_data.relative_bounding_box
        box = (
            relative.xmin * width,
            relative.ymin * height,
            relative.width * width,
            relative.height * height,
        )
        if box[2] <= 0 or box[3] <= 0:
            continue
        score = float(detection.score[0]) if detection.score else 0.0
        boxes.append((box, score))

    boxes.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
    return boxes[:max_faces]


def _crop_around(
    encoded: np.ndarray, box: tuple[float, float, float, float], scale: float
) -> tuple[np.ndarray, int, int]:
    """Cut a padded region around a detection, in full-resolution pixels."""
    x, y, width, height = (value / scale for value in box)
    centre_x, centre_y = x + width / 2.0, y + height / 2.0
    # Face Mesh wants context around the face, not a tight crop.
    half = max(width, height) * _CROP_EXPANSION / 2.0

    image_height, image_width = encoded.shape[:2]
    x0 = max(0, int(centre_x - half))
    y0 = max(0, int(centre_y - half))
    x1 = min(image_width, int(centre_x + half))
    y1 = min(image_height, int(centre_y + half))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.empty((0, 0, 3), dtype=np.uint8), 0, 0
    return encoded[y0:y1, x0:x1], x0, y0


def detect_faces(
    image: LinearImage,
    *,
    min_confidence: float = 0.5,
    max_faces: int = MAX_FACES,
) -> list[DetectedFace]:
    """Every face in the frame, largest first, landmarked in image pixels.

    Two stages: the full-range detector finds faces anywhere in the frame,
    then Face Mesh landmarks each one from a crop.  Landmark coordinates are
    mapped back to the full-resolution image, which is where skin is
    measured -- the crop is only ever an input to the landmarker.
    """
    import cv2

    with _quiet():
        import mediapipe as mp

        detector_input, scale = image.detector_image()
        boxes = _detection_boxes(detector_input, min_confidence, max_faces)
        if not boxes:
            raise NoFaceFound(f"no face detected in {image.path.name}")

        encoded = image.encoded8()
        found: list[DetectedFace] = []
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_confidence,
        ) as mesh:
            for box, score in boxes:
                crop, x0, y0 = _crop_around(encoded, box, scale)
                if crop.size == 0:
                    continue

                # Upscale a small crop so the landmarker sees enough detail;
                # keep the aspect ratio so landmarks are not skewed.
                crop_height, crop_width = crop.shape[:2]
                longest = max(crop_height, crop_width)
                resized, factor = crop, 1.0
                if longest < _CROP_MIN_EDGE:
                    factor = _CROP_MIN_EDGE / longest
                elif longest > _CROP_MAX_EDGE:
                    factor = _CROP_MAX_EDGE / longest
                if factor != 1.0:
                    resized = cv2.resize(
                        crop,
                        (max(2, round(crop_width * factor)), max(2, round(crop_height * factor))),
                        interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC,
                    )

                result = mesh.process(np.ascontiguousarray(resized))
                if not result.multi_face_landmarks:
                    continue

                landmarks = np.array(
                    [
                        [x0 + lm.x * crop_width, y0 + lm.y * crop_height]
                        for lm in result.multi_face_landmarks[0].landmark
                    ],
                    dtype=np.float64,
                )
                found.append(
                    DetectedFace(
                        landmarks=landmarks,
                        score=score,
                        box=(x0, y0, crop_width, crop_height),
                    )
                )

    if not found:
        raise NoFaceFound(
            f"{len(boxes)} face(s) detected in {image.path.name} but none could be landmarked"
        )
    return sorted(found, key=lambda f: f.size, reverse=True)


def select_face(
    faces: list[DetectedFace],
    image: LinearImage,
    *,
    select: str = "largest",
) -> tuple[DetectedFace, str]:
    """Choose which detected face to measure, and say how it was chosen."""
    if not faces:
        raise NoFaceFound("no faces to select from")
    if select == "largest":
        return faces[0], "largest face"
    if select == "center":
        height, width = image.linear.shape[:2]
        middle = np.array([width / 2.0, height / 2.0])
        chosen = min(
            faces, key=lambda f: float(np.linalg.norm(f.landmarks.mean(axis=0) - middle))
        )
        return chosen, "most central face"
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
    chosen, _how = select_face(detect_faces(image, min_confidence=min_confidence), image, select=select)
    return chosen.landmarks


def head_yaw(landmarks: np.ndarray) -> float:
    """How far the head is turned, from 0 (square on) to 1 (full profile).

    Measured as where the nose tip falls along the line joining the outer
    eye corners.  It needs no camera model and no 3-D fit, and it degrades
    gracefully: a slightly turned head scores slightly above zero.
    """
    left, right, nose = landmarks[_EYE_OUTER_LEFT], landmarks[_EYE_OUTER_RIGHT], landmarks[1]
    axis = right - left
    denominator = float(axis @ axis)
    if denominator <= 0:
        return 1.0
    position = float((nose - left) @ axis) / denominator
    return float(min(1.0, abs(position - 0.5) * 2.0))


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


def _measure(
    name: str,
    linear: np.ndarray,
    center: np.ndarray,
    radius: float,
    keep_pixels: bool = False,
) -> Patch | None:
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
        pixels=candidate.copy() if keep_pixels else None,
    )


def sample_skin(
    image: LinearImage,
    *,
    patches: tuple[str, ...] = DEFAULT_PATCHES,
    min_confidence: float = 0.5,
    landmarks: np.ndarray | None = None,
    select: str = "largest",
    keep_pixels: bool = False,
) -> SkinSample:
    """Detect the face in ``image`` and measure its skin tone in Lab."""
    warnings: list[str] = []
    faces_found = 1
    score = float("nan")
    if landmarks is None:
        detected = detect_faces(image, min_confidence=min_confidence)
        faces_found = len(detected)
        chosen, how = select_face(detected, image, select=select)
        landmarks = chosen.landmarks
        score = chosen.score
        if faces_found > 1:
            warnings.append(
                f"{faces_found} faces detected; measured the {how}. "
                "Check the overlay, and use --face to pick a different one."
            )

    scale = float(np.linalg.norm(landmarks[_EYE_OUTER_LEFT] - landmarks[_EYE_OUTER_RIGHT]))
    if not np.isfinite(scale) or scale <= 0:
        raise NoFaceFound(f"degenerate face geometry in {image.path.name}")
    radius = scale * _PATCH_RADIUS_FRACTION
    yaw = head_yaw(landmarks)
    if yaw > POSE_REJECT_YAW:
        warnings.append(
            f"head is turned well off camera (yaw {yaw:.2f}); the cheeks are in different "
            "light and this sample should not be trusted"
        )
    elif yaw > POSE_WARN_YAW:
        warnings.append(f"head is turned (yaw {yaw:.2f}); cheek patches may disagree")
    if scale < _SMALL_FACE_EYE_PX:
        warnings.append(
            f"small face: {scale:.0f} px between eye corners. On a JPEG the chroma is "
            "subsampled, so this measurement rests on few colour samples"
        )

    measured: list[Patch] = []
    for name in patches:
        if name not in _PATCH_LANDMARKS:
            raise ValueError(f"unknown patch {name!r}; expected {sorted(_PATCH_LANDMARKS)}")
        patch = _measure(
            name,
            image.linear,
            _blend(landmarks, _PATCH_LANDMARKS[name]),
            radius * _PATCH_RADIUS_SCALE.get(name, 1.0),
            keep_pixels=keep_pixels,
        )
        if patch is None:
            warnings.append(f"{name}: too few usable pixels, patch dropped")
            continue
        if patch.pixels_used < _LOW_PIXEL_WARN:
            warnings.append(f"{name}: only {patch.pixels_used} usable pixels")
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
        detector_score=score,
        pose_yaw=yaw,
        face_scale=scale,
        luminance_range=luminance_range,
        colour_note=image.colour_note,
        camera=image.camera,
        warnings=warnings,
    )


def remeasure(linear: np.ndarray, patch: Patch) -> Patch | None:
    """Measure the same disc again on a differently rendered image.

    Used to check the solver's fast path against a real render: the patch
    has to be the same pixels in the same place, not a fresh detection.
    """
    return _measure(patch.name, linear, np.array(patch.center), patch.radius)


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
        detector_score=float("nan"),
        pose_yaw=float("nan"),
        face_scale=float("nan"),
        luminance_range=0.0,
        colour_note=image.colour_note,
        camera=image.camera,
        warnings=["measured from an explicit rectangle, not a detected face"],
    )
