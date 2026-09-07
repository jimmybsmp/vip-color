"""Solving for the white balance that puts skin on target.

The search would be unaffordable done naively -- a 5600 px NEF takes about
seven seconds to render, and a solve needs dozens of trials.  It is not done
naively.  White balance is applied to camera RGB *before* the camera-to-sRGB
matrix, so a render at one white balance reaches any other by a single 3x3
transform::

    I1 = (M . diag(wb1 / wb0) . M^-1) . I0

That is exact rather than an approximation, and it is checked against real
LibRaw renders in the tests: mean residual about 3e-5, and under 0.04
delta-E on a skin patch.  So the file is rendered once, the surviving patch
pixels are kept, and every trial white balance is a matrix multiply on a few
thousand pixels.

**Lightness is matched before colour is compared.**  A reference JPEG has
been through Camera Raw's exposure and tone curve; a neutral RAW render has
not, and on real files that is about two stops.  Comparing them directly
would have the solver trying to fix a rendering difference by moving the
white balance.  So the sample is scaled to the target's lightness first --
that scale factor *is* the exposure trim -- and the white balance is solved
against what is left, which is hue and chroma.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import raw as rawmod
from . import whitebalance as wb
from .color import (
    XYZ_TO_LINEAR_SRGB,
    delta_e_2000,
    linear_srgb_to_lab,
    linear_srgb_to_xyz,
)
from .faces import NoFaceFound, SkinSample, detect_faces, sample_skin, select_face
from .images import load_linear
from .profile import SkinProfile, chroma_of, hue_difference, hue_of

#: Search bounds.  Wider than what counts as plausible, so that a solve
#: wanting an implausible answer can say so rather than being clamped into
#: looking reasonable.
SEARCH_TEMP_K = (2000.0, 15000.0)
SEARCH_TINT = (-150.0, 150.0)

#: Above this, the 3x3 shortcut is no longer standing in for a real render
#: closely enough to be quiet about.
MODEL_ERROR_WARN_DE = 0.5


class SolveError(Exception):
    """Raised when a white balance cannot be solved."""


@dataclass
class Solution:
    """What the solver decided, and how much to trust it."""

    path: Path
    camera: str
    temperature: float
    tint: float
    multipliers: tuple[float, float, float]
    as_shot: tuple[float, float]
    neutral: tuple[float, float]
    exposure_stops: float
    predicted_lab: np.ndarray
    target_lab: np.ndarray
    delta_e: float
    hue_error_deg: float
    chroma_error: float
    within_tolerance: bool
    model_error: float
    sample: SkinSample
    cross_space: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def temp_shift(self) -> float:
        return self.temperature - self.as_shot[0]

    @property
    def tint_shift(self) -> float:
        return self.tint - self.as_shot[1]


class SkinUnderWhiteBalance:
    """The measured skin of one frame, re-measurable at any white balance."""

    def __init__(self, image, sample: SkinSample, info: rawmod.RawInfo, base_wb, path=None):
        kept = [patch for patch in sample.patches if patch.pixels is not None]
        if not kept:
            raise SolveError("skin was measured without keeping pixels; cannot re-evaluate")
        self.sample = sample
        self.info = info
        self.path = Path(path) if path is not None else sample.path
        self.base_wb = np.asarray(base_wb, dtype=np.float64)[:3]
        self.cam_xyz = info.rgb_xyz_matrix[:3]
        # LibRaw's camera-RGB to linear-sRGB matrix, rebuilt from the same
        # camera characterisation it used.
        self.cam_to_srgb = XYZ_TO_LINEAR_SRGB @ np.linalg.inv(self.cam_xyz)
        self.srgb_to_cam = np.linalg.inv(self.cam_to_srgb)
        self._pixels = [patch.pixels for patch in kept]
        self._weights = np.array([len(patch.pixels) for patch in kept], dtype=np.float64)

    def lab_at(self, temperature: float, tint: float) -> np.ndarray:
        """Skin Lab as it would be under this white balance."""
        multipliers = wb.temp_tint_to_multipliers(self.cam_xyz, temperature, tint)
        return self._lab_for_multipliers(multipliers)

    def _lab_for_multipliers(self, multipliers: np.ndarray) -> np.ndarray:
        ratio = np.asarray(multipliers, dtype=np.float64)[:3] / self.base_wb
        transform = self.cam_to_srgb @ np.diag(ratio) @ self.srgb_to_cam
        labs = []
        for pixels in self._pixels:
            moved = pixels @ transform.T
            labs.append(linear_srgb_to_lab(np.median(moved, axis=0)))
        return np.average(np.array(labs), axis=0, weights=self._weights)

    def model_error(self, temperature: float, tint: float, half_size: bool = True) -> float:
        """How far the fast path is from an actual render, in delta-E.

        The 3x3 shortcut is exact only when the matrix rawpy reports is the
        one LibRaw renders with.  On a native NEF it is, and the error
        measures a few hundredths of a delta-E.  On DNG input LibRaw adapts
        the matrix during processing and reports the adapted one, and the
        shortcut drifts by a few delta-E.  Rather than trust either case,
        this renders once more and measures.
        """
        from .faces import remeasure

        multipliers = wb.temp_tint_to_multipliers(self.cam_xyz, temperature, tint)
        rendered = rawmod.render_raw(
            self.path,
            (multipliers[0], multipliers[1], multipliers[2], multipliers[1]),
            half_size=half_size,
        )
        kept = [patch for patch in self.sample.patches if patch.pixels is not None]
        labs, weights = [], []
        for patch in kept:
            measured = remeasure(rendered, patch)
            if measured is None:
                continue
            labs.append(measured.lab)
            weights.append(measured.pixels_used)
        if not labs:
            return float("nan")
        actual = np.average(np.array(labs), axis=0, weights=np.array(weights, dtype=np.float64))

        predicted = self._lab_for_multipliers(multipliers)
        # LibRaw renormalises multipliers per render, which is a scalar on
        # lightness; the solve only ever uses hue and chroma, so lightness is
        # matched away before comparing.
        matched, _stops = self.matched_to_lightness(predicted, float(actual[0]))
        return float(delta_e_2000(matched, actual))

    def matched_to_lightness(self, lab_sample: np.ndarray, target_L: float) -> tuple[np.ndarray, float]:
        """Rescale a sample to a target lightness; return it and the stops used.

        Scaling happens in linear light, which is what an exposure slider
        does, so the returned stops are directly usable as an exposure trim.
        """
        from .color import lab_to_linear_srgb

        linear = lab_to_linear_srgb(lab_sample)
        current_Y = float(linear_srgb_to_xyz(linear)[1])
        target_linear = lab_to_linear_srgb(np.array([target_L, 0.0, 0.0]))
        target_Y = float(linear_srgb_to_xyz(target_linear)[1])
        if current_Y <= 1e-9 or target_Y <= 0:
            return lab_sample, 0.0
        scale = target_Y / current_Y
        return linear_srgb_to_lab(linear * scale), float(np.log2(scale))


def _objective(
    skin: SkinUnderWhiteBalance,
    profile: SkinProfile,
    temperature: float,
    tint: float,
    hue_weight: float,
) -> float:
    """How wrong this white balance is, in delta-E, after matching lightness.

    Not every temperature/tint pair is a realisable illuminant for a given
    sensor -- towards the corners of the search box the camera's response to
    the implied light goes negative in a channel. Those are reported as
    infinitely bad, which keeps the search inside the feasible region
    without needing a separate feasibility test.
    """
    try:
        sampled = skin.lab_at(temperature, tint)
    except ValueError:
        return float("inf")
    matched, _stops = skin.matched_to_lightness(sampled, profile.lightness)
    difference = float(delta_e_2000(matched, profile.lab_array))
    if hue_weight == 1.0:
        return difference
    # Re-weighting hue against chroma: the hue term is the part that survives
    # a rendering difference, so it can be made to dominate when the target
    # and the sample were measured in different spaces.
    hue_term = hue_difference(hue_of(matched), profile.hue)
    chroma_term = abs(chroma_of(matched) - profile.chroma)
    return float(np.hypot(hue_term * hue_weight, chroma_term))


def _nelder_mead(
    function,
    start: np.ndarray,
    step: np.ndarray,
    bounds: list[tuple[float, float]],
    iterations: int = 200,
    tolerance: float = 1e-7,
) -> tuple[np.ndarray, float]:
    """Minimise a 2-D function without derivatives, following a valley.

    Coordinate descent is the obvious choice here and the wrong one: the
    objective has a narrow valley running diagonally, because temperature
    and tint trade off against each other -- a warmer setting with more
    green lands on nearly the same skin colour as a cooler one with less.
    From the floor of such a valley, moving along either axis alone goes
    uphill, so coordinate descent stops short.  A simplex walks the diagonal.
    """

    def clamp(point: np.ndarray) -> np.ndarray:
        return np.array(
            [np.clip(value, low, high) for value, (low, high) in zip(point, bounds)]
        )

    simplex = [clamp(np.asarray(start, dtype=np.float64))]
    for axis in range(len(start)):
        vertex = np.asarray(start, dtype=np.float64).copy()
        vertex[axis] += step[axis]
        simplex.append(clamp(vertex))
    scores = [function(vertex) for vertex in simplex]

    for _ in range(iterations):
        order = np.argsort(scores)
        simplex = [simplex[i] for i in order]
        scores = [scores[i] for i in order]
        if abs(scores[-1] - scores[0]) < tolerance and np.max(
            np.abs(np.array(simplex[1:]) - simplex[0])
        ) < tolerance:
            break

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = clamp(centroid + (centroid - simplex[-1]))
        reflected_score = function(reflected)

        if reflected_score < scores[0]:
            expanded = clamp(centroid + 2.0 * (centroid - simplex[-1]))
            expanded_score = function(expanded)
            if expanded_score < reflected_score:
                simplex[-1], scores[-1] = expanded, expanded_score
            else:
                simplex[-1], scores[-1] = reflected, reflected_score
        elif reflected_score < scores[-2]:
            simplex[-1], scores[-1] = reflected, reflected_score
        else:
            contracted = clamp(centroid + 0.5 * (simplex[-1] - centroid))
            contracted_score = function(contracted)
            if contracted_score < scores[-1]:
                simplex[-1], scores[-1] = contracted, contracted_score
            else:
                for index in range(1, len(simplex)):
                    simplex[index] = clamp(simplex[0] + 0.5 * (simplex[index] - simplex[0]))
                    scores[index] = function(simplex[index])

    best = int(np.argmin(scores))
    return simplex[best], scores[best]


def solve_white_balance(
    skin: SkinUnderWhiteBalance,
    profile: SkinProfile,
    *,
    hue_weight: float = 1.0,
    restarts: int = 3,
) -> tuple[float, float]:
    """Find the temperature and tint that put this skin on target.

    A coarse sweep finds promising basins, then a simplex refines from the
    best few.  The search runs in reciprocal temperature (mireds), where the
    response to a change is roughly uniform across the range and where a
    mired and a tint unit are of comparable size, so the simplex does not
    have to cope with axes of wildly different scale.

    Several restarts because the objective is not always unimodal: warm-and-
    green can rival cool-and-magenta, and starting only from the single best
    grid point occasionally settles into the poorer of the two.
    """
    bounds = [
        (1e6 / SEARCH_TEMP_K[1], 1e6 / SEARCH_TEMP_K[0]),
        (SEARCH_TINT[0], SEARCH_TINT[1]),
    ]

    def score_at(point) -> float:
        mired, tint = float(point[0]), float(point[1])
        if mired <= 0:
            return float("inf")
        return _objective(skin, profile, 1e6 / mired, tint, hue_weight)

    mireds = np.linspace(bounds[0][0], bounds[0][1], 30)
    tints = np.linspace(SEARCH_TINT[0], SEARCH_TINT[1], 21)
    grid = [
        (score_at((mired, tint)), mired, tint)
        for mired in mireds
        for tint in tints
    ]
    grid.sort(key=lambda item: item[0])
    if not np.isfinite(grid[0][0]):
        raise SolveError(
            "no white balance in the search range renders this sample at all; "
            "the frame may be clipped or the face measurement unusable"
        )

    starts: list[tuple[float, float]] = []
    for _score, mired, tint in grid:
        # Keep restarts genuinely separate rather than three points in one basin.
        if all(abs(mired - m) > 30.0 or abs(tint - t) > 40.0 for m, t in starts):
            starts.append((mired, tint))
        if len(starts) >= restarts:
            break

    step = np.array([15.0, 20.0])
    best_point, best_score = None, float("inf")
    for start in starts:
        point, score = _nelder_mead(score_at, np.array(start), step, bounds)
        if score < best_score:
            best_point, best_score = point, score

    if best_point is None:
        raise SolveError("white balance search did not converge")
    return float(1e6 / best_point[0]), float(best_point[1])


def solve_file(
    path: str | Path,
    profile: SkinProfile,
    *,
    neutral: str = "daylight",
    select: str = "largest",
    hue_weight: float = 1.0,
    half_size: bool = True,
    check_model: bool = False,
) -> Solution:
    """Solve one RAW file against a target profile."""
    path = Path(path)
    from .images import is_raw

    if not is_raw(path):
        raise SolveError(f"{path.name} is not a RAW file; white balance can only be solved on RAW")

    info = rawmod.read_info(path)
    base_wb = rawmod.neutral_wb(info, neutral)
    image = load_linear(path, neutral=neutral, half_size=half_size)

    found = detect_faces(image)
    picked, _how = select_face(found, image, select=select)
    sample = sample_skin(image, landmarks=picked.landmarks, select=select, keep_pixels=True)
    sample.faces_found = len(found)
    sample.detector_score = picked.score

    skin = SkinUnderWhiteBalance(image, sample, info, base_wb, path=path)
    temperature, tint = solve_white_balance(skin, profile, hue_weight=hue_weight)

    solved_lab = skin.lab_at(temperature, tint)
    matched, stops = skin.matched_to_lightness(solved_lab, profile.lightness)
    multipliers = wb.temp_tint_to_multipliers(info.rgb_xyz_matrix, temperature, tint)

    warnings = list(sample.warnings)
    warnings.extend(wb.is_plausible(temperature, tint))

    model_error = float("nan")
    if check_model:
        model_error = skin.model_error(temperature, tint, half_size=half_size)
        if np.isfinite(model_error) and model_error > MODEL_ERROR_WARN_DE:
            warnings.append(
                f"the fast white balance model disagrees with an actual render by "
                f"{model_error:.2f} dE at the solved setting. That is the shortcut "
                "breaking down, not the skin measurement; treat this solve as approximate"
            )

    cross_space = profile.render_space != "linear"
    if cross_space:
        warnings.append(
            "the target was measured on rendered files and this is a linear RAW render. "
            "Hue carries across that difference; chroma does not, because Camera Raw's "
            "tone curve changes saturation. Treat the chroma gap as indicative"
        )

    return Solution(
        path=path,
        camera=info.camera,
        temperature=temperature,
        tint=tint,
        multipliers=tuple(float(v) for v in multipliers[:3]),
        as_shot=wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, info.camera_wb),
        neutral=wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, base_wb),
        exposure_stops=stops,
        predicted_lab=matched,
        target_lab=profile.lab_array,
        delta_e=float(delta_e_2000(matched, profile.lab_array)),
        hue_error_deg=hue_difference(hue_of(matched), profile.hue),
        chroma_error=float(chroma_of(matched) - profile.chroma),
        within_tolerance=float(delta_e_2000(matched, profile.lab_array)) <= profile.tolerance_de,
        model_error=model_error,
        sample=sample,
        cross_space=cross_space,
        warnings=warnings,
    )
