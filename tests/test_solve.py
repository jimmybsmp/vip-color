"""Solving white balance, against targets whose answer is known."""

import numpy as np
import pytest
from PIL import Image

from synth import mosaic, write_dng
from vipcolor import raw as rawmod
from vipcolor import whitebalance as wb
from vipcolor.color import delta_e_2000, srgb_decode
from vipcolor.faces import detect_faces, sample_skin, select_face
from vipcolor.images import load_linear
from vipcolor.profile import manual_profile
from vipcolor.solve import (
    MODEL_ERROR_WARN_DE,
    SkinUnderWhiteBalance,
    SolveError,
    solve_file,
    solve_white_balance,
)

FIXTURE = "tests/fixtures/face_astronaut.png"


@pytest.fixture(scope="module")
def portrait_raw(tmp_path_factory):
    """A synthetic RAW of a real face, so a solve has something to measure."""
    path = tmp_path_factory.mktemp("solve") / "portrait.dng"
    linear = srgb_decode(np.asarray(Image.open(FIXTURE).convert("RGB"), dtype=np.float64) / 255.0)
    cfa, neutral = mosaic(linear)
    write_dng(path, cfa, as_shot_neutral=neutral)
    return path


@pytest.fixture(scope="module")
def skin(portrait_raw):
    info = rawmod.read_info(portrait_raw)
    base = rawmod.neutral_wb(info, "daylight")
    image = load_linear(portrait_raw, neutral="daylight", half_size=False)
    found = detect_faces(image)
    picked, _how = select_face(found, image)
    sample = sample_skin(image, landmarks=picked.landmarks, keep_pixels=True)
    return SkinUnderWhiteBalance(image, sample, info, base)


def test_the_fast_path_is_measured_against_a_real_render(skin):
    """The shortcut is checkable, and on DNG it is caught being wrong.

    Trial white balances are a 3x3 matrix multiply rather than a re-render,
    which is exact only when the matrix rawpy reports is the one LibRaw
    renders with. On a native NEF it is: measured against a real D500 file,
    the model error is 0.001 delta-E. On DNG input LibRaw adapts the matrix
    during processing and reports the adapted one, so the shortcut drifts by
    a few delta-E -- which is precisely what this fixture is, and precisely
    what the check must not stay quiet about.
    """
    errors = [skin.model_error(temperature, tint) for temperature, tint in
              ((4200, -20.0), (6500, 10.0), (8800, 35.0))]

    assert all(np.isfinite(error) for error in errors)
    assert max(errors) > MODEL_ERROR_WARN_DE, (
        "a DNG's adapted matrix must be detected, not silently tolerated"
    )


def test_solve_file_reports_the_model_check_when_asked(portrait_raw, skin):
    target = manual_profile(
        lab=tuple(float(v) for v in skin.lab_at(6000, 5.0)), render_space="linear"
    )

    quiet = solve_file(portrait_raw, target)
    assert np.isnan(quiet.model_error), "the extra render must be opt-in"

    checked = solve_file(portrait_raw, target, check_model=True)
    assert np.isfinite(checked.model_error)
    assert any("fast white balance model" in warning for warning in checked.warnings)


@pytest.mark.parametrize(
    "temperature,tint",
    [(5200, 10.0), (3400, -25.0), (7800, 40.0), (6500, -60.0), (2800, 60.0), (12000, 25.0)],
)
def test_solver_recovers_a_white_balance_it_was_given(skin, temperature, tint):
    """The solver must find its way back to a white balance it was given.

    This is a search test, not an accuracy test: truth and solve both run
    through the fast path, so it pins that the minimiser finds the minimum,
    not that the minimum is where a real render would put it. model_error
    is what speaks to the latter.
    """
    truth = skin.lab_at(temperature, tint)
    target = manual_profile(lab=tuple(float(v) for v in truth), render_space="linear")

    solved_temp, solved_tint = solve_white_balance(skin, target)

    assert solved_temp == pytest.approx(temperature, rel=0.01)
    assert solved_tint == pytest.approx(tint, abs=1.0)


def test_solver_follows_a_diagonal_valley(skin):
    """Regression: coordinate descent stalled here, a simplex does not.

    Temperature and tint trade off, so the objective has a narrow valley
    running diagonally -- warm-and-green lands near cool-and-magenta. From
    the floor of it, moving along either axis alone goes uphill, and an
    earlier coordinate-descent search stopped 814 K short on this case.
    """
    truth = skin.lab_at(2800, 60.0)
    target = manual_profile(lab=tuple(float(v) for v in truth), render_space="linear")
    solved_temp, solved_tint = solve_white_balance(skin, target)
    assert abs(solved_temp - 2800) < 50
    assert abs(solved_tint - 60.0) < 2.0


def test_lightness_matching_returns_usable_exposure_stops(skin):
    lab = skin.lab_at(6500, 10.0)
    brighter, stops = skin.matched_to_lightness(lab, float(lab[0]) + 15.0)
    assert brighter[0] == pytest.approx(lab[0] + 15.0, abs=0.05)
    assert stops > 0

    same, no_stops = skin.matched_to_lightness(lab, float(lab[0]))
    assert no_stops == pytest.approx(0.0, abs=1e-6)
    assert same == pytest.approx(lab, abs=1e-6)


def test_solve_file_reports_shifts_and_refuses_non_raw(portrait_raw):
    info = rawmod.read_info(portrait_raw)
    image = load_linear(portrait_raw, neutral="daylight")
    found = detect_faces(image)
    picked, _how = select_face(found, image)
    measured = sample_skin(image, landmarks=picked.landmarks, keep_pixels=True)

    target = manual_profile(
        lab=tuple(float(v) for v in measured.lab), render_space="linear"
    )
    solution = solve_file(portrait_raw, target)

    assert solution.delta_e < 1.0
    assert solution.camera
    assert solution.as_shot[0] > 0
    assert solution.multipliers[1] == pytest.approx(1.0, abs=1e-9)
    assert isinstance(solution.warnings, list)

    with pytest.raises(SolveError, match="not a RAW file"):
        solve_file(FIXTURE, target)


def test_cross_space_targets_are_flagged(portrait_raw):
    """A rendered target against a linear render must say so."""
    rendered_target = manual_profile(lab=(65.0, 23.0, 26.0), render_space="rendered")
    solution = solve_file(portrait_raw, rendered_target)
    assert solution.cross_space
    assert any("rendered files" in warning for warning in solution.warnings)
