"""Temperature and tint, against known illuminants and the real renderer."""

import numpy as np
import pytest

from synth import NIKONISH_XYZ_TO_CAM, mosaic, write_dng
from vipcolor import raw as rawmod
from vipcolor import whitebalance as wb
from vipcolor.color import linear_srgb_to_lab

# CIE chromaticities of the standard daylight illuminants.
D65_XY = (0.31272, 0.32903)
D50_XY = (0.34567, 0.35850)


def test_daylight_illuminants_land_on_their_known_temperatures():
    """D65 and D50 are named for their correlated colour temperatures."""
    temperature, tint = wb.xy_to_temp_tint(*D65_XY)
    assert temperature == pytest.approx(6500, abs=60)
    # Daylight sits slightly off the Planckian locus, so tint is not zero --
    # which is why Camera Raw shows daylight frames at around +10, not 0.
    assert 0 < tint < 25

    temperature, _ = wb.xy_to_temp_tint(*D50_XY)
    assert temperature == pytest.approx(5000, abs=60)


def test_a_blackbody_is_its_own_temperature_at_zero_tint():
    for kelvin in (2500, 3200, 5000, 6500, 9000):
        temperature, tint = wb.xy_to_temp_tint(*wb.planckian_xy(kelvin))
        assert temperature == pytest.approx(kelvin, rel=1e-3)
        assert tint == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize(
    "temperature,tint",
    [(2800, 0.0), (4200, -40.0), (5500, 12.0), (6500, 0.0), (9000, 55.0), (14000, -90.0)],
)
def test_temp_tint_round_trips_through_chromaticity(temperature, tint):
    recovered = wb.xy_to_temp_tint(*wb.temp_tint_to_xy(temperature, tint))
    assert recovered[0] == pytest.approx(temperature, rel=1e-4)
    assert recovered[1] == pytest.approx(tint, abs=1e-3)


def test_uv_round_trip():
    for x, y in (D65_XY, D50_XY, (0.4, 0.4), (0.25, 0.3)):
        assert wb.uv_to_xy(*wb.xy_to_uv(x, y)) == pytest.approx((x, y), abs=1e-12)


def test_libraw_derives_daylight_balance_from_the_colour_matrix(tmp_path):
    """The premise the whole temperature model rests on.

    A temperature is turned into multipliers by putting the illuminant
    through the camera's XYZ-to-camera matrix. If that were not the same
    characterisation LibRaw itself uses, every solved temperature would be
    wrong in a way nothing else would catch. So: the camera's response to
    D65 through the matrix written into the file must reproduce LibRaw's own
    daylight preset.

    On a native NEF this also holds for the matrix rawpy reports, checked
    against a real D500 file: agreement to 1e-6.
    """
    cfa, neutral = mosaic(np.full((64, 64, 3), 0.4))
    info = rawmod.read_info(write_dng(tmp_path / "m.dng", cfa, as_shot_neutral=neutral))

    d65_xyz = np.array([0.95047, 1.0, 1.08883])
    response = NIKONISH_XYZ_TO_CAM @ d65_xyz
    implied = response[1] / response

    np.testing.assert_allclose(implied, np.array(info.daylight_wb[:3]), atol=1e-4)


def test_dng_input_reports_an_adapted_matrix(tmp_path):
    """A limitation worth pinning down rather than discovering later.

    For DNG input LibRaw adapts the colour matrix during processing, so
    rgb_xyz_matrix is not the ColorMatrix the file carries -- on this
    synthetic camera that moves its idea of D65 by a few hundred kelvin.
    Native NEF files are unaffected: LibRaw uses its built-in Adobe
    coefficients directly, and there the reported matrix and the daylight
    preset agree exactly. Temperatures for DNG input are therefore on a
    slightly different footing than for NEF, which is what this records.
    """
    cfa, neutral = mosaic(np.full((64, 64, 3), 0.4))
    info = rawmod.read_info(write_dng(tmp_path / "m.dng", cfa, as_shot_neutral=neutral))

    assert not np.allclose(info.rgb_xyz_matrix[:3], NIKONISH_XYZ_TO_CAM, atol=1e-3)
    # Self-consistency survives it, which is what keeps the solver correct:
    # the same matrix is used to go both ways.
    temperature, tint = wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, info.daylight_wb)
    back = wb.temp_tint_to_multipliers(info.rgb_xyz_matrix, temperature, tint)
    np.testing.assert_allclose(back, np.array(info.daylight_wb[:3]), atol=1e-6)


def test_multipliers_round_trip_through_temp_tint(tmp_path):
    cfa, neutral = mosaic(np.full((64, 64, 3), 0.4))
    info = rawmod.read_info(write_dng(tmp_path / "m.dng", cfa, as_shot_neutral=neutral))

    for multipliers in (info.daylight_wb, info.camera_wb):
        temperature, tint = wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, multipliers)
        back = wb.temp_tint_to_multipliers(info.rgb_xyz_matrix, temperature, tint)
        np.testing.assert_allclose(back, np.array(multipliers[:3]), atol=1e-6)


def _grey_card_lab(path, info, temperature, tint):
    multipliers = wb.temp_tint_to_multipliers(info.rgb_xyz_matrix, temperature, tint)
    rendered = rawmod.render_raw(
        path, (multipliers[0], multipliers[1], multipliers[2], multipliers[1]), half_size=True
    )
    return linear_srgb_to_lab(np.median(rendered[16:48, 16:48].reshape(-1, 3), axis=0))


@pytest.fixture(scope="module")
def grey_card(tmp_path_factory):
    path = tmp_path_factory.mktemp("wb") / "grey.dng"
    cfa, neutral = mosaic(np.full((128, 128, 3), 0.35))
    write_dng(path, cfa, as_shot_neutral=neutral)
    return path, rawmod.read_info(path)


def test_a_grey_card_at_its_own_neutral_renders_neutral(grey_card):
    """End to end: temperature and tint, through multipliers, through LibRaw."""
    path, info = grey_card
    temperature, tint = wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, info.daylight_wb)
    lab = _grey_card_lab(path, info, temperature, tint)
    assert lab[1] == pytest.approx(0.0, abs=0.05)
    assert lab[2] == pytest.approx(0.0, abs=0.05)


def test_sign_conventions_match_camera_raw(grey_card):
    """Positive tint reads magenta and higher temperature reads warmer.

    These are conventions, not derivations: getting either backwards would
    write a sidecar that moves the image the wrong way, and nothing else in
    the pipeline would notice.
    """
    path, info = grey_card
    base_temp, base_tint = wb.multipliers_to_temp_tint(info.rgb_xyz_matrix, info.daylight_wb)

    greener = _grey_card_lab(path, info, base_temp, base_tint - 40)
    magenta = _grey_card_lab(path, info, base_temp, base_tint + 40)
    assert greener[1] < -1.0, "negative tint must render green"
    assert magenta[1] > 1.0, "positive tint must render magenta"

    cool = _grey_card_lab(path, info, base_temp - 1500, base_tint)
    warm = _grey_card_lab(path, info, base_temp + 2000, base_tint)
    assert cool[2] < -1.0, "a lower temperature must render cooler"
    assert warm[2] > 1.0, "a higher temperature must render warmer"


def test_implausible_answers_are_reported():
    assert wb.is_plausible(5500, 10) == []
    assert any("temperature" in complaint for complaint in wb.is_plausible(14000, 0))
    assert any("tint" in complaint for complaint in wb.is_plausible(5500, 200))


def test_illuminants_outside_the_camera_gamut_are_refused(grey_card):
    """Not every temperature and tint is a light this sensor could see."""
    _path, info = grey_card
    with pytest.raises(ValueError):
        wb.temp_tint_to_multipliers(info.rgb_xyz_matrix, 2000, 150)
