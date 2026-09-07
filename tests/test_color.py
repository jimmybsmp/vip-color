"""Colorimetry, checked against published reference values."""

import numpy as np
import pytest

from vipcolor.color import (
    delta_e_2000,
    lab_to_linear_srgb,
    linear_srgb_to_lab,
    srgb8_to_lab,
    srgb_decode,
    srgb_encode,
)

# Sharma, Wu & Dalal (2005) CIEDE2000 test data.  These are the pairs the
# formula's branch cases live in: hue wrap either side of 0/360, near-neutral
# chroma, and the dark end.
SHARMA_CASES = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, -1.3802, -84.2814), (50.0, 0.0, -82.7485), 1.0000),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0011), 7.2195),
    ((50.0, -0.001, 2.49), (50.0, 0.0009, -2.49), 4.8045),
    ((50.0, 2.5, 0.0), (50.0, 0.0, -2.5), 4.3065),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((50.0, 2.5, 0.0), (56.0, -27.0, -3.0), 31.9030),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.135), (0.9033, -0.0636, -0.5514), 0.9082),
]


@pytest.mark.parametrize("lab1,lab2,expected", SHARMA_CASES)
def test_delta_e_2000_matches_sharma(lab1, lab2, expected):
    assert delta_e_2000(np.array(lab1), np.array(lab2)) == pytest.approx(expected, abs=1e-4)


def test_delta_e_2000_is_symmetric_and_zero_on_identity():
    rng = np.random.default_rng(7)
    a = rng.uniform([0, -80, -80], [100, 80, 80], (200, 3))
    b = rng.uniform([0, -80, -80], [100, 80, 80], (200, 3))
    np.testing.assert_allclose(delta_e_2000(a, b), delta_e_2000(b, a), atol=1e-12)
    np.testing.assert_allclose(delta_e_2000(a, a), 0.0, atol=1e-12)


def test_lab_round_trip_including_the_dark_linear_branch():
    rng = np.random.default_rng(11)
    for values in (rng.uniform(0, 1, (500, 3)), rng.uniform(0, 0.002, (500, 3))):
        np.testing.assert_allclose(
            lab_to_linear_srgb(linear_srgb_to_lab(values)), values, atol=1e-12
        )


def test_srgb_transfer_round_trip():
    values = np.linspace(0.0, 1.0, 1001)
    np.testing.assert_allclose(srgb_decode(srgb_encode(values)), values, atol=1e-12)


def test_known_srgb_anchors():
    """Mid grey and white land where CIELAB says they should.

    White comes out at L* 100.000004 rather than exactly 100: the published
    sRGB-to-XYZ matrix's luminance row sums to 1.0000001, so its own rounding
    shows up here.  That is a difference of about 1e-6 delta-E, and matching
    the published matrix is worth more than forcing this to be exact.
    """
    white = srgb8_to_lab(np.array([255, 255, 255]))
    assert white[0] == pytest.approx(100.0, abs=1e-4)
    assert white[1] == pytest.approx(0.0, abs=1e-3)
    assert white[2] == pytest.approx(0.0, abs=1e-3)
    # sRGB 128 is L* ~53.6, the canonical "middle grey is not 50" result.
    grey = srgb8_to_lab(np.array([128, 128, 128]))
    assert grey[0] == pytest.approx(53.585, abs=0.01)
