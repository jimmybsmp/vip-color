"""RAW decoding, exercised through a synthetic DNG with a known answer."""

import numpy as np
import pytest
from PIL import Image

from synth import mosaic, write_dng
from vipcolor import faces, raw as rawmod
from vipcolor.color import delta_e_2000, srgb_decode
from vipcolor.images import load_linear

FIXTURE = "tests/fixtures/face_astronaut.png"


@pytest.fixture(scope="module")
def source_linear():
    return srgb_decode(np.asarray(Image.open(FIXTURE).convert("RGB"), dtype=np.float64) / 255.0)


def test_normalise_wb_fills_the_missing_second_green():
    """LibRaw reports G2 as 0 on Bayer sensors, meaning 'same as G1'."""
    assert rawmod.normalise_wb([2.4, 1.0, 1.5, 0.0]) == (2.4, 1.0, 1.5, 1.0)


def test_normalise_wb_scales_to_unit_green():
    result = rawmod.normalise_wb([600.0, 256.0, 410.0, 256.0])
    assert result[1] == 1.0
    assert result[0] == pytest.approx(600 / 256)


def test_normalise_wb_rejects_nonsense():
    with pytest.raises(ValueError):
        rawmod.normalise_wb([1.0, 0.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        rawmod.normalise_wb([1.0, 1.0, 1.0])


def test_read_info_recovers_camera_identity(tmp_path, source_linear):
    cfa, neutral = mosaic(source_linear)
    path = write_dng(tmp_path / "id.dng", cfa, as_shot_neutral=neutral)
    info = rawmod.read_info(path)
    assert "NIKON" in info.camera
    assert info.raw_size == (source_linear.shape[1], source_linear.shape[0])
    assert info.daylight_wb[1] == 1.0


@pytest.mark.parametrize(
    "gain",
    [(1.0, 1.0, 1.0), (1.35, 1.0, 0.72), (0.78, 1.0, 1.45), (1.6, 1.0, 0.55)],
)
def test_render_recovers_skin_tone_under_any_illuminant(tmp_path, source_linear, gain):
    """The point of the whole pipeline: the light must not change the answer.

    A frame shot under a strong cast, rendered at the white balance that
    cancels that cast, has to report the same skin Lab as the original.
    """
    reference = faces.sample_skin(load_linear(FIXTURE))

    cfa, neutral = mosaic(source_linear, illuminant_gain=gain)
    path = write_dng(tmp_path / "cast.dng", cfa, as_shot_neutral=neutral)
    rendered = faces.sample_skin(load_linear(path, neutral="asshot", half_size=False))

    assert float(delta_e_2000(rendered.lab, reference.lab)) < 2.0


def test_neutral_source_selection(tmp_path, source_linear):
    cfa, neutral = mosaic(source_linear, illuminant_gain=(1.35, 1.0, 0.72))
    path = write_dng(tmp_path / "n.dng", cfa, as_shot_neutral=neutral)
    info = rawmod.read_info(path)

    assert rawmod.neutral_wb(info, "unity") == (1.0, 1.0, 1.0, 1.0)
    assert rawmod.neutral_wb(info, "asshot") == info.camera_wb
    assert rawmod.neutral_wb(info, "daylight") == info.daylight_wb
    with pytest.raises(ValueError):
        rawmod.neutral_wb(info, "tungsten")


def test_render_is_linear_not_gamma_encoded(tmp_path, source_linear):
    """A linear render of a mid-grey ramp must stay linear.

    If LibRaw's gamma ever stops being honoured, skin sampling still looks
    plausible but the solver's linearity assumption quietly breaks, so this
    is asserted directly.
    """
    ramp = np.zeros((64, 64, 3))
    ramp[:, :, :] = np.linspace(0.05, 0.8, 64)[None, :, None]
    cfa, neutral = mosaic(ramp)
    path = write_dng(tmp_path / "ramp.dng", cfa, as_shot_neutral=neutral)
    rendered = load_linear(path, neutral="asshot", half_size=False).linear

    green = rendered[16:48, :, 1]
    column_means = green.mean(axis=0)
    # A linear ramp stays a straight line; an sRGB-encoded one does not.
    x = np.arange(len(column_means))
    fit = np.polyfit(x, column_means, 1)
    residual = np.max(np.abs(np.polyval(fit, x) - column_means))
    assert residual < 0.02
