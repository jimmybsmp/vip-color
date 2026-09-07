"""Temperature and tint, in the terms Camera Raw uses.

White balance has two representations here and they must not be confused:

- **Multipliers** -- what LibRaw renders with, and what the sensor actually
  needs. Unambiguous, but meaningless to a person.
- **Temperature and tint** -- what Camera Raw shows and what goes in the
  XMP sidecar. Meaningful to a person, but only defined relative to a model
  of what illuminants exist.

The model here is the one the DNG specification describes: a temperature is
a point on the Planckian locus, and tint is a displacement perpendicular to
that locus in the CIE 1960 UCS, where perpendicular distance corresponds to
the direction the eye reads as green/magenta.

Converting that illuminant to camera multipliers uses the camera's own
XYZ-to-camera matrix, which LibRaw exposes as ``rgb_xyz_matrix``.  That this
is the right matrix is not assumed: the camera's response to D65 through it
reproduces LibRaw's own ``daylight_whitebalance`` to within 1e-6, which is
checked in the tests.  It is the same camera characterisation Adobe ships,
so temperatures computed here should land close to Camera Raw's -- but
"close" is a claim only a person with Camera Raw open can confirm, which is
what ``vipcolor verify`` is for.
"""

from __future__ import annotations

import numpy as np

#: Adobe's tint scaling, from the DNG SDK's dng_temperature.  Tint is a
#: displacement in CIE 1960 UCS divided by this, which is what puts tint on
#: the roughly -150..+150 scale Camera Raw shows.
TINT_SCALE = -3000.0

#: The Planckian approximation below is only defined over this range.
MIN_TEMP_K = 1667.0
MAX_TEMP_K = 25000.0

#: What this tool considers a plausible answer.  Outside it, the solve is
#: reported but flagged: a skin sample that wants 14000 K is telling you
#: something about the sample, not about the light.
SANE_TEMP_RANGE = (2500.0, 10000.0)
SANE_TINT_RANGE = (-150.0, 150.0)


def planckian_xy(temperature: float) -> tuple[float, float]:
    """CIE 1931 xy chromaticity of a blackbody at ``temperature`` kelvin.

    The cubic approximation to the Planckian locus (Kim et al.), which is
    accurate to well under a just-noticeable difference across its range and
    avoids carrying colour matching functions around.
    """
    T = float(np.clip(temperature, MIN_TEMP_K, MAX_TEMP_K))
    if T <= 4000.0:
        x = -0.2661239e9 / T**3 - 0.2343589e6 / T**2 + 0.8776956e3 / T + 0.179910
    else:
        x = -3.0258469e9 / T**3 + 2.1070379e6 / T**2 + 0.2226347e3 / T + 0.240390

    if T <= 2222.0:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif T <= 4000.0:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return float(x), float(y)


def xy_to_uv(x: float, y: float) -> tuple[float, float]:
    """CIE 1931 xy to CIE 1960 UCS uv."""
    denominator = -2.0 * x + 12.0 * y + 3.0
    if abs(denominator) < 1e-12:
        raise ValueError(f"degenerate chromaticity ({x}, {y})")
    return (4.0 * x / denominator, 6.0 * y / denominator)


def uv_to_xy(u: float, v: float) -> tuple[float, float]:
    """CIE 1960 UCS uv back to CIE 1931 xy."""
    denominator = 2.0 * u - 8.0 * v + 4.0
    if abs(denominator) < 1e-12:
        raise ValueError(f"degenerate chromaticity ({u}, {v})")
    return (3.0 * u / denominator, 2.0 * v / denominator)


def _locus_uv(temperature: float) -> np.ndarray:
    return np.array(xy_to_uv(*planckian_xy(temperature)), dtype=np.float64)


def _locus_normal(temperature: float) -> np.ndarray:
    """Unit vector perpendicular to the Planckian locus, in uv.

    Oriented so that moving along +normal makes the illuminant greener,
    which -- since tint describes the *correction* applied, not the light --
    is what makes a positive tint read as magenta in the rendered image, as
    Camera Raw shows it.
    """
    step = max(1.0, temperature * 1e-4)
    ahead = _locus_uv(min(MAX_TEMP_K, temperature + step))
    behind = _locus_uv(max(MIN_TEMP_K, temperature - step))
    tangent = ahead - behind
    length = float(np.hypot(*tangent))
    if length < 1e-12:
        raise ValueError(f"degenerate locus tangent at {temperature} K")
    tangent /= length
    # Rotate the tangent by 90 degrees.
    return np.array([-tangent[1], tangent[0]], dtype=np.float64)


def temp_tint_to_xy(temperature: float, tint: float) -> tuple[float, float]:
    """The illuminant chromaticity for a temperature and tint."""
    base = _locus_uv(temperature)
    offset = _locus_normal(temperature) * (float(tint) / TINT_SCALE)
    return uv_to_xy(*(base + offset))


def xy_to_temp_tint(x: float, y: float) -> tuple[float, float]:
    """The temperature and tint of an illuminant chromaticity.

    Searched rather than solved in closed form: the nearest point on the
    locus is found in reciprocal temperature, where the locus is close to
    straight, then tint is the signed perpendicular distance to it.
    """
    target = np.array(xy_to_uv(x, y), dtype=np.float64)

    def distance_at(mired: float) -> float:
        return float(np.hypot(*(target - _locus_uv(1e6 / mired))))

    # Coarse sweep in mired, then a golden-section refinement.
    grid = np.linspace(1e6 / MAX_TEMP_K, 1e6 / MIN_TEMP_K, 400)
    best = min(grid, key=distance_at)
    low = max(grid[0], best - 4.0)
    high = min(grid[-1], best + 4.0)
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(60):
        span = high - low
        a, b = high - golden * span, low + golden * span
        if distance_at(a) < distance_at(b):
            high = b
        else:
            low = a
        if span < 1e-9:
            break
    mired = (low + high) / 2.0
    temperature = float(np.clip(1e6 / mired, MIN_TEMP_K, MAX_TEMP_K))

    offset = target - _locus_uv(temperature)
    tint = float(np.dot(offset, _locus_normal(temperature)) * TINT_SCALE)
    return temperature, tint


def white_xy_to_multipliers(xyz_to_camera: np.ndarray, x: float, y: float) -> np.ndarray:
    """Camera white balance multipliers for an illuminant chromaticity.

    The camera's response to the illuminant is what a neutral surface under
    it produces; the multipliers that neutralise it are the reciprocal,
    normalised so green is 1.
    """
    if y <= 0:
        raise ValueError(f"illuminant chromaticity has non-positive y: {y}")
    white_xyz = np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)
    response = np.asarray(xyz_to_camera, dtype=np.float64)[:3] @ white_xyz
    if np.any(response <= 0):
        raise ValueError(f"camera response to this illuminant is not positive: {response}")
    return response[1] / response


def multipliers_to_white_xy(xyz_to_camera: np.ndarray, multipliers) -> tuple[float, float]:
    """The illuminant chromaticity implied by white balance multipliers."""
    multipliers = np.asarray(multipliers, dtype=np.float64)[:3]
    if np.any(multipliers <= 0):
        raise ValueError(f"multipliers must be positive, got {multipliers}")
    response = multipliers[1] / multipliers
    white_xyz = np.linalg.solve(np.asarray(xyz_to_camera, dtype=np.float64)[:3], response)
    total = float(white_xyz.sum())
    if total <= 0:
        raise ValueError("implied illuminant has non-positive XYZ sum")
    return (float(white_xyz[0] / total), float(white_xyz[1] / total))


def temp_tint_to_multipliers(xyz_to_camera: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    return white_xy_to_multipliers(xyz_to_camera, *temp_tint_to_xy(temperature, tint))


def multipliers_to_temp_tint(xyz_to_camera: np.ndarray, multipliers) -> tuple[float, float]:
    return xy_to_temp_tint(*multipliers_to_white_xy(xyz_to_camera, multipliers))


def is_plausible(temperature: float, tint: float) -> list[str]:
    """Complaints about a solved white balance, empty if it looks sane."""
    complaints = []
    if not SANE_TEMP_RANGE[0] <= temperature <= SANE_TEMP_RANGE[1]:
        complaints.append(
            f"temperature {temperature:.0f} K is outside {SANE_TEMP_RANGE[0]:.0f}-"
            f"{SANE_TEMP_RANGE[1]:.0f} K"
        )
    if not SANE_TINT_RANGE[0] <= tint <= SANE_TINT_RANGE[1]:
        complaints.append(
            f"tint {tint:+.0f} is outside {SANE_TINT_RANGE[0]:+.0f}..{SANE_TINT_RANGE[1]:+.0f}"
        )
    return complaints
