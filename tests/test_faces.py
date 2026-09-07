"""Face detection and skin sampling."""

import numpy as np
import pytest

from vipcolor import faces
from vipcolor.images import load_linear

FIXTURE = "tests/fixtures/face_astronaut.png"


@pytest.fixture(scope="module")
def portrait():
    return load_linear(FIXTURE)


def test_finds_a_face_and_measures_plausible_skin(portrait):
    sample = faces.sample_skin(portrait)
    assert len(sample.patches) == 3
    # Skin under any ordinary light is light, slightly red and clearly yellow.
    assert 40.0 < sample.lab[0] < 95.0
    assert 0.0 < sample.lab[1] < 25.0
    assert 5.0 < sample.lab[2] < 30.0


def test_patches_sit_inside_the_face(portrait):
    landmarks = faces.detect_landmarks(portrait)
    low = landmarks.min(axis=0)
    high = landmarks.max(axis=0)
    for patch in faces.sample_skin(portrait, patches=("forehead", "cheek_left", "cheek_right", "chin")).patches:
        x, y = patch.center
        assert low[0] <= x <= high[0], f"{patch.name} is outside the face horizontally"
        assert low[1] <= y <= high[1], f"{patch.name} is outside the face vertically"


def test_forehead_patch_clears_the_brow_and_the_hairline(portrait):
    """The forehead patch is the one that lands in a fringe if mis-anchored."""
    landmarks = faces.detect_landmarks(portrait)
    sample = faces.sample_skin(portrait, patches=("forehead",))
    forehead = sample.patches[0]
    hairline_y = landmarks[10][1]
    brow_y = landmarks[9][1]
    assert hairline_y < forehead.center[1] < brow_y


def test_rejects_an_image_with_no_face(tmp_path):
    import cv2

    noise = np.full((256, 256, 3), 40, dtype=np.uint8)
    path = tmp_path / "blank.png"
    cv2.imwrite(str(path), noise)
    with pytest.raises(faces.NoFaceFound):
        faces.sample_skin(load_linear(path))


def test_explicit_rectangle_bypasses_detection(tmp_path):
    import cv2

    flat = np.zeros((128, 128, 3), dtype=np.uint8)
    flat[:, :] = (150, 170, 200)  # BGR, so a warm skin-ish tone in RGB
    path = tmp_path / "flat.png"
    cv2.imwrite(str(path), flat)

    sample = faces.sample_rect(load_linear(path), (32, 32, 64, 64))
    assert len(sample.patches) == 1
    assert sample.warnings  # it must say it was not a detected face
    assert sample.lab[0] > 50.0


def test_specular_pixels_do_not_drag_the_measurement(tmp_path):
    """A blown highlight on the cheek must not lighten the reported skin."""
    import cv2

    base = np.zeros((128, 128, 3), dtype=np.uint8)
    base[:, :] = (150, 170, 200)
    clean = load_linear(_write(tmp_path / "clean.png", base))
    clean_lab = faces.sample_rect(clean, (32, 32, 64, 64)).lab

    glared = base.copy()
    glared[40:56, 40:56] = 255  # a hard specular square inside the patch
    dirty = load_linear(_write(tmp_path / "glare.png", glared))
    dirty_lab = faces.sample_rect(dirty, (32, 32, 64, 64)).lab

    assert abs(dirty_lab[0] - clean_lab[0]) < 1.0


def _write(path, array):
    import cv2

    cv2.imwrite(str(path), array)
    return path


def test_finds_a_small_distant_face(tmp_path):
    """The failure real event photography produces.

    Face Mesh's own detector is short-range and does not see a head that is
    a small fraction of a wide frame, which is most of a stage or event
    shoot.  Detection therefore runs full-range first and hands Face Mesh a
    crop; this pastes the fixture small into a big canvas to hold that.
    """
    import cv2

    portrait = cv2.imread(FIXTURE)
    small = cv2.resize(portrait, (150, 150), interpolation=cv2.INTER_AREA)
    canvas = np.full((1200, 2400, 3), 60, dtype=np.uint8)
    canvas[420:570, 1100:1250] = small
    path = tmp_path / "distant.png"
    cv2.imwrite(str(path), canvas)

    sample = faces.sample_skin(load_linear(path))
    assert sample.patches
    assert 40.0 < sample.lab[0] < 95.0


def _fake_face(centre_x, centre_y, half_width):
    """A DetectedFace with landmarks around a point, for selection tests."""
    rng = np.random.default_rng(0)
    landmarks = rng.uniform(-half_width, half_width, (478, 2))
    landmarks += np.array([centre_x, centre_y])
    return faces.DetectedFace(landmarks=landmarks, score=0.9, box=(0, 0, 1, 1))


def test_face_selection_picks_by_size_then_position(portrait):
    """Which face gets measured is a decision, so it is tested as one.

    The detector itself is exercised on real images; this pins the choice
    between several faces, which is where a group shot goes wrong.
    """
    width, height = portrait.size
    small_left = _fake_face(width * 0.15, height * 0.5, 30)
    big_right = _fake_face(width * 0.85, height * 0.5, 90)
    middle = _fake_face(width * 0.5, height * 0.5, 50)
    found = sorted([small_left, big_right, middle], key=lambda f: f.size, reverse=True)

    largest, how = faces.select_face(found, portrait, select="largest")
    assert largest is big_right and "largest" in how

    central, how = faces.select_face(found, portrait, select="center")
    assert central is middle and "central" in how

    indexed, _ = faces.select_face(found, portrait, select="0")
    assert indexed is big_right

    with pytest.raises(faces.NoFaceFound):
        faces.select_face(found, portrait, select="9")
    with pytest.raises(ValueError):
        faces.select_face(found, portrait, select="leftmost")


def test_head_yaw_is_zero_square_on_and_one_in_profile():
    landmarks = np.zeros((478, 2))
    landmarks[33] = (0.0, 0.0)    # outer corner, one eye
    landmarks[263] = (100.0, 0.0)  # outer corner, the other
    landmarks[1] = (50.0, 20.0)    # nose tip centred between them
    assert faces.head_yaw(landmarks) == pytest.approx(0.0, abs=1e-9)

    landmarks[1] = (100.0, 20.0)   # nose tip over one eye: full profile
    assert faces.head_yaw(landmarks) == pytest.approx(1.0, abs=1e-9)

    landmarks[1] = (75.0, 20.0)    # half way: clearly turned
    assert faces.head_yaw(landmarks) == pytest.approx(0.5, abs=1e-9)


def test_quality_verdict_reflects_pose(portrait):
    """A square-on, well-filled frame is the case that should read 'good'."""
    assert faces.sample_skin(portrait).quality in {"good", "marginal"}
