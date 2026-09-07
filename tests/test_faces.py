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
