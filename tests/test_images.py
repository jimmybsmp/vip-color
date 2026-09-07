"""Loading, colour-space handling and file dispatch."""

import numpy as np
import pytest
from PIL import Image, ImageCms

from vipcolor.color import delta_e_2000
from vipcolor.images import is_raw, is_supported, iter_images, load_linear

FIXTURE = "tests/fixtures/face_astronaut.png"


def test_suffix_dispatch():
    assert is_raw("a.NEF") and is_raw("b.nef") and is_raw("c.dng")
    assert not is_raw("d.jpg")
    assert is_supported("d.JPG") and is_supported("e.tiff")
    assert not is_supported("f.txt")


def test_untagged_file_is_assumed_srgb_and_says_so():
    loaded = load_linear(FIXTURE)
    assert "assumed sRGB" in loaded.colour_note


def test_embedded_srgb_profile_is_honoured_as_a_no_op(tmp_path):
    """The ICC path must run, and must not shift a file that is already sRGB."""
    plain = load_linear(FIXTURE)

    tagged = tmp_path / "tagged.jpg"
    profile = ImageCms.createProfile("sRGB")
    Image.open(FIXTURE).convert("RGB").save(
        tagged, quality=98, icc_profile=ImageCms.ImageCmsProfile(profile).tobytes()
    )
    converted = load_linear(tagged)

    assert "converted from embedded profile" in converted.colour_note
    difference = np.abs(converted.linear - plain.linear).mean()
    assert difference < 0.01  # JPEG quantisation only


def test_exif_rotation_is_applied(tmp_path):
    """A portrait frame stored sideways has to be uprighted before detection."""
    source = Image.open(FIXTURE).convert("RGB").resize((200, 100))
    rotated = tmp_path / "rot.jpg"
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW on display
    source.save(rotated, exif=exif)

    loaded = load_linear(rotated)
    assert loaded.size == (100, 200)


def test_missing_file_and_unsupported_type(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_linear(tmp_path / "nope.jpg")
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello")
    with pytest.raises(ValueError):
        load_linear(unsupported)


def test_iter_images_lists_a_folder(tmp_path):
    for name in ("a.jpg", "b.NEF", "c.txt"):
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in iter_images(tmp_path)] == ["a.jpg", "b.NEF"]
    assert [p.name for p in iter_images(tmp_path, raw_only=True)] == ["b.NEF"]
