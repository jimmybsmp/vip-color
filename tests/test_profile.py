"""Building, storing and overriding the target skin tone."""

from pathlib import Path

import numpy as np
import pytest

from vipcolor.color import delta_e_2000
from vipcolor.faces import Patch, SkinSample
from vipcolor.profile import (
    MIN_TOLERANCE_DE,
    PROFILE_VERSION,
    ProfileError,
    SkinProfile,
    build_profile,
    circular_mean_degrees,
    hue_difference,
    hue_of,
    manual_profile,
    merge_profiles,
)


def _sample(name, lab, *, quality="good"):
    """A SkinSample with a chosen quality verdict, without running a detector."""
    settings = {
        "good": dict(pose_yaw=0.05, face_scale=120.0, chroma_spread=1.0, pixels=500),
        "marginal": dict(pose_yaw=0.35, face_scale=60.0, chroma_spread=3.0, pixels=200),
        "poor": dict(pose_yaw=0.90, face_scale=15.0, chroma_spread=30.0, pixels=20),
    }[quality]
    patch = Patch(
        name="cheek_left",
        center=(0.0, 0.0),
        radius=5.0,
        lab=np.array(lab, dtype=np.float64),
        linear_rgb=np.zeros(3),
        pixels_total=settings["pixels"],
        pixels_used=settings["pixels"],
        clipped_fraction=0.0,
    )
    sample = SkinSample(
        path=Path(name),
        lab=np.array(lab, dtype=np.float64),
        patches=[patch],
        spread=0.0,
        chroma_spread=settings["chroma_spread"],
        luminance_range=0.0,
        colour_note="test fixture",
        faces_found=1,
        detector_score=0.9,
        pose_yaw=settings["pose_yaw"],
        face_scale=settings["face_scale"],
    )
    assert sample.quality == quality, "fixture does not produce the quality it claims"
    return sample


REFERENCE = [
    _sample("a.jpg", (65.0, 23.0, 26.0)),
    _sample("b.jpg", (64.0, 22.5, 27.0)),
    _sample("c.jpg", (66.0, 23.5, 25.5)),
]


def test_circular_mean_handles_the_wrap_point():
    """Hue is an angle; averaging 350 and 10 arithmetically gives 180."""
    mean, _ = circular_mean_degrees(np.array([350.0, 10.0]), np.array([1.0, 1.0]))
    assert mean == pytest.approx(0.0, abs=1e-9) or mean == pytest.approx(360.0, abs=1e-9)

    mean, spread = circular_mean_degrees(np.array([48.0, 50.0]), np.array([1.0, 1.0]))
    assert mean == pytest.approx(49.0, abs=1e-6)
    assert spread > 0.0

    mean, spread = circular_mean_degrees(np.array([45.0, 45.0]), np.array([1.0, 3.0]))
    assert mean == pytest.approx(45.0, abs=1e-9)
    assert spread == pytest.approx(0.0, abs=1e-9)


def test_hue_difference_is_shortest_way_round():
    assert hue_difference(10.0, 350.0) == pytest.approx(20.0)
    assert hue_difference(350.0, 10.0) == pytest.approx(20.0)
    assert hue_difference(48.0, 50.0) == pytest.approx(2.0)


def test_profile_centres_on_its_references():
    profile = build_profile(REFERENCE)
    assert profile.samples_used == 3
    assert profile.render_space == "rendered"
    assert profile.lab[0] == pytest.approx(65.0, abs=0.6)
    # Self-consistency: the stored hue is the hue of the stored centre, to
    # within the 3-decimal rounding the JSON format keeps.
    assert profile.hue == pytest.approx(hue_of(profile.lab_array), abs=1e-3)
    for source in profile.sources:
        assert profile.distance(source.lab) <= profile.tolerance_de


def test_poor_frames_are_recorded_but_do_not_move_the_target():
    """The failure this guards is real: one profile-view frame in a set."""
    clean = build_profile(REFERENCE)
    with_outlier = build_profile(REFERENCE + [_sample("bad.jpg", (58.0, 14.5, 7.6), quality="poor")])

    assert with_outlier.samples_seen == 4
    assert with_outlier.samples_used == 3
    assert not [s for s in with_outlier.sources if s.file == "bad.jpg"][0].used
    assert float(delta_e_2000(np.array(clean.lab), np.array(with_outlier.lab))) < 0.01


def test_marginal_frames_count_less_than_good_ones():
    pulled = build_profile(REFERENCE + [_sample("m.jpg", (60.0, 30.0, 35.0), quality="marginal")])
    unweighted = np.mean(
        [[65.0, 23.0, 26.0], [64.0, 22.5, 27.0], [66.0, 23.5, 25.5], [60.0, 30.0, 35.0]], axis=0
    )
    # The marginal frame moves the target, but less than a plain mean would.
    assert profile_distance(pulled) < float(delta_e_2000(unweighted, np.array(build_profile(REFERENCE).lab)))


def profile_distance(profile):
    return float(delta_e_2000(np.array(profile.lab), np.array(build_profile(REFERENCE).lab)))


def test_refuses_to_mix_rendered_and_raw_references():
    """A JPEG carries Camera Raw's rendering; a neutral RAW render does not."""
    mixed = REFERENCE + [_sample("d.nef", (32.0, 12.8, 8.0))]
    with pytest.raises(ProfileError, match="mixes rendered files"):
        build_profile(mixed)


def test_refuses_when_everything_is_below_the_floor():
    with pytest.raises(ProfileError, match="below 'good'"):
        build_profile([_sample("m.jpg", (65.0, 23.0, 26.0), quality="marginal")], min_quality="good")
    with pytest.raises(ProfileError):
        build_profile([])


def test_tolerance_never_tightens_past_the_visible_threshold():
    identical = [_sample(f"{i}.jpg", (65.0, 23.0, 26.0)) for i in range(3)]
    assert build_profile(identical).tolerance_de == pytest.approx(MIN_TOLERANCE_DE)
    assert build_profile(REFERENCE, tolerance=1.0).tolerance_de == pytest.approx(1.0)


def test_manual_target_from_an_srgb_range_sets_its_own_tolerance():
    """'The face runs about 200-218 red' has to become a real bound."""
    profile = manual_profile(rgb_range=((200, 140, 112), (218, 150, 122)))
    assert profile.origin == "manual"
    assert profile.tolerance_de > MIN_TOLERANCE_DE
    low = manual_profile(rgb=(200, 140, 112))
    high = manual_profile(rgb=(218, 150, 122))
    # The target sits between the corners it was given.
    assert min(low.lab[0], high.lab[0]) <= profile.lab[0] <= max(low.lab[0], high.lab[0])


def test_manual_target_needs_exactly_one_form():
    with pytest.raises(ProfileError):
        manual_profile()
    with pytest.raises(ProfileError):
        manual_profile(lab=(65.0, 23.0, 26.0), rgb=(210, 140, 110))
    with pytest.raises(ProfileError):
        manual_profile(rgb_range=((218, 150, 122), (200, 140, 112)))


def test_merge_weight_spans_measured_to_manual():
    measured = build_profile(REFERENCE)
    hand = manual_profile(lab=(70.0, 18.0, 30.0))

    # Merging nothing in is the identity, up to the stored rounding.
    assert merge_profiles(measured, hand, 0.0).lab == pytest.approx(measured.lab, abs=1e-3)
    all_hand = merge_profiles(measured, hand, 1.0)
    assert all_hand.hue == pytest.approx(hand.hue, abs=1e-3)
    assert all_hand.lightness == pytest.approx(hand.lightness, abs=1e-3)

    half = merge_profiles(measured, hand, 0.5)
    assert min(measured.hue, hand.hue) <= half.hue <= max(measured.hue, hand.hue)
    with pytest.raises(ProfileError):
        merge_profiles(measured, hand, 1.5)


def test_merge_refuses_across_rendering_spaces():
    measured = build_profile(REFERENCE)
    linear_hand = manual_profile(lab=(32.0, 12.0, 8.0), render_space="linear")
    with pytest.raises(ProfileError, match="different renderings"):
        merge_profiles(measured, linear_hand)


def test_save_and_load_round_trip(tmp_path):
    profile = build_profile(REFERENCE)
    path = profile.save(tmp_path / "p.json")
    loaded = SkinProfile.load(path)

    assert loaded.lab == profile.lab
    assert loaded.hue == profile.hue
    assert len(loaded.sources) == len(profile.sources)
    assert loaded.sources[0].file == profile.sources[0].file


def test_load_rejects_a_future_version(tmp_path):
    """Old files must fail loudly rather than be read with today's meaning."""
    import json

    path = tmp_path / "future.json"
    path.write_text(json.dumps({"version": PROFILE_VERSION + 1}))
    with pytest.raises(ProfileError, match="profile version"):
        SkinProfile.load(path)

    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ProfileError, match="not valid JSON"):
        SkinProfile.load(broken)
