"""Command line entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from . import raw as rawmod
from .color import delta_e_2000
from .faces import DEFAULT_PATCHES, NoFaceFound, SkinSample, sample_rect, sample_skin
from .images import iter_images, load_linear
from .profile import (
    QUALITY_ORDER,
    ProfileError,
    SkinProfile,
    build_profile,
    manual_profile,
    merge_profiles,
)

PATCH_CHOICES = ("forehead", "cheek_left", "cheek_right", "chin")


def _parse_rect(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = value.replace(" ", "").split(",")
    if len(parts) != 4:
        raise click.BadParameter("expected four comma-separated integers: X,Y,W,H")
    try:
        x, y, width, height = (int(p) for p in parts)
    except ValueError as error:
        raise click.BadParameter(f"not an integer: {error}") from error
    if width <= 0 or height <= 0:
        raise click.BadParameter("width and height must be positive")
    return (x, y, width, height)


def _sample_to_dict(sample: SkinSample) -> dict:
    return {
        "file": str(sample.path),
        "camera": sample.camera,
        "colour_note": sample.colour_note,
        "lab": [round(float(v), 3) for v in sample.lab],
        "srgb8": list(sample.srgb8),
        "faces_found": sample.faces_found,
        "detector_score": round(sample.detector_score, 4),
        "pose_yaw": round(sample.pose_yaw, 4),
        "face_scale_px": round(sample.face_scale, 1),
        "quality": sample.quality,
        "spread_de2000": round(sample.spread, 3),
        "chroma_spread_de2000": round(sample.chroma_spread, 3),
        "luminance_range": round(sample.luminance_range, 3),
        "patches": [
            {
                "name": p.name,
                "center": [round(c, 1) for c in p.center],
                "radius": round(p.radius, 1),
                "lab": [round(float(v), 3) for v in p.lab],
                "srgb8": list(p.srgb8),
                "pixels_used": p.pixels_used,
                "pixels_total": p.pixels_total,
                "clipped_fraction": round(p.clipped_fraction, 4),
            }
            for p in sample.patches
        ],
        "warnings": sample.warnings,
    }


def _print_sample(sample: SkinSample) -> None:
    lab = sample.lab
    click.echo(f"  {sample.path.name}")
    if sample.camera:
        click.echo(f"    camera        {sample.camera}")
    click.echo(f"    render        {sample.colour_note}")
    if sample.faces_found:
        click.echo(
            f"    detection     {sample.faces_found} face(s),"
            f" confidence {sample.detector_score:.2f},"
            f" {sample.face_scale:.0f} px between eyes, yaw {sample.pose_yaw:.2f}"
        )
        colour = {"good": "green", "marginal": "yellow", "poor": "red"}[sample.quality]
        click.secho(f"    quality       {sample.quality}", fg=colour)
    click.echo(
        f"    skin Lab      L* {lab[0]:6.2f}   a* {lab[1]:6.2f}   b* {lab[2]:6.2f}"
        f"      (sRGB {sample.srgb8[0]},{sample.srgb8[1]},{sample.srgb8[2]})"
    )
    click.echo(
        f"    agreement     chroma spread {sample.chroma_spread:.2f} dE2000,"
        f" L* range {sample.luminance_range:.2f}"
    )
    for patch in sample.patches:
        click.echo(
            f"      {patch.name:<12} L* {patch.lab[0]:6.2f}  a* {patch.lab[1]:6.2f}"
            f"  b* {patch.lab[2]:6.2f}   {patch.pixels_used:>6}/{patch.pixels_total} px"
            f" ({patch.usable_fraction:.0%} kept)"
            + (f"  {patch.clipped_fraction:.0%} clipped" if patch.clipped_fraction > 0.01 else "")
        )
    for warning in sample.warnings:
        click.secho(f"    warning: {warning}", fg="yellow")


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="vipcolor")
def main() -> None:
    """Skin-tone driven colour correction for repeat portrait subjects."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def info(path: Path) -> None:
    """Report a RAW file's camera identity and white balance presets."""
    from .images import is_raw

    if not is_raw(path):
        raise click.ClickException(f"{path.name} is not a RAW file")
    details = rawmod.read_info(path)
    click.echo(f"file            {details.path}")
    click.echo(f"camera          {details.camera}")
    click.echo(f"raw size        {details.raw_size[0]} x {details.raw_size[1]}")
    click.echo(f"colour filter   {details.color_desc}")
    click.echo("as-shot wb      " + "  ".join(f"{m:.4f}" for m in details.camera_wb))
    click.echo("daylight wb     " + "  ".join(f"{m:.4f}" for m in details.daylight_wb))
    click.echo("camera -> XYZ matrix (LibRaw, rows RGB):")
    for row in details.rgb_xyz_matrix[:3]:
        click.echo("    " + "  ".join(f"{v: .5f}" for v in row))


@main.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--patches",
    default=",".join(DEFAULT_PATCHES),
    show_default=True,
    help=f"Comma-separated patches to sample, from: {', '.join(PATCH_CHOICES)}.",
)
@click.option(
    "--neutral",
    type=click.Choice(rawmod.NEUTRAL_SOURCES),
    default="daylight",
    show_default=True,
    help="Which white balance defines the neutral RAW render.",
)
@click.option(
    "--rect",
    default=None,
    help="Skip face detection and sample X,Y,W,H instead (frames with no face).",
)
@click.option("--full-size", is_flag=True, help="Demosaic at full resolution instead of half.")
@click.option(
    "--overlay",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write an annotated PNG per file into this folder to check patch placement.",
)
@click.option(
    "--face",
    default="largest",
    show_default=True,
    help="Which face to measure when several are in frame: largest, center, or an index.",
)
@click.option("--landmarks/--no-landmarks", default=False, help="Draw all landmarks on the overlay.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def sample(
    path: Path,
    patches: str,
    neutral: str,
    rect: str | None,
    full_size: bool,
    overlay: Path | None,
    face: str,
    landmarks: bool,
    as_json: bool,
) -> None:
    """Measure skin tone in a file or a folder, and print it in CIELAB."""
    from . import faces as facesmod
    from .debug import write_overlay

    chosen = tuple(p.strip() for p in patches.split(",") if p.strip())
    unknown = [p for p in chosen if p not in PATCH_CHOICES]
    if unknown:
        raise click.BadParameter(f"unknown patch(es): {', '.join(unknown)}")
    rectangle = _parse_rect(rect)

    targets = iter_images(path)
    if not targets:
        raise click.ClickException(f"no supported images found in {path}")

    results: list[dict] = []
    failures: list[tuple[Path, str]] = []
    for target in targets:
        try:
            image = load_linear(target, neutral=neutral, half_size=not full_size)
            detected = None
            if rectangle is not None:
                measured = sample_rect(image, rectangle)
            else:
                found = facesmod.detect_faces(image)
                picked, _how = facesmod.select_face(found, image, select=face)
                detected = picked.landmarks
                measured = sample_skin(
                    image, patches=chosen, landmarks=detected, select=face
                )
                measured.faces_found = len(found)
                measured.detector_score = picked.score
                if len(found) > 1:
                    note = (
                        f"{len(found)} faces detected; measured the "
                        f"{_how}. Check the overlay, and use --face to pick another."
                    )
                    if note not in measured.warnings:
                        measured.warnings.insert(0, note)
        except (NoFaceFound, ValueError, OSError) as error:
            failures.append((target, str(error)))
            if not as_json:
                click.secho(f"  {target.name}: {error}", fg="red")
            continue

        if overlay is not None:
            write_overlay(
                image,
                measured,
                overlay / f"{target.stem}_patches.png",
                landmarks=detected if landmarks else None,
            )
        results.append(_sample_to_dict(measured))
        if not as_json:
            _print_sample(measured)

    if as_json:
        click.echo(
            json.dumps(
                {
                    "samples": results,
                    "failures": [{"file": str(p), "error": e} for p, e in failures],
                },
                indent=2,
            )
        )
    elif len(targets) > 1:
        click.echo(f"\n{len(results)} sampled, {len(failures)} failed, of {len(targets)} files")
        if len(results) > 1:
            labs = [r["lab"] for r in results]
            import numpy as np

            labs_array = np.array(labs)
            centre = labs_array.mean(axis=0)
            spread = max(float(delta_e_2000(row, centre)) for row in labs_array)
            click.echo(
                f"mean skin Lab   L* {centre[0]:.2f}  a* {centre[1]:.2f}  b* {centre[2]:.2f}"
                f"   (furthest file is {spread:.2f} dE2000 from it)"
            )

    if failures and not results:
        sys.exit(1)


def _parse_triple(value: str, what: str) -> tuple[float, float, float]:
    parts = value.replace(" ", "").split(",")
    if len(parts) != 3:
        raise click.BadParameter(f"expected three comma-separated numbers for {what}")
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError as error:
        raise click.BadParameter(f"not a number in {what}: {error}") from error


def _parse_rgb_range(value: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Parse 'R,G,B-R,G,B': the low and high corners of an sRGB box."""
    if "-" not in value:
        raise click.BadParameter("expected two sRGB triples separated by '-', e.g. 200,140,115-215,155,130")
    low_text, high_text = value.split("-", 1)
    low = _parse_triple(low_text, "the lower sRGB bound")
    high = _parse_triple(high_text, "the upper sRGB bound")
    return (
        tuple(int(round(v)) for v in low),  # type: ignore[return-value]
        tuple(int(round(v)) for v in high),  # type: ignore[return-value]
    )


def _describe_profile(profile: SkinProfile) -> None:
    click.echo(f"  origin        {profile.origin}  ({profile.render_space} space)")
    if profile.note:
        click.echo(f"  note          {profile.note}")
    click.echo(
        f"  target Lab    L* {profile.lab[0]:6.2f}   a* {profile.lab[1]:6.2f}   "
        f"b* {profile.lab[2]:6.2f}      (sRGB {profile.srgb8[0]},{profile.srgb8[1]},"
        f"{profile.srgb8[2]})"
    )
    if profile.samples_used:
        click.echo(
            f"  spread        L* +/-{profile.lab_std[0]:.2f}  "
            f"a* +/-{profile.lab_std[1]:.2f}  b* +/-{profile.lab_std[2]:.2f}"
        )
    click.echo(
        f"  hue           {profile.hue:.2f} deg +/-{profile.hue_std:.2f}"
        f"   (tolerance {profile.hue_tolerance_deg:.2f} deg)"
    )
    click.echo(f"  chroma        {profile.chroma:.2f} +/-{profile.chroma_std:.2f}")
    click.echo(f"  lightness     L* {profile.lightness:.2f} +/-{profile.lightness_std:.2f}")
    click.echo(f"  tolerance     {profile.tolerance_de:.2f} dE2000")
    if profile.samples_seen:
        click.echo(f"  built from    {profile.samples_used} of {profile.samples_seen} frames")


@main.command("build-profile")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("skin-profile.json"),
    show_default=True,
    help="Where to write the profile.",
)
@click.option(
    "--min-quality",
    type=click.Choice(QUALITY_ORDER),
    default="marginal",
    show_default=True,
    help="Lowest sample quality allowed to contribute to the target.",
)
@click.option(
    "--tolerance",
    type=float,
    default=None,
    help="Fix the target tolerance in dE2000 instead of deriving it from the spread.",
)
@click.option(
    "--face",
    default="largest",
    show_default=True,
    help="Which face to measure when several are in frame: largest, center, or an index.",
)
@click.option(
    "--overlay",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Write an annotated PNG per reference frame, to check what was measured.",
)
@click.option("--target-lab", default=None, help="Hand-supplied Lab target, 'L,a,b'.")
@click.option("--target-rgb", default=None, help="Hand-supplied sRGB target, 'R,G,B'.")
@click.option(
    "--target-rgb-range",
    default=None,
    help="Hand-supplied sRGB range, 'R,G,B-R,G,B'. Its midpoint is the target and "
    "half its width becomes the tolerance.",
)
@click.option(
    "--target-mode",
    type=click.Choice(["merge", "override"]),
    default="merge",
    show_default=True,
    help="Whether a hand-supplied target blends with the measured one or replaces it.",
)
@click.option(
    "--merge-weight",
    type=float,
    default=0.5,
    show_default=True,
    help="How much the hand-supplied target counts when merging, 0 to 1.",
)
@click.option("--dry-run", is_flag=True, help="Report the profile without writing it.")
@click.option("--json", "as_json", is_flag=True, help="Emit the profile as JSON on stdout.")
def build_profile_command(
    path: Path,
    output: Path,
    min_quality: str,
    tolerance: float | None,
    face: str,
    overlay: Path | None,
    target_lab: str | None,
    target_rgb: str | None,
    target_rgb_range: str | None,
    target_mode: str,
    merge_weight: float,
    dry_run: bool,
    as_json: bool,
) -> None:
    """Learn the target skin tone from a folder of correctly graded frames."""
    from . import faces as facesmod
    from .debug import write_overlay

    hand_supplied = [target_lab, target_rgb, target_rgb_range]
    if sum(value is not None for value in hand_supplied) > 1:
        raise click.BadParameter("give at most one of --target-lab, --target-rgb, --target-rgb-range")

    targets = iter_images(path)
    if not targets:
        raise click.ClickException(f"no supported images found in {path}")

    samples: list[SkinSample] = []
    failures: list[tuple[Path, str]] = []
    for target in targets:
        try:
            image = load_linear(target)
            found = facesmod.detect_faces(image)
            picked, _how = facesmod.select_face(found, image, select=face)
            measured = sample_skin(image, landmarks=picked.landmarks, select=face)
            measured.faces_found = len(found)
            measured.detector_score = picked.score
        except (NoFaceFound, ValueError, OSError) as error:
            failures.append((target, str(error)))
            click.secho(f"  skipped {target.name}: {error}", fg="red")
            continue
        if overlay is not None:
            write_overlay(image, measured, overlay / f"{target.stem}_patches.png")
        samples.append(measured)

    if not samples:
        raise click.ClickException("no reference frame could be measured")

    try:
        measured_profile = build_profile(
            samples, min_quality=min_quality, tolerance=tolerance
        )
    except ProfileError as error:
        raise click.ClickException(str(error)) from error

    profile = measured_profile
    if any(value is not None for value in hand_supplied):
        try:
            hand = manual_profile(
                lab=_parse_triple(target_lab, "--target-lab") if target_lab else None,
                rgb=(
                    tuple(int(round(v)) for v in _parse_triple(target_rgb, "--target-rgb"))
                    if target_rgb
                    else None
                ),
                rgb_range=_parse_rgb_range(target_rgb_range) if target_rgb_range else None,
                tolerance=tolerance,
                render_space=measured_profile.render_space,
            )
            profile = (
                hand
                if target_mode == "override"
                else merge_profiles(measured_profile, hand, weight=merge_weight)
            )
        except ProfileError as error:
            raise click.ClickException(str(error)) from error

    if as_json:
        import dataclasses

        click.echo(json.dumps(dataclasses.asdict(profile), indent=2))
    else:
        click.echo("")
        _describe_profile(profile)
        excluded = [source for source in profile.sources if not source.used]
        if excluded:
            click.echo("")
            click.secho(f"  {len(excluded)} frame(s) excluded from the target:", fg="yellow")
            for source in excluded:
                click.echo(
                    f"    {Path(source.file).name}  quality={source.quality}"
                    f"  yaw={source.pose_yaw:.2f}  eye={source.face_scale:.0f}px"
                )
        for source in profile.sources:
            if source.used:
                distance = profile.distance(source.lab)
                marker = " " if distance <= profile.tolerance_de else "!"
                click.echo(
                    f"   {marker} {Path(source.file).name[:44]:<46}"
                    f" dE {distance:5.2f}   hue {source.hue:6.2f}"
                )
        if failures:
            click.echo(f"\n  {len(failures)} file(s) could not be measured")

    if dry_run:
        click.echo("\n  --dry-run: nothing written")
        return

    profile.save(output)
    if not as_json:
        click.echo(f"\n  written to {output}")


@main.command("show-profile")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def show_profile_command(path: Path) -> None:
    """Print a saved profile."""
    try:
        profile = SkinProfile.load(path)
    except ProfileError as error:
        raise click.ClickException(str(error)) from error
    click.echo("")
    _describe_profile(profile)
