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
    if sample.faces_found > 1:
        click.echo(f"    faces found   {sample.faces_found}")
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
                detected, _how = facesmod.select_face(found, image, select=face)
                measured = sample_skin(
                    image, patches=chosen, landmarks=detected, select=face
                )
                measured.faces_found = len(found)
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
