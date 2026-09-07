# vipcolor

Consistent skin-tone colour correction for a recurring portrait subject,
working from Nikon RAW files and writing Adobe Camera Raw XMP sidecars.

**Status: Phase 1 complete** — RAW decode and skin sampling. Phases 2–7
(profile builder, WB solver, XMP writer, Photoshop launch, match-this-shot,
batch reporting) are not built yet.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

Python 3.11+. Everything runs offline: MediaPipe 0.10.x carries its face-mesh
weights inside the wheel, so nothing is fetched at runtime.

## What works today

```bash
# What does this RAW file say about itself?
vipcolor info /path/to/DSC_1234.NEF

# Measure skin tone, one file or a whole folder
vipcolor sample /path/to/DSC_1234.NEF
vipcolor sample ~/reference-jpegs --overlay ./check

# Group shot? Pick the face
vipcolor sample group.jpg --face center --overlay ./check

# No face in frame (a test file, or a grey card)
vipcolor sample chart.NEF --rect 1200,800,300,300
```

`--overlay DIR` writes an annotated PNG per file showing exactly which pixels
were measured. **Use it the first time you run against your own files** — the
numbers cannot tell you a patch landed in someone's fringe, but the picture can.

## The decisions that matter

**Rendering.** RAW files are decoded with `use_camera_wb=False`,
`use_auto_wb=False`, explicit white balance multipliers, `gamma=(1,1)`,
`output_bps=16`, `no_auto_bright=True`, `output_color=sRGB`. In plain terms: a
scene-linear 16-bit render in sRGB primaries, with nothing automatic.

- White balance is multiplicative, so the solver's linear fit is only valid on
  linear pixels.
- `no_auto_bright` matters more than it looks. LibRaw's auto-brighten rescales
  each file independently, which would hide exposure differences between frames
  — the differences we want to measure.
- The neutral anchor is the camera's **daylight** preset, not unity
  multipliers. Unity means raw sensor counts, which differ per body, so a Z8
  and a D810 would land on different "neutral" renders and a profile built from
  both would be incoherent. `--neutral asshot|unity` overrides this.

**Colour space.** CIELAB under **D65**, matching sRGB's native white, so the
only chromatic adaptation anywhere is the one the white-balance solve is
explicitly modelling. Differences are CIEDE2000, never Euclidean CIE76 — CIE76
overstates differences exactly where skin sits.

**Rendered files carry a colour space too.** A Lightroom export is very often
Adobe RGB or ProPhoto rather than sRGB. Tagged files are converted through
littleCMS; untagged files are assumed sRGB and the output says so. Reading an
Adobe RGB export as sRGB would bake several delta-E of error into the target
profile, silently.

**Detection is two-stage.** Face Mesh's built-in detector is BlazeFace
*short range*, which assumes the face fills much of the frame. Event
photography does the opposite -- a head a couple of hundred pixels wide in a
3000 px frame -- and that model simply does not see it. So faces are located
with the **full-range** detector, and Face Mesh is handed a padded crop of each
one. Landmarks are mapped back to the full-resolution image, which is where
skin is actually measured. On a set of real event frames this was the
difference between zero faces found and all of them.

**Sampling.** Three patches by default — forehead, both cheeks below the eyes —
placed from blends of several MediaPipe landmarks so a small landmark wobble
moves a patch by much less than its own radius. Inside each patch, pixels are
rejected for clipping, for sitting in the brightest 15% (sheen, sweat, glare)
or darkest 10% (pores, stubble, creases), and then for disagreeing in colour
with the patch's own median by more than 12 delta-E — which is what removes
hair, spectacle frames and lip edge. The reported value is the median of what
survives, typically ~75% of the pixels.

**Disagreement is reported, not hidden.** `chroma spread` is the largest
CIEDE2000 between patches with brightness held constant; `L* range` is the
brightness difference. Splitting them matters because a forehead brighter than
a cheek is ordinary directional light, while patches that disagree in *colour*
mean mixed light sources or a patch that is not on skin — only the second
threatens a white-balance solve.

**Every sample carries a verdict.** `good`, `marginal` or `poor`, from head
pose, face size, patch agreement and pixel counts. Head turn is scored as
`yaw`: where the nose tip falls along the line between the outer eye corners,
0 square-on to 1 in profile. It matters because a turned head puts one cheek in
different light from the other -- on a real reference set, frontal frames
scored 0.02 to 0.10 while a three-quarter profile scored 0.88 and its patches
disagreed by 30 delta-E. That frame's skin measurement sat 15 delta-E from the
rest of the set; without a pose score it would have quietly poisoned the
profile.

**No face means no answer.** Frames where no face is detected are reported as
failures, never guessed at.

## A finding that shapes the solver

A reference JPEG has already been through Camera Raw: baseline exposure, tone
curve, the photographer's own grade. The neutral RAW render here deliberately
has none of that. On real files the gap is about two stops — reference JPEGs of
the subject sample at L* 64, while a linear NEF render of a comparable face
samples at L* 32.

So **absolute Lab cannot be compared between a rendered reference and a linear
RAW render.** Measured on a real NEF across four stops of exposure:

| quantity | change across ±2 stops |
| --- | --- |
| hue angle | **0.29 degrees** |
| chroma / L* | 52% |
| absolute Lab (delta-E 2000) | 40.5 |

Hue angle is the exposure-invariant quantity, and hue is also most of what
white balance actually controls. The consequence for Phase 3 is that the solve
should target **hue angle, and chroma only after lightness has been matched**,
with exposure handled as its own trim — not a single delta-E against an
absolute Lab target, which would try to correct a rendering difference by
moving the white balance and would land in the wrong place.

## Deviations from the original spec

- **`colour-science` is not used.** The Lab and CIEDE2000 code is hand-rolled
  and verified against `colour-science` to 0.0 maximum difference over 4000
  random Lab pairs plus the Sharma test vectors. That drops `colour` and
  `scipy` — about 60 MB and a compiled dependency — from the install.
- **MediaPipe is pinned to `0.10.x`.** 1.x removed the `mp.solutions` Face Mesh
  API in favour of the Tasks API, which downloads a 3.7 MB model at runtime.
  0.10.x bundles its weights. It pins `numpy<2`, so numpy is pinned to match.
- **`dlib` is not needed.** MediaPipe installs cleanly on Python 3.11.

## Testing

```bash
.venv/bin/python -m pytest
```

44 tests, no proprietary files needed. `tests/synth.py` writes a valid
uncompressed DNG from a rendered image, which is how the LibRaw path is tested:
a frame is synthesised under a known illuminant cast, decoded at the white
balance that cancels it, and the recovered skin tone is required to land within
2 delta-E of the original. It currently lands within 1.3 across casts from
heavily warm to heavily blue.

The face fixture is NASA's public-domain portrait of Eileen Collins, via
scikit-image's sample data.

## Known gaps

- Validated on a real Nikon NEF (D500, 5600x3728): camera identity, white
  balance presets, colour matrix, decode, detection and sampling all work,
  in about 7 seconds per file. Not yet validated on the Z8, Z7 or D810, nor
  against any file with a known-correct hand grade.
- The `--face largest` default is a guess about which person is the subject,
  and on a real five-person group frame it picked the wrong man. Check the
  overlay on any group shot, or prefer `--face center`. Better still, build
  profiles from solo frames only.
- Detection can still fail on a very wide frame containing a very small face:
  MediaPipe resizes its detector input to a square without letterboxing, so a
  2.5:1 frame horizontally squashes every face in it. Letterboxing was tried
  and made real-world results slightly worse, so it was not adopted. Tiled
  detection would be the fix if this shows up in practice; `--rect` is the
  escape hatch meanwhile.
- Camera differences between the Z8, Z7 and D810 are handled by LibRaw's
  per-body colour matrices and daylight presets, but this has not been
  confirmed on real files from all three bodies.
