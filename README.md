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

**No face means no answer.** Frames where no face is detected are reported as
failures, never guessed at.

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

40 tests, no proprietary files needed. `tests/synth.py` writes a valid
uncompressed DNG from a rendered image, which is how the LibRaw path is tested:
a frame is synthesised under a known illuminant cast, decoded at the white
balance that cancels it, and the recovered skin tone is required to land within
2 delta-E of the original. It currently lands within 1.3 across casts from
heavily warm to heavily blue.

The face fixture is NASA's public-domain portrait of Eileen Collins, via
scikit-image's sample data.

## Known gaps

- Not yet validated against a real NEF, or against files with a known-correct
  hand grade. Everything RAW-side is proven against synthetic DNGs only.
- The `--face largest` default is a guess about which person is the subject.
  For group shots, check the overlay.
- Camera differences between the Z8, Z7 and D810 are handled by LibRaw's
  per-body colour matrices and daylight presets, but this has not been
  confirmed on real files from all three bodies.
