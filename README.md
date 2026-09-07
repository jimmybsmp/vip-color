# vipcolor

Consistent skin-tone colour correction for a recurring portrait subject,
working from Nikon RAW files and writing Adobe Camera Raw XMP sidecars.

**Status: Phases 1–3 complete** — RAW decode, skin sampling, the reference
profile builder, and the white balance solver. Phases 4–7 (XMP writer,
Photoshop launch, match-this-shot, batch reporting) are not built yet.
Nothing is written to disk yet: `correct` reports and stops.

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

# Learn the target from frames you consider correctly graded
vipcolor build-profile ~/reference-jpegs --face center -o skin-profile.json
vipcolor show-profile skin-profile.json

# Solve the white balance that puts a RAW's skin on target (writes nothing)
vipcolor correct shoot/DSC_1234.NEF --profile skin-profile.json

# Check a solve against a grade you already know is right
vipcolor verify DSC_1234.NEF --profile skin-profile.json --expected-xmp DSC_1234.xmp
vipcolor verify DSC_1234.NEF --profile skin-profile.json --expected-temp 5400 --expected-tint 8

# Override or blend in a target you judged by eye
vipcolor build-profile ~/reference-jpegs --target-rgb-range 200,140,112-218,150,122
vipcolor build-profile ~/reference-jpegs --target-lab 66,23,27 --target-mode override
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

## The reference profile

`build-profile` measures a folder of frames whose colour you already consider
right, and writes the target to JSON. Rebuild it whenever you add references —
it is a plain re-run of the same command.

- **Bad frames are excluded, not averaged in.** Every sample carries the
  quality verdict described above, and anything below `--min-quality`
  (default `marginal`) is recorded in the file but contributes nothing. On a
  real six-frame set this excluded a three-quarter-profile shot sitting 15
  delta-E from the rest.
- **Marginal frames count less than good ones**, at 0.4 weight, rather than
  being all-or-nothing.
- **The tolerance is derived from the evidence** — mean plus two standard
  deviations of the references' own scatter, floored at 2 delta-E, which is
  roughly where a difference on skin stops being visible. `--tolerance` fixes
  it by hand.
- **Hue spread is computed as a circular statistic.** Skin never sits near the
  0/360 wrap, but a mean that is only right for convenient inputs is a trap for
  whoever reuses it.
- **The centre is the mean Lab**, with hue and chroma read off it, so
  `hue_of(profile.lab) == profile.hue`. The circular statistics set the
  *spread*, which is what drives the tolerance.
- **A profile records its rendering space, and mixing is refused.** Building
  one profile from both JPEGs and RAWs raises an error rather than averaging
  two things that differ by two stops — see the finding below.

You can also supply the target by hand, which is the "I know the face runs
about here" case: `--target-lab`, `--target-rgb`, or `--target-rgb-range`. A
range becomes a real bound — its midpoint is the target and half its diagonal
becomes the tolerance — instead of being silently treated as an exact value.
`--target-mode merge|override` and `--merge-weight` decide whether it blends
with the measured profile or replaces it. Merging blends lightness, chroma and
hue in their own terms rather than averaging a* and b*, so a disagreement
between the two does not drag the result towards grey.

### What it produced on a real set

Six frames of one subject across four events and four different backdrops:

| | default `--face largest` | `--face center` |
| --- | --- | --- |
| hue spread | ±2.13° | **±0.63°** |
| tolerance | 3.88 delta-E | **3.02 delta-E** |
| worst frame | 3.61 delta-E | 2.53 delta-E |

The difference is entirely the group shot, where `largest` measured the wrong
person. Five frames shot at different events on different backdrops agreeing on
hue to two-thirds of a degree is the result that says this approach works —
and an eyeballed sRGB range supplied by hand landed within 1° of it.

## The solver

`correct` finds the temperature and tint that put a frame's skin on the target,
and reports them. It writes nothing yet.

**Temperature and tint mean what Camera Raw means by them.** A temperature is a
point on the Planckian locus and tint is a displacement perpendicular to it in
the CIE 1960 UCS, which is the model the DNG specification describes. Turning
that illuminant into camera multipliers uses the camera's own XYZ-to-camera
matrix. That this is the right matrix is checked rather than assumed: the
camera's response to D65 through it reproduces LibRaw's own daylight preset to
1e-6 on a real NEF. The model puts D65 at 6500 K and D50 at 4998 K against
their true 6504 and 5003, and a blackbody returns its own temperature at tint
zero. Sign conventions are verified by rendering a synthetic grey card through
the real pipeline: positive tint renders magenta, higher temperature renders
warmer, as Camera Raw shows them.

**One render, not dozens.** White balance is applied to camera RGB before the
camera-to-sRGB matrix, so a render at one white balance reaches any other by a
single 3x3 similarity transform. That is exact, not an approximation, and it
turns a search that would take a minute into one that takes about a second and
a half. Measured against real LibRaw renders of a D500 NEF, the model error is
**0.001 delta-E**.

That shortcut has one precondition — the matrix rawpy reports must be the one
LibRaw renders with — and it does not always hold. On DNG input LibRaw adapts
the matrix during processing and reports the adapted one, where the shortcut
drifts by **1 to 4 delta-E**. So it is not trusted silently: `--check-model`
renders once more at the solved setting and reports how far the model is from
reality, and warns above 0.5 delta-E. Native NEF, which is what this tool is
for, is unaffected.

**The search follows a valley.** Temperature and tint trade off against each
other — warm-and-green lands on nearly the same skin colour as cool-and-magenta
— so the objective has a narrow diagonal valley. Coordinate descent stalls on
it: an earlier version stopped 814 K short, because from the valley floor
moving along either axis alone goes uphill. A simplex with a few restarts
follows the diagonal. Given a white balance to recover, it now returns it to
within **0.2 K and 0.001 tint** across nine cases from 2800 K to 12000 K.

**Lightness is matched before colour is compared**, and that scale factor is
reported as the exposure trim. Comparing a Camera Raw–rendered reference
against a linear RAW render directly would have the solver trying to fix a
rendering difference by moving the white balance.

Illuminants outside the camera's gamut — the corners of the search box, where
the sensor's response to the implied light goes negative — are scored as
infinitely bad, which keeps the search feasible without a separate test.
Solutions outside 2500–10000 K or a tint of +/-150 are reported and flagged
rather than clamped into looking reasonable.

### Checking it against your own work

`vipcolor verify` is the accuracy test, and it runs entirely on your machine:

```bash
vipcolor verify DSC_1234.NEF --profile skin-profile.json --expected-xmp DSC_1234.xmp
```

Point it at a RAW you have already graded by hand together with the temperature
and tint you chose, and it reports how close the solver came, in kelvin and in
mired. Mired is the honest measure — a gap of a few hundred kelvin means
something quite different at 3000 K than at 9000 K, while about 10 mired is
roughly where a difference stops being visible on skin.

`vipcolor info` also prints the file's as-shot temperature and tint. Opening the
same file in Camera Raw with White Balance set to As Shot and comparing those
two numbers is the quickest check that this tool's idea of temperature agrees
with Adobe's.

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

86 tests, no proprietary files needed. `tests/synth.py` writes a valid
uncompressed DNG from a rendered image, which is how the LibRaw path is tested:
a frame is synthesised under a known illuminant cast, decoded at the white
balance that cancels it, and the recovered skin tone is required to land within
2 delta-E of the original. It currently lands within 1.3 across casts from
heavily warm to heavily blue.

The face fixture is NASA's public-domain portrait of Eileen Collins, via
scikit-image's sample data.

## Known gaps

- The solver has not been checked against a grade a person made. Every
  accuracy figure above is either internal consistency or a comparison against
  LibRaw; whether it lands on *your* look is what `vipcolor verify` is for.
- Camera Raw applies a tone curve that a neutral RAW render does not, which
  changes saturation but not hue. Solving a rendered-space target against a RAW
  therefore matches hue soundly and chroma only indicatively; the warning says
  so, and `--hue-weight` lets hue dominate.
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
