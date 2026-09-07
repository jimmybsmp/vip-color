"""Annotated overlays, so patch placement can be checked by eye.

Numbers alone cannot tell you the forehead patch landed in a fringe.  This
draws where the sampler actually looked.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .faces import SkinSample
from .images import LinearImage


def write_overlay(
    image: LinearImage,
    sample: SkinSample,
    destination: Path,
    *,
    landmarks: np.ndarray | None = None,
) -> Path:
    """Write an sRGB PNG of ``image`` with sampled patches circled."""
    import cv2

    canvas = cv2.cvtColor(image.encoded8(), cv2.COLOR_RGB2BGR).copy()
    longest = max(canvas.shape[:2])
    # Keep annotation legible on a 4000 px render and a 500 px crop alike.
    thickness = max(1, round(longest / 700))
    font_scale = max(0.4, longest / 1600)

    if landmarks is not None:
        for x, y in landmarks:
            cv2.circle(canvas, (int(x), int(y)), max(1, thickness // 2), (0, 140, 255), -1)

    for patch in sample.patches:
        center = (int(patch.center[0]), int(patch.center[1]))
        cv2.circle(canvas, center, int(patch.radius), (0, 230, 0), thickness)
        label = f"{patch.name} L{patch.lab[0]:.0f} a{patch.lab[1]:.0f} b{patch.lab[2]:.0f}"
        cv2.putText(
            canvas,
            label,
            (center[0] - int(patch.radius), center[1] - int(patch.radius) - 4 * thickness),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 230, 0),
            thickness,
            cv2.LINE_AA,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), canvas):
        raise OSError(f"could not write overlay to {destination}")
    return destination
