# Optical Flow from Scratch

Two classical dense optical flow estimators implemented with nothing but NumPy
and SciPy: Lucas Kanade and Horn Schunck. Given two consecutive frames the code
estimates how every pixel moved from the first frame to the second.

Both methods start from the same idea, brightness constancy. A small patch of
the image keeps its intensity as it moves, so the change in brightness over time
is explained entirely by motion across the spatial gradient. Linearising that
statement gives the optical flow constraint

    Ix * u + Iy * v + It = 0

where `Ix` and `Iy` are the spatial gradients of the image, `It` is the
difference between the two frames, and `(u, v)` is the flow we want. One
equation per pixel is not enough to solve for two unknowns, so each method adds
a different assumption to close the gap.

## The two methods

**Lucas Kanade** assumes the flow is constant inside a small window around each
pixel. That turns the single under determined equation into an over determined
2x2 least squares system, which is solved in closed form at every pixel using
the local structure tensor. It is fast and works well where the image has
texture in both directions. In flat regions the system is poorly conditioned, so
a small ridge term is added to the diagonal to keep it stable.

**Horn Schunck** treats flow estimation as a global energy minimisation. It
trades the brightness constancy residual against a smoothness penalty that
discourages the flow from changing abruptly between neighbours. The minimiser is
found by iterating a simple update that nudges each pixel toward the local
average flow and then corrects it along the image gradient. The `alpha` weight
sets how much smoothness wins over data fidelity.

## Flow convention

`u` is the horizontal displacement along columns (the x axis) and `v` is the
vertical displacement along rows (the y axis). Positive `u` means content moved
right between the frames, positive `v` means it moved down. The synthetic data
generator uses the same `(dx, dy)` convention, so the ground truth shift lines up
directly with the estimated `(u, v)`.

## Layout

```
src/flow.py      gradients, lucas_kanade, horn_schunck, make_shifted_pair
tests/           pytest behaviour checks
```

## Usage

```python
import numpy as np
from src.flow import make_shifted_pair, lucas_kanade, horn_schunck

# Build a frame pair related by a known shift of one pixel to the right.
frame1, frame2 = make_shifted_pair(shape=(80, 80), shift=(1.0, 0.0), seed=0)

u, v = lucas_kanade(frame1, frame2, window_size=21, sigma=1.0)
print("Lucas Kanade median flow:", np.median(u), np.median(v))

u, v = horn_schunck(frame1, frame2, alpha=1.0, num_iterations=400, sigma=1.0)
print("Horn Schunck median flow:", np.median(u), np.median(v))
```

`make_shifted_pair` generates a random smoothed texture and translates it by a
known amount, so the shift you pass in is the ground truth that the estimators
should recover. Smoothing matters here. The flow constraint is a linear
approximation, and it only tracks motion that is small relative to the spatial
scale of the texture, so a sharp random image would break it while a smoothed one
behaves well.

## Tests

The test suite checks the behaviour that should hold for any correct
implementation rather than fixed magic numbers:

- Gradients vanish for a constant image and match a known ramp.
- Both estimators report near zero flow when the two frames are identical.
- Both estimators recover a known translation on the image interior within a
  tolerance, across several shift directions including sub pixel and diagonal
  shifts.
- The sign of the recovered flow matches the direction of motion.
- The two independent estimators agree with each other on the same input.
- Output shapes match the input and mismatched frame shapes raise an error.

Run them with:

```
python -m pytest tests/ -q
```

All 15 tests pass on CPU in well under a second with no downloads.

## Notes

The implementations are deliberately single scale. Both methods rely on the
linearised brightness constancy constraint, which is only valid for motion that
is small compared to the texture scale. Larger displacements would need a
coarse to fine pyramid, which is a natural extension but is outside the scope
here.
