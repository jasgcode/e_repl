---
date: 2026-11-17
status: adopted
summary: Images are BGR uint8 arrays, encoded to JPEG once when the model's code shows them
---

# 0004. Images are BGR uint8 arrays, encoded to JPEG once when the model's code shows them

## Decision

`show()` and `llm_query` take HxW gray or HxWx3 BGR `uint8` arrays. `encode_image` shrinks and
encodes them with Pillow at that moment. The resulting bytes are sent to the model and written to
the run directory unchanged.

## Why

- The first tools for this harness are OpenCV and a RealSense `bgr8` stream. One channel order
  across tools, prompt, and harness removes a silent red-blue swap.
- Encoding once makes the logged image the image the model saw, at the same size and quality.
- Pillow instead of OpenCV keeps the harness installable beside either `opencv-python` or
  `opencv-python-headless`, which cannot share an environment.

## Alternatives

- RGB arrays: the convention of Pillow and most models, but every camera frame in the first
  project would need a flip at each `show()`.
- Encoding separately for the API and for the log, as in `simple_101`: the log then held the
  full-size frame, not what was sent.
- Accepting arrays and Pillow images both: a Pillow image is RGB, so the two inputs would
  disagree on channel order.

## Cost if wrong

A project whose tools produce RGB flips channels in its own tools, or changes one line in
`encode_image` and supersedes this decision.
