# Calibration

`src/hyperjev/calibration.py` provides deterministic held-out temperature
fitting and probability conversion without a numerical framework dependency.
`src/hyperjev/metrics.py` adds task-specific ECE, Brier, typed quality, and
score interval metrics. Risk-coverage curves and threshold promotion remain
model/data-dependent follow-up gates.

Create a versioned calibration manifest from held-out logits and labels:

```bash
uv run hyperjev calibrate \
  --input heldout-logits.json \
  --output runs/phase3/calibration.json
```
