# Phase 6 model registry

Model manifests are registered with dataset hash, source commit, task versions,
calibration version, runtime, precision, and lifecycle status. The registry
allows only explicit transitions:

```text
trained -> evaluated -> calibrated -> candidate -> canary -> active
                                             \-> retired
```

Activating a new model retires the previous active model and records an event.
Rollback requires an explicit model id; there is no implicit “latest” artifact
selection. The local JSON registry is an operational contract for the later
Rust/service registry, not evidence that a trained checkpoint exists.

The manifest builder makes that evidence boundary explicit. It accepts a
training plan, a held-out calibration manifest, and an existing checkpoint;
then it records the dataset hash, typed task versions, calibration version,
git commit, runtime/precision, absolute checkpoint path, and checkpoint
SHA-256. It fails if the checkpoint or required artifact metadata is missing.

```bash
uv run hyperjev model manifest \
  --training-plan runs/phase3/training-plan.json \
  --calibration runs/phase3/calibration.json \
  --checkpoint runs/phase3/reference-student.pt \
  --git-commit "$(git rev-parse HEAD)" \
  --output runs/phase6/model-manifest.json
uv run hyperjev model register --manifest runs/phase6/model-manifest.json
```

Manifest generation is not promotion. Registration starts at `trained`; quality,
calibration, canary, and rollback evidence are still required for later explicit
transitions.

CLI examples:

```bash
uv run hyperjev model register --manifest manifest.json
uv run hyperjev model promote hyperjev-0.3b-poc --to canary
uv run hyperjev model promote hyperjev-0.3b-poc --to active
uv run hyperjev model rollback hyperjev-previous
```
