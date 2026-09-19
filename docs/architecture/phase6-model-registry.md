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

CLI examples:

```bash
uv run hyperjev model register --manifest manifest.json
uv run hyperjev model promote hyperjev-0.3b-poc --to canary
uv run hyperjev model promote hyperjev-0.3b-poc --to active
uv run hyperjev model rollback hyperjev-previous
```
