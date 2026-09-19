# Phase 3 Student contract

The product model remains an encoder with typed decision heads. The registry is
the source of truth for the head inventory:

- boolean tasks produce two-class logits;
- choice tasks produce one logit per registered candidate;
- score tasks produce two parameters for the score head;
- all heads consume one shared pooled encoder representation.

`student_manifest` is dependency-free and can be checked in CI. The optional
`build_torch_model` backend creates the minimal shared encoder/head module when
PyTorch is installed on the DGX Spark training image. No backbone checkpoint,
tokenizer, or trained quality result is bundled yet.

Calibration starts with deterministic temperature fitting over held-out logits.
The fitted artifact records temperature and NLL before/after; it must be kept
separate from training data and tied to a model version before promotion.

`HyperJev-D` remains a separate research track and is not imported by this
product model path.
