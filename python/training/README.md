# Training

`src/hyperjev/student.py` defines the registry-derived encoder + typed-head
contract and an optional PyTorch implementation. `src/hyperjev/training.py`
validates normalized records and writes a reproducible training plan. The
current repository does not vendor a backbone or claim a trained checkpoint.
On a DGX Spark training image, use the plan as the compatibility input before
adding the actual tokenizer/backbone and optimization loop:

```bash
uv run hyperjev student manifest --model-id hyperjev-0.3b-poc
uv run hyperjev train plan \
  --dataset runs/phase2/dataset.jsonl \
  --output runs/phase3/training-plan.json \
  --model-id hyperjev-0.3b-poc
```
