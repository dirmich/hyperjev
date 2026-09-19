# Training

`src/hyperjev/student.py` defines the registry-derived encoder + typed-head
contract and an optional PyTorch implementation. The current repository does
not vendor a backbone or claim a trained checkpoint. On a DGX Spark training
image, use the manifest as the compatibility input before adding the actual
tokenizer/backbone and optimization loop:

```bash
uv run hyperjev student manifest --model-id hyperjev-0.3b-poc
```
