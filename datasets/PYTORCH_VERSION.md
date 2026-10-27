# PyTorch version selection (training / development environment)

For this machine: Windows + **GTX 1650 (Turing, compute capability sm_75, 4 GB VRAM)**, driver 466.92 → supports CUDA up to 11.3.

## Conclusion

```
torch 2.7.1+cu118   +   transformers 4.57.x   +   Python 3.10 / 3.11
```

```bash
# Create an isolated environment (3.11 is the most reliable; 3.13 already on this machine also works, see below)
py -3.11 -m venv .venv
.venv\Scripts\activate

# Key point: you must use the cu118 wheels. cu121/cu124/cu126 install fine on driver 466.92 but cannot use the GPU
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Laya dependencies (via a domestic mirror)
pip install transformers>=4.48 safetensors huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# Verify
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# Expected: 2.7.1+cu118 11.8 True (7, 5)
```

## Why 2.7.1

| Constraint | Explanation |
| --- | --- |
| Laya declares `torch>=2.0.0` | Anything from 2.0 up; **the lower bound is not the binding constraint**, the pitfall lies elsewhere |
| Driver 466.92 = CUDA 11.3 | Only **cu118** or earlier wheels work; cu121/cu124/cu126 are built against a higher target than the driver allows and fail with "CUDA driver version is insufficient" |
| cu118 is end-of-life | **The last version on the official cu118 index is 2.7.1** (2.8 onward only ships cu126/cu128/cu129), so 2.7.1 is the ceiling on this path |
| Python support | 2.7.1+cu118 ships the full cp39/310/311/312/313 Windows wheel set, so 3.13 on this machine also installs |

Verified against the official indexes:

- `https://download.pytorch.org/whl/cu118/torch/` → highest is `2.7.1`
- `https://download.pytorch.org/whl/cu121/torch/` → highest is `2.5.1` (and this machine's driver cannot use it)

## transformers version

**Must be ≥ 4.48.0**: earlier versions lack ModernBERT support and model loading fails outright. Laya's
`pyproject.toml` also pins `transformers>=4.48.0`.

Note that **transformers ≥ 5.0 requires `torch>=2.6`**, which is compatible with this setup; but
`option_layout="parallel"` training (which needs transformers >= 5) only works on the eager backend and
there is no reason to enable it here. **Pin the 4.5x series**:

```bash
pip install "transformers>=4.48,<5"
```

## A performance trap you must know (GTX 1650 specific)

ModernBERT's hidden dimension is 1024, so the attention softmax has `dim_size = 1024`. Looking at PyTorch's
`aten/src/ATen/native/cuda/SoftMax.cu`, forward branch:

```cuda
if (dim_size <= 2048 && dim_size*sizeof(scalar_t) <= 8192) {
    dispatch_softmax_forward<...>();          // sub-warp / 128-thread fast path (with C10_WARP_SIZE=16 each lane handles 2,
} else {                                      // dim_size=1024 lands just inside <=2048)
    ... ILP / 512-thread general path ...
}
```

But `at::cuda::warp_size()` is **32** on sm_5x–7x, and at that point the general path's condition
`dim_size(1024) <= warp_size^2(1024)` requires **at least 2 elements per lane** — yet 1024 elements can only
give each of 32 lanes one element → **it falls back to a 32-thread slow kernel**. Turing (sm_75) is the most
typical victim architecture.

Result: on a GTX 1650 the per-sample attention cost is an order of magnitude worse than theory, and **bf16
does not help** (the bottleneck is thread count, not bandwidth). Therefore:

- Do not expect to train on 53k samples with this 4 GB card; neither VRAM nor speed is sufficient
- Use this machine only for **`--dry-run` data validation, `predict` smoke tests, and small-batch debugging**
- Do the real training on a cloud GPU (T4/A10/4090 are all sm_80+, warp=32 but fp16 goes through
  `dispatch_softmax_forward`'s half path or the tensor-core branch, so they are unaffected)

If this is the only card you have and you must train locally, the practical approach is to **cap `max_len` at
256 or below**, use micro-batch 1 with a large grad-accum, and accept being 10× slower than the cloud.

## Alternatives

| Scenario | Choice |
| --- | --- |
| 4 GB card locally, only debugging | `torch 2.7.1+cu118` + `max_len<=256` |
| No usable GPU locally, pure CPU data validation | `pip install torch --index-url https://download.pytorch.org/whl/cpu` (any version; 2.7.1 is fine) |
| Cloud training (recommended) | `torch 2.7.1+cu126` or `2.8.x+cu128` with an A10/4090; cu118 works for T4 too |
| After a driver upgrade | Driver ≥ 525 unlocks the whole cu118/cu121 line; ≥ 550 unlocks cu126. **Even so, do not train on this machine** — 4 GB of VRAM is the hard limit |

## What not to do

- ❌ `pip install torch` (the default source pulls cu12x builds, and it installs globally)
- ❌ Installing `torch 2.8+` and then hunting for cu118 wheels — there is no cu118 build from 2.8 onward
- ❌ Installing `tilelang` (`laya[fast]`) — it needs sm_80+; the GTX 1650 does not support it
- ❌ Pairing `laya[onnx]` with torch<2.3 (NumPy 2 conflicts with the torch 1.x ABI, as the official pyproject comment already states)
