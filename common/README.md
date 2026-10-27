# common — measurement and reporting shared by all three models

Scripts here belong to no single model: deployment measurement and comparison-report generation.

## Layout

```
common/
├─ deploy_metrics.py     measure parameters / weight size / latency / throughput / GPU memory under one protocol
├─ to_fp16.py            convert a checkpoint to float16 and report its size (reproduces the 195.1 MB figure)
├─ make_compare.py       aggregate Laya vs FinBERT metrics into a Markdown comparison report
└─ reports/
   ├─ deploy_metrics.json  measured deployment metrics for all three models (finbert / laya / fb_laya)
   ├─ deploy_fb.log        measurement output for the FinBERT side
   └─ deploy_laya.log      measurement output for the Laya side
```

## Why separate processes are mandatory

`deploy_metrics.py`'s three subcommands depend on torch builds from different conda environments. Running them back to back in one process makes them interfere with each other (and can silently produce wrong memory or latency figures). **They must be invoked separately**:

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
```

All three write the same `reports/deploy_metrics.json`, each updating only its own key (`finbert` / `laya` / `fb_laya`), so they never overwrite one another.

`laya` and `fb_laya` share the same `bench_laya()` inference path; only the encoder behind `--base` differs, so the figures are directly comparable:

```python
bench_laya()                                      # mmBERT version, ckpt = LAYA_CKPT
bench_laya(FB_LAYA_CKPT, model_key="fin-laya-fb") # FinBERT version, ckpt = FB_LAYA_CKPT
```

## Measurement protocol

| Metric | FinBERT implementation | Laya implementation |
|---|---|---|
| Parameters | `sum(p.numel())` | Read the safetensors header (`struct.unpack("<Q")`) and sum each tensor's shape |
| Weight size | Walk the directory summing `.safetensors/.bin/.pt` | Same |
| Latency p50 / p95 | `transformers` forward, batch=1, 200 runs, percentiles after warmup | `Router.predict`, single request, 100 runs |
| Throughput | 200 runs at batch=1; 30 runs at batch=32 | batch=1 |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` (batch=32) | Same (single request) |

Both measurement rounds ran on an RTX 4090 24 GB with all other processes idle.

## Measured results

| | Laya (mmBERT) | FinLaya-FinBERT | FinBERT classifier |
|---|---:|---:|---:|
| Parameters | 321.9 M | **117.2 M** | 102.3 M |
| Weight size | 614.0 MB (170 tensors, all F16) | **223.6 MB** (235 tensors, all F16) | 390 MB (fp32) → 195 MB (F16) |
| Latency p50 | 18.64 ms | **8.66 ms** | 5.93 ms |
| Latency p95 | 19.34 ms | **8.92 ms** | 5.98 ms |
| Throughput batch=1 | 53.0 QPS | **113.0 QPS** | 145.3 QPS |
| Throughput batch=32 | — | — | 2857.7 QPS |
| Peak GPU memory | 1499 MB | **667 MB** | 486 MB |

> On the size figure: `weights_only_mb` / `dir_size_mb` skip `checkpoint_latest/` by default. That is a mid-training snapshot holding the same weights as the top-level `model.safetensors`; including it nearly doubles the reported size (Laya would show 1228 MB, FinLaya-FinBERT 447 MB).

> The 195.1 MB figure for FinBERT converted to F16 comes from a separate earlier measurement whose conversion code was not kept.
> `to_fp16.py` has since been written to reproduce it the same way:
>
> ```bash
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --check    # inspect the dtype distribution
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --dry-run  # estimate the size only
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --out <dir>
> ```
>
> That script has only been syntax-checked and **was never re-run on the server**; the fp32 figure of 390 MB is the reliably reproducible one.

## Generating the comparison report

```bash
python common/make_compare.py       # → docs/COMPARE_Laya_vs_FinBERT.md
```

Data sources:

| File | Contents |
|---|---|
| `FinLaya-mmBERT/reports/eval/ft_*_metrics.json` | Laya, full protocol |
| `FinLaya-mmBERT/reports/eval/ft_*_metrics_strict.json` | Laya, strict protocol |
| `FinBert/reports/finbert_all.json` | FinBERT, full + strict protocol |
| `common/reports/deploy_metrics.json` | Deployment metrics for both |

Paths can be overridden with `LAYA_REPORTS`; by default they resolve to the `reports/` directory here.
