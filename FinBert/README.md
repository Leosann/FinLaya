# FinBERT — discriminative baseline

A control baseline built on `yiyanghkust/finbert-tone-chinese` (102M, pretrained on Chinese financial text). To compare fairly with Laya, **each decision task gets its own fine-tuned classifier head**, so four tasks means four independent models.

## Task mapping

Laya's three decision primitives mapped onto FinBERT classifier heads:

| FinBERT task name | Dataset | qid | Primitive | Classes | Train / eval |
|---|---|---|---:|---:|---:|
| `absa` | CFSC-ABSA | `sentiment_absa` | `choice` | 3 | 33,184 / 8,312 |
| `finfe` | FinFE | `sentiment_social` | `choice` | 3 | 16,157 / 2,020 |
| `nsp_has` | FinNSP | `neg_has` | `noul` | 2 | 4,000 / 500 |
| `nsp_subject` | FinNSP | `neg_subject` | `choice` | 19 | 4,000 / 259 |

> `nsp_subject` has 19 classes rather than Laya's 21: FinBERT's label set is drawn directly from the **training set**, and two entities (two long company names in the training data) appear only in the evaluation set with zero training samples, so they never enter the label set. This is an inherent limitation of the discriminative route — **a new entity requires retraining**, whereas Laya only needs the `criteria` in the question text edited.

## Layout

```
FinBert/
├─ scripts/
│  ├─ setup_finbert.sh          standalone conda env (isolated from the laya env)
│  ├─ finbert_baseline.py       [core] train four tasks + evaluate + full metric set
│  └─ fb_eval.py                batched evaluation entry point, appends to reports/finbert_all.json
├─ logs/
│  └─ finbert_train.log         actual training output for the four tasks
└─ reports/
   └─ finbert_all.json          both metric sets (full + strict): per-class P/R/F1, ECE, Brier, NLL, selective accuracy, latency, GPU memory
```

## Running

```bash
# 1) Environment (python 3.11 + torch 2.6.0+cu126 + transformers>=4.40,<5 + sklearn)
bash scripts/setup_finbert.sh

ENV_PY=/data/miniconda/envs/finbert/bin/python

# 2) Train the four classifier heads
$ENV_PY scripts/finbert_baseline.py train

# 3) Evaluate — must be batched; running all at once gets killed by the OOM killer with no Traceback
$ENV_PY scripts/fb_eval.py absa
$ENV_PY scripts/fb_eval.py finfe
$ENV_PY scripts/fb_eval.py nsp_has
$ENV_PY scripts/fb_eval.py nsp_subject
```

`fb_eval.py` computes two tags per task, `full` and `strict` (strict removes samples whose `state` hash overlaps the training set), and appends them to `reports/finbert_all.json`, so running in batches never overwrites earlier results.

## Training hyperparameters

```python
epochs=3, batch_size=32, lr=2e-5, max_len=256, seed=42
AdamW(weight_decay=0.01) + linear warmup(10%) + fp16 GradScaler + clip_grad_norm(1.0)
```

Measured wall time (RTX 4090):

| Task | Time | Final-epoch loss |
|---|---:|---:|
| `absa` | 218 s | 0.1371 |
| `finfe` | 63 s | 0.3593 |
| `nsp_has` | 28 s | 0.0429 |
| `nsp_subject` | 16 s | 0.2516 |

## Metric implementations

| Metric | Implementation |
|---|---|
| Macro-F1 | Over classes **that have support in the test set** (`macro_f1_present`); zero-support classes are excluded from the average |
| Per-class P/R/F1 | Emitted per class, with support |
| ECE | 15 equal-width bins, weighted sum of `|acc - conf|` |
| Brier | Multiclass `Σ(p_i - y_i)² / n`; a top-1 variant `(p_top - 1)²` is also reported |
| NLL | Mean of `-log p(correct)` |
| AURC | Area under the risk–coverage curve; measures whether confidence ranking is reliable |
| Selective accuracy | Sort by confidence descending, take the top X%, and report accuracy on that subset (X = 50 / 80 / 90) |
| Latency / throughput | `latency_p50_ms` / `latency_p95_ms` / `throughput_qps`, measured after batch=64 warmup |
| GPU memory | `torch.cuda.max_memory_allocated()` |

## Results summary

| Task | n | acc | mF1 | ECE | strict acc | strict n |
|---|---:|---:|---:|---:|---:|---:|
| FinFE | 2,020 | 79.26% | 0.7776 | 0.0947 | 79.52% | 923 |
| FinNSP `neg_has` | 500 | 95.60% | 0.9559 | 0.0311 | 95.59% | 499 |
| FinNSP `neg_subject` | 259 | 95.37% | 0.5038¹ | 0.0616 | 95.37% | 259 |
| CFSC-ABSA | 8,312 | 92.78% | 0.8881 | 0.0339 | 92.94% | 7,870 |

¹ 5 of the 19 classes have zero support; over the 14 classes with support the figure is **0.6838**.

Full numbers are in `reports/finbert_all.json`; the side-by-side comparison is in [../docs/COMPARE_Laya_vs_FinBERT.md](../docs/COMPARE_Laya_vs_FinBERT.md).

## Notes

- **A standalone conda env is required.** `laya` needs `transformers>=4.48,<5` while the FinBERT branch uses `>=4.40,<5`; installing both together makes them clobber each other's dependencies.
- **The trained weights are not in this repository** (about 1.5 GB for the four heads). `reports/finbert_all.json` records `model_dir_mb` and `n_params` for each evaluation, so the size conclusions remain reproducible.
- The `TASKS` mapping is hardcoded at the top of `finbert_baseline.py`; adding a task takes one more `(train_file, eval_file, qid)` line.
