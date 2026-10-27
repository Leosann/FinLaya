# FinLaya — Chinese financial fine-tuning of the Laya decision model

This directory holds all of the Laya side: code, configs, logs, and evaluation reports. The main experiment write-up is in **[REPORT.md](REPORT.md)**; the two-model comparison is in **[../docs/COMPARE_Laya_vs_FinBERT.md](../docs/COMPARE_Laya_vs_FinBERT.md)**.

## Layout

```
FinLaya/
├─ REPORT.md                    main experiment report (conclusions / data / training / pitfalls / quick numbers)
├─ scripts/                     runnable scripts, in execution order
│  ├─ train.sh                  dryrun | train | eval | all
│  ├─ baseline.sh               zero-shot baseline on the unfine-tuned backbone
│  └─ eval_ft.sh                post-fine-tune evaluation (calls ../../common/laya/run_eval.py)
├─ configs/
│  ├─ rl_agent_config.json      inference config produced by training (temperatures / abstention thresholds / question defs)
│  ├─ questions.json            full definitions of the 4 qids
│  └─ train_report.json         before/after + temperature fitting + abstention thresholds
├─ logs/                        train_run / dryrun / baseline / eval_ft
├─ reports/
│  ├─ eval/                     ft_* evaluation reports + recomputed metrics (full & strict)
│  └─ baseline/                 base_* unfine-tuned baseline reports
└─ tokenizer/                   tokenizer bundled with the ckpt (3 files)

../common/laya/                  evaluation tools shared with FinBertLaya
  ├─ run_eval.py                 [core] Router.register a local ckpt + evaluate
  ├─ laya_metrics.py             recompute every metric from a report; supports --strict
  ├─ check_leak.py               audit train/test `state` overlap
  └─ strict_acc.py               recompute the strict protocol, aligned to the report's case order
```

## Execution order

```bash
# 0) Data (generate under datasets/ at the repo root, then upload)
python ../datasets/to_laya_format.py
python ../datasets/validate_laya_format.py

# 1) Dry run to confirm 53,341 lines / 55,562 questions parse correctly
bash scripts/train.sh dryrun

# 2) Train (4 epochs, about 32 minutes on a 4090, 9.2 GB GPU memory)
bash scripts/train.sh train

# 3) Evaluate the three test sets
bash scripts/eval_ft.sh

# 4) Recompute every metric (full + strict protocol)
DATA=${LAYA_DATA:-$LAYA_ROOT/laya}
for f in finfe_eval finnsp_eval cfsc_absa_test; do
  python ../common/laya/laya_metrics.py reports/eval/ft_${f}.json ${DATA}/${f}.jsonl
  python ../common/laya/laya_metrics.py reports/eval/ft_${f}.json ${DATA}/${f}.jsonl --strict
done

# 5) Overlap audit (optional; conclusions are in REPORT.md §2)
python ../common/laya/check_leak.py
```

## Two pitfalls you must know

1. **`laya-evals run --model <local dir>` always fails.** That flag only accepts registry aliases (`english` / `multilingual` / `ml` / `laya` / `default` / `typed-decisions`); passing a path gets rejected by `router.resolve()`. A local ckpt can only be registered through the official `Router.register(name, source)`, which `run_eval.py` wraps.
2. **`EvalReport` is not a dict.** Serialize with `report.to_json()` / `report.to_markdown()`; there is no `to_dict()`.

## Quick numbers

| Task | n (full / strict) | acc | mF1 | ECE | strict acc |
|---|---|---:|---:|---:|---:|
| FinFE `sentiment_social` | 2020 / 923 | 77.97% | 0.7690 | 0.0739 | 77.03% |
| FinNSP `neg_has` | 500 / 499 | 95.00% | 0.9499 | 0.0506 | 94.99% |
| FinNSP `neg_subject` | 259 / 259 | 96.91% | 0.5419¹ | 0.1984 | 96.91% |
| CFSC `sentiment_absa` | 8312 / 7870 | 93.37% | 0.8993 | 0.0638 | 93.58% |

¹ 7 of the 21 classes have zero support and are forced to 0; over the 14 classes with support the figure is **0.8129**.

## Environment variables

| Variable | Default |
|---|---|
| `LAYA_ROOT` | `/data/laya-fin` |
| `LAYA_DATA` | `${LAYA_ROOT}/laya` |
| `LAYA_CKPT` | `${LAYA_ROOT}/ckpt` |
| `LAYA_REPORTS` | `${LAYA_ROOT}/reports` |
| `LAYA_ENV` | `/data/miniconda/envs/laya` |
| `HF_ENDPOINT` | `https://hf-mirror.com` |
| `EVAL_BATCH` | `16` |

## What is not stored here

- **The trained weights (644 MB) are not in this repository**; only `configs/` and `tokenizer/` are kept. Retrain by repeating step 2 above.
- The `/data/laya-fin/...` paths that appear in logs and reports are the real paths on the server at the time, left as-is so they line up line by line with `logs/`.
