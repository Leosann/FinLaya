# FinBertLaya — Laya with FinBERT as the encoder

Swapping Laya's mmBERT encoder (322M) for FinBERT (102M), **with no changes to the laya source**, yields a 117.2M decision model. Full analysis in **[../docs/FINBERT_AS_ENCODER.md](../docs/FINBERT_AS_ENCODER.md)**.

| | |
|---|---|
| Encoder | `yiyanghkust/finbert-tone-chinese` (BertModel 102.27M) |
| Decision head | Laya native (14.97M, **identical** to the mmBERT version) |
| Total | **117.24M** (mmBERT version 321.9M, −63.6%) |
| Training | 4 epochs / 3,068 steps / **23 minutes** / ~3.8 GB GPU memory |
| Key result | FinFE strict protocol **80.28%** (mmBERT version 77.03%, **+3.25 pt**) |
| Deployment | 223.6 MB / 8.66 ms p50 / 113 QPS / 667 MB GPU memory |

## Layout

```
FinBertLaya/
├─ scripts/
│  ├─ make_fb_base.py    [core] build the FinBERT-Laya base checkpoint
│  ├─ train.sh           RLCD training (hyperparameters identical to the mmBERT version)
│  └─ eval.sh            evaluate the three test sets + recompute metrics
├─ configs/
│  ├─ fb_rl_agent_config.json   inference config produced by training (temperatures / abstention thresholds)
│  └─ fb_train_report.json      before/after + temperature fitting
├─ logs/
│  ├─ fb_train.log       training log (including the before/after table)
│  └─ fb_eval.log        output from evaluating the three test sets
└─ reports/
   ├─ fb_*_eval.json               raw laya-evals reports
   └─ fb_*_metrics{,_strict}.json  recomputed metrics (full / strict)
```

## Why no source changes are needed

Three pieces of source evidence (`laya 0.4.15`):

1. `DecisionModel.__init__`: `d = encoder.config.hidden_size`, `nhead = max(1, d // 64)` — the decision head is built from the encoder
2. `build_model`: uses a generic `AutoModel.from_pretrained(cfg["encoder"])`, and `cfg["encoder"]` accepts any HF ID
3. `laya-train --base` has `dest` set to `model_dir`, accepting any path

The only hard prerequisite is a matching **hidden_size**. mmBERT-base and FinBERT are both **768**, so the decision head keeps its shape (14.97M, exactly the same).

## Running

```bash
export LAYA_ROOT=/data/laya-fin
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$LAYA_ROOT/hf_cache"

# 1) Build the base checkpoint: encoder loads FinBERT weights, decision head randomly initialized
python scripts/make_fb_base.py

# 2) Dry run (expect 53341 lines / 55562 questions / skipped {})
laya-train --data "$LAYA_ROOT/laya/all_train.jsonl" --eval "$LAYA_ROOT/laya/finfe_eval.jsonl" \
  --out "$LAYA_ROOT/fb_ckpt" --base "$LAYA_ROOT/fb_base" --dry-run

# 3) Train
bash scripts/train.sh

# 4) Evaluate
bash scripts/eval.sh
bash scripts/eval.sh finfe_eval      # a single dataset

# 5) Deployment metrics (in the laya env)
python ../common/deploy_metrics.py fb_laya
```

## What the base checkpoint changes

`make_fb_base.py` copies the skeleton from the mmBERT version's `rl_agent_config.json` and changes only four things:

| Field | Original | New | Reason |
|---|---|---|---|
| `encoder` | `jhu-clsp/mmBERT-base` | `yiyanghkust/finbert-tone-chinese` | swap the encoder |
| `max_len` | 1024 | **512** | BERT's `max_position_embeddings` ceiling |
| `temperature` | `[1.4378, 1.0, 1.4585]` | `[1.0, 1.0, 1.0]` | the old temperatures were fitted on mmBERT representations and are invalid after the swap |
| `abstention_thresholds` | three entries | removed | same reason; retraining refits them |

The whole `training` field is removed as well, so the mmBERT run's training metadata is not inherited.

## Produced automatically after training

| Item | Value |
|---|---|
| choice temperature | 1.3531 |
| noul temperature | 1.9008 |
| Abstention threshold `choice:3-5` | 0.5968 |
| Abstention threshold `noul:2` | 0.8953 |
| Abstention threshold `choice:11+` | 0.7937 |
| FinFE dev accuracy | 0.4386 → **0.7802** |
| FinFE dev ECE | 0.0967 → **0.0485** |

## Quick results

| Task | Protocol | n | mmBERT-Laya | **FinBERT-Laya** | delta |
|---|---|---:|---:|---:|---:|
| FinFE | full | 2,020 | 77.97% | **78.02%** | +0.05 pt |
| FinFE | strict | 923 | 77.03% | **80.28%** | **+3.25 pt** |
| FinNSP `neg_has` | full | 500 | 95.00% | **95.20%** | +0.20 pt |
| FinNSP `neg_has` | strict | 499 | 94.99% | **95.19%** | +0.20 pt |
| FinNSP `neg_subject` | full | 259 | **96.91%** | 95.75% | −1.16 pt |
| CFSC-ABSA | full | 8,312 | **93.37%** | 93.01% | −0.36 pt |
| CFSC-ABSA | strict | 7,870 | **93.58%** | 93.01% | −0.57 pt |

## Two pitfalls

1. **`laya_metrics.py` writes `_metrics.json` itself.** Do not redirect its stdout to those files — that overwrites the JSON with human-readable text. `eval.sh` discards it to `/dev/null`.
2. **Passing an HF ID to `--base` fails.** `resolve_checkpoint_dir` runs `snapshot_download` for Hub IDs and its `allow_patterns` only covers laya-layout files; the FinBERT repo has no `rl_agent_config.json`, so it errors out immediately. You must build a local base directory first.

## What is not stored here

- The trained weights (223.6 MB) are not in this repository; rerunning the steps above takes 23 minutes.
- `fb_base/` (223.6 MB) is likewise not committed; `make_fb_base.py` rebuilds it in about a minute.
