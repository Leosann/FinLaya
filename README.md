<div align="center">

<img src="assets/logo-wordmark-260.png" alt="FinLaya" height="72" align="middle" />

# FinLaya: Chinese financial sentiment as a decision problem — a third of the parameters, the same accuracy

Sentiment analysis reformulated as decisions over a closed option set: the model emits calibrated probabilities and can abstain by confidence.

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-FFD21E)](https://huggingface.co/zhanghanshuo/FinLaya)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Laya](https://img.shields.io/badge/Laya-0.4.15-6E56CF)](https://github.com/laya-ai/laya)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)

[简体中文](README.zh-CN.md) · [Model weights](https://huggingface.co/zhanghanshuo/FinLaya) · [Key findings](#key-findings) · [Results](#results) · [Quick start](#quick-start) · [Docs](#docs)

</div>

---

## Overview

[Laya](https://github.com/laya-ai/laya) is a **decision model** (System-1). Its defining difference from autoregressive generation is the output form: given a *state* plus a *question with options*, it returns a **probability distribution over the options** instead of free-form text. Three properties follow, and each is measurable:

- **Calibrated by construction** — the output is a softmax distribution, so ECE / Brier / NLL are directly computable, and a threshold can route each case to auto-process or human review.
- **Closed answer space** — no hallucination, no format drift. The metric is accuracy; no parsing step is needed.
- **Single forward pass** — one forward pass per decision, so latency is low and stable.

This project reframes three Chinese financial sentiment datasets (CFSC-ABSA / FinFE / FinNSP) as Laya decision tasks, trains three models, and evaluates all of them under one consistent protocol to answer a concrete question:

> **Laya's default mmBERT encoder is heavy (322M). Can it be swapped for a 102M FinBERT without losing accuracy?**

**Yes — with zero changes to the laya source code.**

### Model names at a glance

`FinLaya` is the **family name**, covering two Laya models that share an identical decision head and differ only in the encoder:

| Name used in this repo | Directory | Encoder | Params | Also known as |
|---|---|---|---:|---|
| **FinLaya-mmBERT** | `FinLaya-mmBERT/` | `jhu-clsp/mmBERT-base` | 321.9 M | the main experiment, the accuracy reference |
| **FinLaya-FinBERT** | `FinLaya-FinBERT/` | `yiyanghkust/finbert-tone-chinese` | 117.2 M | the encoder-swap variant, **the one on Hugging Face** |

Two things that commonly cause confusion:

- The Hugging Face repo is called [`zhanghanshuo/FinLaya`](https://huggingface.co/zhanghanshuo/FinLaya) but holds **FinLaya-FinBERT**, not the mmBERT model. The repo name predates the split into two variants.
- The third model, **FinBERT** (`FinBert/`), is a different architecture entirely — four independent classifiers, no Laya decision head. It is the discriminative baseline, not a FinLaya variant.

## Key findings

### 1. Encoder swapping is native to the framework, not a hack

Three source references in `laya 0.4.15`:

```python
# common.py:582  the decision head is built from the encoder's hidden_size
d = encoder.config.hidden_size
nhead = max(1, d // 64)
layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)

# common.py:694  build_model uses generic AutoModel; cfg["encoder"] accepts any HF ID
enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")

# train_cli.py:45  --base maps to dest=model_dir, which accepts any path
parser.add_argument("--base", "--model", dest="model_dir", ...)
```

The only hard prerequisite is a matching **hidden_size**. mmBERT-base and FinBERT are both **768**, so the decision heads are structurally identical (14.97M parameters, no differences).

### 2. The parameter gap is almost entirely multilingual vocabulary embeddings

| | FinLaya-mmBERT | FinLaya-FinBERT |
|---|---:|---:|
| Vocabulary size | 256,000 | 21,128 |
| **Embedding parameters** | **196.61 M** | **16.23 M** |
| Encoder body | 110.4 M (22 layers) | 85.0 M (12 layers) |
| Decision head | 14.97 M | **14.97 M (identical)** |
| **Total** | **321.9 M** | **117.2 M** |

Chinese financial tasks do not need 256k-scale multilingual coverage; that embedding capacity is structural redundancy.

### 3. Accuracy does not degrade — and FinFE under the strict protocol improves

| Task | Protocol | FinLaya-mmBERT (322M) | **FinLaya-FinBERT (117M)** | FinBERT classifier (102M) |
|---|---|---:|---:|---:|
| FinFE social sentiment | full | 77.97 | 78.02 | **79.26** |
| FinFE social sentiment | strict | 77.03 | **80.28** | 79.52 |
| FinNSP negative presence | full | 95.00 | 95.20 | **95.60** |
| FinNSP negative subject | full | **96.91** | 95.75 | 95.37 |
| CFSC-ABSA aspect-level | full | **93.37** | 93.01 | 92.78 |

The **+3.25 pt gain for FinFE under the strict protocol** is the largest difference across all comparisons. Attribution: 54.3% of the FinFE evaluation set duplicates training text verbatim. FinLaya-mmBERT is fine-tuned from the official laya checkpoint and thus has a memorization advantage on seen samples. Once duplicates are removed that advantage disappears, while FinLaya-FinBERT — whose decision head is randomly initialized and learned from scratch — generalizes better.

### 4. Deployment-side gains

| | FinLaya-mmBERT | **FinLaya-FinBERT** | Change |
|---|---:|---:|---:|
| Parameters | 321.9 M | **117.2 M** | **−63.6%** |
| Weight size | 614.0 MB | **223.6 MB** | **−63.6%** |
| Latency p50 | 18.64 ms | **8.66 ms** | **−53.5%** |
| Throughput (batch=1) | 53.0 QPS | **113.0 QPS** | **×2.13** |
| Peak GPU memory | 1499 MB | **667 MB** | **−55.5%** |

## The three models

| Model | Directory | Encoder | Decision head | Params | Role |
|---|---|---|---:|---:|---|
| **FinLaya-mmBERT** | `FinLaya-mmBERT/` | mmBERT 322M | Laya native 15M | 321.9 M | Main experiment, accuracy reference; the only one whose architecture supports long context (trained at 1024 tokens, ceiling 8192) |
| **FinLaya-FinBERT** | `FinLaya-FinBERT/` | FinBERT 102M | Laya native 15M | **117.2 M** | The encoder-swap variant: one model covers all 4 question types at half the latency |
| **FinBERT** | `FinBert/` | FinBERT 102M | 4 separate classifiers | 102.3 M | Discriminative baseline; fastest and most accurate per single task, but requires 4 models |

> **Naming.** `FinLaya` on its own is the family name, not a model. The two Laya models differ only in the encoder — check that column when reading results. Note also that the Hugging Face repo [`zhanghanshuo/FinLaya`](https://huggingface.co/zhanghanshuo/FinLaya) holds **FinLaya-FinBERT** (117.2 M); the repo name predates the split into two variants.

## Results

### Accuracy and Macro-F1 (full / strict)

Strict protocol = removing samples whose text SHA256 matches the training set exactly (FinFE removes 1,097/2,020, CFSC 442/8,312, FinNSP 1/500).

| Task | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---|---|
| FinFE accuracy | 77.97 / 77.03 | 78.02 / **80.28** | **79.26** / 79.52 |
| FinFE Macro-F1 | **0.7690** / 0.7094 | 0.7629 / **0.7371** | 0.7776 / 0.7258 |
| FinNSP negative presence accuracy | 95.00 / 94.99 | 95.20 / 95.19 | **95.60** / **95.59** |
| FinNSP negative presence Macro-F1 | 0.9499 / 0.9498 | 0.9520 / 0.9518 | **0.9559** / **0.9558** |
| FinNSP negative subject accuracy | **96.91** | 95.75 | 95.37 |
| FinNSP negative subject Macro-F1 (14 classes present) | **0.8129** | 0.7003 | 0.6838 |
| CFSC-ABSA accuracy | **93.37** / **93.58** | 93.01 / 93.01 | 92.78 / 92.94 |
| CFSC-ABSA Macro-F1 | **0.8993** / **0.9025** | 0.8909 / 0.8905 | 0.8881 / 0.8902 |

### Probability calibration (lower is better)

| Task | Metric | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | ECE | 0.0739 | **0.0485** | 0.0947 |
| FinFE | NLL | 0.6467 | **0.5878** | 0.6244 |
| FinFE | Brier | 0.3516 | **0.3259** | 0.3285 |
| FinNSP negative presence | ECE | 0.0506 | 0.0882 | **0.0311** |
| FinNSP negative subject | ECE | 0.1984 | 0.1546 | **0.0616** |
| CFSC-ABSA | ECE | 0.0638 | 0.0607 | **0.0339** |
| CFSC-ABSA | NLL | 0.2658 | 0.2424 | **0.2157** |

### Selective accuracy (top X% by confidence auto-processed, full protocol)

| Task | Coverage | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | @50% | 89.50 | **91.39** | 90.99 |
| FinFE | @80% | 83.79 | **85.09** | 85.02 |
| FinFE | @90% | 81.35 | 81.96 | **82.40** |
| FinNSP negative presence | @50% | 99.60 | 99.60 | **100.0** |
| FinNSP negative subject | @50% | 99.23 | 98.46 | **99.23** |
| CFSC-ABSA | @50% | 98.65 | 99.47 | **99.83** |

Under the strict protocol the FinFE @50 gap is wider: **mmBERT 92.21 / FinLaya-FinBERT 95.02 / FinBERT 94.81**.

> The full version (per-class recall, AURC, mean confidence, training cost, model-selection decision tree) is in [`docs/THREE_WAY_COMPARISON.md`](docs/THREE_WAY_COMPARISON.md).

## Quick start

### Installation

The two environments must stay isolated — `laya` and `finbert` have different dependency ranges.

```bash
# Laya environment
conda create -y -p /data/miniconda/envs/laya python=3.11
PY=/data/miniconda/envs/laya/bin/python
$PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install "transformers>=4.48,<5" laya==0.4.15

# FinBERT environment
bash FinBert/scripts/setup_finbert.sh
```

### Data preparation

```bash
python datasets/to_laya_format.py       # raw data → Laya JSONL
python datasets/validate_laya_format.py # verify qid definition consistency (exit 0 = pass)
```

### Train FinLaya (mmBERT encoder)

```bash
bash FinLaya-mmBERT/scripts/train.sh dryrun   # dry run first: confirm 53,341 lines / 55,562 questions parse
bash FinLaya-mmBERT/scripts/train.sh train    # 4 epochs
```

### Train FinLaya-FinBERT (FinBERT encoder)

```bash
# 1) build the base checkpoint: FinBERT encoder + randomly initialized decision head (~1 min)
python FinLaya-FinBERT/scripts/make_fb_base.py

# 2) train (all hyperparameters identical to the mmBERT version)
bash FinLaya-FinBERT/scripts/train.sh
```

### Train FinBert (classifier baseline)

```bash
ENV_PY=/data/miniconda/envs/finbert/bin/python
$ENV_PY FinBert/scripts/finbert_baseline.py train
```

### Evaluation

```bash
# FinLaya
bash FinLaya-mmBERT/scripts/eval_ft.sh
bash FinLaya-mmBERT/scripts/eval_ft.sh finfe_eval              # single dataset

# FinLaya-FinBERT
bash FinLaya-FinBERT/scripts/eval.sh

# FinBert (must be batched; running all at once gets killed by the OOM killer with no Traceback)
$ENV_PY FinBert/scripts/fb_eval.py absa
$ENV_PY FinBert/scripts/fb_eval.py finfe
```

Recompute the full metric set (Macro-F1 / per-class recall / NLL / Brier / selective accuracy / strict protocol):

```bash
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl --strict
```

> **Note**: `laya_metrics.py` writes `<report>_metrics.json` itself — do **not** redirect its stdout to a file of the same name, or the freshly written JSON gets overwritten with human-readable text.

### Deployment metrics

The three models must be measured in separate processes (they depend on different envs' torch builds):

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
```

### Tests

The metric and data-conversion layers are plain Python + numpy, so they run without a GPU, model
weights or the raw datasets:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

`tests/` covers the sentiment-metric implementations (`common/laya/laya_metrics.py` and the copies in
`FinBert/scripts/finbert_baseline.py`, cross-checked against each other), the Laya JSONL validator,
the three dataset converters and the strict-protocol leak checks. It also **re-derives every
published number** in `FinLaya-mmBERT/reports/eval/ft_*_metrics.json` from the committed evaluation
reports, so a regression in the metric code fails the suite instead of silently changing the tables
above. GitHub Actions runs it on Python 3.11 and 3.12 for every push to `main` and every PR.

> `AURC` is the one metric the suite checks with a `1e-3` tolerance rather than exactly: the report
> probabilities are rounded to 4 decimals, so confidence ties are common (1,051 of FinFE's 2,020
> cases) and the area under the risk-coverage curve depends on how ties are ordered. `selective_acc@k`
> is computed with a stable sort and does reproduce the published values exactly.

## Model download

### Hugging Face (recommended)

Full weights are published at [**huggingface.co/zhanghanshuo/FinLaya**](https://huggingface.co/zhanghanshuo/FinLaya).

```bash
pip install -U huggingface_hub
huggingface-cli download zhanghanshuo/FinLaya --local-dir weights/finlaya-finbert-hf
```

Behind the GFW, use the mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download zhanghanshuo/FinLaya --local-dir weights/finlaya-finbert-hf
```

The downloaded directory already follows the laya checkpoint layout, so `Router.register(name, path)` works directly.

> **If the download comes back empty**, the files are currently nested under an extra `hf-ready/` prefix in the Hub repo — that upload dropped the `path_in_repo` argument. Until it is fixed, fetch them explicitly:
>
> ```bash
> huggingface-cli download zhanghanshuo/FinLaya --include "hf-ready/*" --local-dir weights/finlaya-finbert-hf
> mv weights/finlaya-finbert-hf/hf-ready/* weights/finlaya-finbert-hf/ && rmdir weights/finlaya-finbert-hf/hf-ready
> ```
>
> See [`weights/HF_UPLOAD_GUIDE.md`](weights/HF_UPLOAD_GUIDE.md) for the one-step fix.

### In this repository

**FinLaya-FinBERT (117.2M)** is also shipped in-repo, split into 3 parts (GitHub's 100 MB per-file limit):

```bash
cd weights/finlaya-finbert
bash restore.sh        # merge + SHA256 verification
```

The merged layout is identical to the HF release. See [`weights/finlaya-finbert/README.md`](weights/finlaya-finbert/README.md).

**FinBERT's four classifiers** (391 MB fp32 each, 1.56 GB total) are not committed — retraining is cheap and the commands are above.

## Datasets

All three datasets come from **[CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)** (Chinese Financial Language Evaluation Benchmark).

| Dataset | Task | Raw files | Size | Rows |
|---|---|---|---:|---:|
| **CFSC-ABSA** | Aspect-based sentiment analysis | `train.txt` / `test.txt` | 14.90 + 3.72 MB | 33,184 / 8,312 |
| **FinFE** | Financial social-media sentiment | `train_list.json` / `eval_list.json` / `problem_list.json` | 1.09 + 0.14 + 0.17 MB | 16,157 / 2,020 / 2,020 |
| **FinNSP** | Financial negative subject identification | `train_list.json` / `eval_list.json` / `problem_list.json` | 3.48 + 0.46 + 0.46 MB | 4,000 / 500 / 500 |

**24.42 MB / 8 files** in total, each SHA256-verified (manifest in [`datasets/README.md`](datasets/README.md)). The data itself is not committed; regenerate it with `datasets/to_laya_format.py`.

> `problem_list.json` is unlabeled test input and is not used in this project's evaluation; evaluation uses `eval_list.json` throughout.
>
> The **54.3% overlap in FinFE** comes from the original dataset split, not from our conversion. Every result is therefore reported under both the full and the strict protocol.

## Processed samples

The conversion produces 9 JSONL files (120,034 lines / 125,235 questions). The three task shapes:

**FinFE — 3-way `choice`**

```json
{"state": "试仓500股，看看门道！",
 "questions": {"sentiment_social": {
   "type": "choice",
   "instructions": "这段股吧/雪球社交文本对股市或个股的情绪倾向是什么？",
   "criteria": {
     "negative": "看空、恐慌、抱怨、亏损、割肉、骂声等消极情绪",
     "neutral":  "中性陈述、疑问、无明确多空倾向",
     "positive": "看多、乐观、期待上涨等积极情绪"}}},
 "expected": {"sentiment_social": "positive"},
 "meta": {"source": "FinFE"}}
```

**CFSC-ABSA — the aspect term lives inside `state`**

```json
{"state": "方面词：长谷投资\n语句：问董秘：请问长谷投资是否是你旗下的公司？【长谷投资】",
 "questions": {"sentiment_absa": {
   "type": "choice",
   "instructions": "句中针对指定方面词的情感倾向是什么？",
   "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}},
 "expected": {"sentiment_absa": "neutral"},
 "meta": {"source": "CFSC-ABSA"}}
```

**FinNSP — two questions per line, `choice` + `noul`**

```json
{"state": "候选实体：恒昌汇财；纳觅财务\n文本：【网传恒昌汇财被查，投资者恐慌】...",
 "questions": {
   "neg_has":     {"type": "noul", "instructions": "文本中是否提及了负面主体？"},
   "neg_subject": {"type": "choice", "instructions": "文本中的负面主体是哪个？",
                   "criteria": {"以上都不是": "...", "恒昌汇财": "...", "...": "..."}}},
 "expected": {"neg_has": true, "neg_subject": "以上都不是"},
 "meta": {"source": "FinNSP"}}
```

> **Critical constraint**: the `questions` definition for a given `qid` must be byte-identical across all lines. `laya/train.py` only `warnings.warn`s on conflict and keeps the first definition, silently dropping later ones — this is the easiest trap to fall into. `validate_laya_format.py` exists to enforce it.

## Project structure

```
FinLaya-mmBERT/       mmBERT encoder + Laya decision head (322M) — main experiment
  ├─ scripts/     train.sh · baseline.sh · eval_ft.sh
  ├─ configs/     rl_agent_config.json · questions.json · train_report.json
  ├─ logs/        train_run · dryrun · baseline · eval_ft
  └─ reports/     eval/ (ft_*) · baseline/ (base_*)
FinLaya-FinBERT/      FinBERT encoder + Laya decision head (117M) — the published weights
  ├─ scripts/     make_fb_base.py · train.sh · eval.sh
  ├─ configs/     fb_rl_agent_config.json · fb_train_report.json
  └─ reports/     fb_*
FinBert/              FinBERT encoder + 4 task classifiers (102M) — discriminative baseline
  ├─ scripts/     setup_finbert.sh · finbert_baseline.py · fb_eval.py
  └─ reports/     finbert_all.json
common/           shared across the three models
  ├─ laya/        run_eval.py · laya_metrics.py · check_leak.py · strict_acc.py
  ├─ deploy_metrics.py · to_fp16.py · make_compare.py
  └─ reports/     deploy_metrics.json
datasets/         download notes + conversion / validation scripts
weights/          trained FinLaya-FinBERT weights (shards + merge script)
docs/             three comparison reports
```

## Docs

| Document | Contents |
|---|---|
| [`FinLaya-mmBERT/REPORT.md`](FinLaya-mmBERT/REPORT.md) | Single-model main experiment: conclusions, data, training, pitfalls, quick numbers |
| [`docs/COMPARE_Laya_vs_FinBERT.md`](docs/COMPARE_Laya_vs_FinBERT.md) | Two-way comparison: Laya vs FinBERT classifiers |
| [`docs/FINBERT_AS_ENCODER.md`](docs/FINBERT_AS_ENCODER.md) | Encoder-swap deep dive: source evidence, method, three-way results, limitations |
| [`docs/THREE_WAY_COMPARISON.md`](docs/THREE_WAY_COMPARISON.md) | Three-model full-metric comparison: per-class recall, AURC, training cost, selection tree |

## Known limitations

1. **The FinFE full protocol has 54.3% training-set overlap.** The 77.97 / 78.02 / 79.26 figures should not be cited as generalization ability; use the strict protocol.
2. **The FinNSP negative-subject task is severely long-tailed.** Of 259 cases, 102 are "以上都不是" and the remaining 157 spread over 13 entities, most with only 1–7 cases. Nearly all of the `0.8129 / 0.7003 / 0.6838` gap comes from 钱宝 (4 cases) and 摩尔龙 (1 case).
3. **The `noul` temperature was fitted on 34 samples** (`train_report.json` carries a warning), so `neg_has` ECE is less trustworthy than other tasks.
4. **FinLaya-FinBERT truncates at 512 tokens.** 5.00% of FinNSP training samples exceed that length (p99=1,851, max=6,777), so tail information is lost. This is an architectural gap between BERT and mmBERT (8192).
5. **The two Laya models start from unequal points.** FinLaya-FinBERT's decision head is randomly initialized (loss 1.3957); FinLaya-mmBERT continues from the official checkpoint (0.3983). Compare endpoints only, not convergence speed.
6. **Single seed.** Differences within 1 pt are not significant; the 3.25 pt FinFE strict-protocol gap is larger but still worth re-checking with multiple seeds.

## Citation

```bibtex
@misc{cflab2022,
  title  = {CFLEB: Chinese Financial Language Evaluation Benchmark},
  author = {CFLab, Sun Yat-sen University},
  year   = {2022},
  url    = {https://github.com/ssymmetry/BBT-FinCUGE-Applications}
}

@misc{laya2024,
  title = {Laya: A Decision Model for Typed Decisions},
  url   = {https://github.com/laya-ai/laya}
}

@inproceedings{yang2020finbert,
  title     = {FinBERT: A Pretrained Language Model for Financial Communications},
  author    = {Yang, Yi and Uy, Mark Christopher Siy and Huang, Allen},
  booktitle = {arXiv:2006.08097},
  year      = {2020}
}
```

## License

Code is MIT. Datasets follow the original CFLEB terms — see [`datasets/README.md`](datasets/README.md).
