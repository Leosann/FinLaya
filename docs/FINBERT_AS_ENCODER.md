# Swapping FinBERT in as Laya's encoder

> **In one sentence**: it works, and it needs **no changes to the laya source**. Afterwards the model has 63.6% fewer parameters, 53.5% lower latency, and 55.5% less GPU memory, with accuracy roughly on par across three tasks; on FinFE under the strict protocol it is **3.25 points higher**.

---

## 1. Why this is allowed

This is not a hack — laya already leaves the opening. Three pieces of source evidence (`laya 0.4.15`):

**① The decision head is built from the encoder's hidden_size** (`common.py:582`)

```python
class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1, no_init=False):
        d = encoder.config.hidden_size          # not hardcoded
        nhead = max(1, d // 64)                 # not hardcoded
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, ...)
        self.head = nn.TransformerEncoder(layer, head_layers, ...)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), ...)
```

**② `build_model` uses a generic `AutoModel`, and `cfg["encoder"]` accepts any HF ID** (`common.py:694`)

```python
def build_model(cfg, encoder_dir=None, pretrained=True, revision=None):
    from transformers import AutoConfig, AutoModel
    ...
    enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, head_layers, n_act)
```

**③ `--base` maps to `dest="model_dir"`, accepting any path or model ID** (`train_cli.py:45`)

```python
parser.add_argument("--base", "--model", dest="model_dir",
                    default="convaiinnovations/laya",
                    help="Base checkpoint path or model ID")
```

So the only prerequisite is a matching **hidden_size**, which keeps the decision head's shape unchanged. Here the two agree:

| | mmBERT-base (original) | FinBERT (proposed) | |
|---|---|---|---|
| model_type | modernbert | bert | — |
| **hidden_size** | **768** | **768** | ✅ match |
| num_hidden_layers | 22 | 12 | — |
| num_attention_heads | 12 | 12 | — |
| intermediate_size | 1152 (GLU, 2304 effective) | 3072 | — |
| vocab_size | 256,000 | 21,128 | — |
| max_position_embeddings | 8192 | 512 | ⚠️ requires lowering max_len |

Two supporting checks:

- `_apply_rope_config(ecfg)` only takes effect when `rope_parameters` exists. A BERT config has no such key, so it **returns immediately with no side effects**.
- `forward` uses `self.encoder(...).last_hidden_state`, which `BertModel` fully supports. The `position_ids` branch only runs when `option_ids is not None` (parallel_layout); this project sets `option_layout: null`, so it never triggers.

### The key number: the savings are entirely the multilingual vocabulary

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| Encoder | of 322M, **196.61M is embedding** | 102.27M (embedding only 16.23M) |
| Decision head | 14.97M | **14.97M (identical)** |
| Total | 321.9M | 117.2M |

The decision heads are exactly the same; the whole difference sits in the encoder. And mmBERT's 196.6M embedding is what you pay for a 256k multilingual vocabulary — **a Chinese financial task never uses it**.

---

## 2. How it was done (three steps, no source changes)

### Step 1: build a FinBERT-Laya base checkpoint

`laya-train --base` requires the directory to be a complete laya layout (`rl_agent_config.json` + `model.safetensors` + `encoder/` + `tokenizer/`). Passing `yiyanghkust/finbert-tone-chinese` directly does not work: `resolve_checkpoint_dir` goes through `snapshot_download`'s `allow_patterns` and fails when it cannot find `rl_agent_config.json`.

So `scripts/make_fb_base.py` does four things:

```python
cfg = json.load(open(SRC + "/rl_agent_config.json"))   # borrow the mmBERT version's config skeleton
cfg["encoder"] = "yiyanghkust/finbert-tone-chinese"
cfg["max_len"] = 512                                   # BERT's positional-encoding ceiling
cfg["temperature"] = [1.0, 1.0, 1.0]                   # the old temperatures were fitted on mmBERT representations; invalidate them
cfg.pop("abstention_thresholds", None)
cfg.pop("training", None)

model = build_model(cfg, pretrained=True)   # encoder loads FinBERT weights, decision head randomly initialized
save_checkpoint(model, tok, cfg, OUT)       # write the four pieces laya.load expects
```

Output `fb_base/`:

```
model.safetensors        223.64 MB   (fp16, 235 tensors)
rl_agent_config.json      encoder=yiyanghkust/finbert-tone-chinese, max_len=512
encoder/config.json       BertConfig
tokenizer/                BertTokenizerFast, vocab.txt + tokenizer.json
```

`load_checkpoint(OUT)` reads it back with `strict=True` and passes. That is the critical validation: the weight keys line up exactly with what laya expects.

### Step 2: train with the original recipe

Hyperparameters are **identical item by item** to the mmBERT version; the only difference is `--base`:

```bash
laya-train --data all_train.jsonl --eval finfe_eval.jsonl \
  --out fb_ckpt --base "$LAYA_ROOT/fb_base" \
  --epochs 4 --micro-batch 24 --grad-accum 3 --device cuda \
  --shuffle-options --label-smoothing 0.05
```

Verify with a dry run first: `53,341 lines / 55,562 questions, skipped {}` — **exactly the same as the mmBERT version** — and the optimizer budget is likewise 3,068 steps.

### Step 3: evaluate under the original protocol

`run_eval.py` (registering the local directory via `Router.register`) followed by recomputation with `laya_metrics.py`, covering both the full and the strict protocol.

---

## 3. Results

### 3.1 Accuracy: three tasks on par, FinFE strict protocol ahead

| Task | Protocol | n | mmBERT-Laya (322M) | **FinBERT-Laya (117M)** | FinBERT classifier (102M) |
|---|---|---:|---:|---:|---:|
| FinFE social sentiment | full | 2,020 | 77.97% | **78.02%** | **79.26%** |
| FinFE social sentiment | strict | 923 | 77.03% | **80.28%** ⭐ | 79.52% |
| FinNSP negative presence | full | 500 | 95.00% | **95.20%** | **95.60%** |
| FinNSP negative presence | strict | 499 | 94.99% | **95.19%** | 95.59% |
| FinNSP negative subject | full | 259 | **96.91%** | 95.75% | 95.37% |
| CFSC-ABSA | full | 8,312 | **93.37%** | 93.01% | 92.78% |
| CFSC-ABSA | strict | 7,870 | **93.58%** | 93.01% | 92.94% |

**The row worth discussing**: on FinFE under the strict protocol, FinBERT-Laya reaches **80.28%** against mmBERT-Laya's 77.03% — **+3.25 pt**.

Why? 54.3% (1,097/2,020) of the FinFE evaluation set is textually identical to the training set. mmBERT-Laya starts from the official laya checkpoint, which has already seen a great deal of public financial text, so it holds a memorization advantage on those "seen" samples; remove the overlap and that advantage disappears. FinBERT-Laya's decision head, by contrast, is **randomly initialized and learned from scratch**, with no such memory, and generalizes better.

### 3.2 Macro-F1

| Task | Protocol | mmBERT-Laya | FinBERT-Laya | FinBERT classifier |
|---|---|---:|---:|---:|
| FinFE | full | **0.7690** | 0.7629 | 0.7776 |
| FinFE | strict | 0.7094 | **0.7371** | 0.7258 |
| FinNSP `neg_has` | full | 0.9499 | **0.9520** | 0.9559 |
| FinNSP `neg_subject` | full | 0.5419 | 0.4669 | 0.5038 |
| FinNSP `neg_subject` (14 classes with support) | full | **0.8129** | 0.7003 | 0.6838 |
| CFSC-ABSA | full | **0.8993** | 0.8909 | 0.8881 |
| CFSC-ABSA | strict | **0.9025** | 0.8905 | 0.8902 |

`neg_subject` is the only clear regression: restricted to classes with support, mF1 drops from 0.8129 to 0.7003. The cause is the long tail — of 259 samples, 102 are `以上都不是` and the other 157 spread over 13 entities, most with only 1–7 samples. Per class:

| Entity | Support | mmBERT-Laya recall | FinBERT-Laya recall |
|---|---:|---:|---:|
| 摩尔龙 | 1 | 1.000 | **0.000** |
| 钱宝 | 4 | 0.750 | **0.250** |
| 钱宝网 | 19 | **1.000** | 0.947 |
| 宜贷网 | 26 | 0.962 | **1.000** |
| 以上都不是 | 102 | 0.961 | 0.961 |

The gap comes almost entirely from two extremely small classes: 摩尔龙 (1 sample) and 钱宝 (4 samples). **This kind of fluctuation should not be read as a capability difference.**

### 3.3 Probability calibration

| Task | Protocol | mmBERT ECE | FinBERT-Laya ECE | FinBERT classifier ECE |
|---|---|---:|---:|---:|
| FinFE | full | 0.0739 | **0.0485** | 0.0947 |
| FinFE | strict | **0.0628** | 0.0710 | 0.0876 |
| FinNSP `neg_has` | full | **0.0506** | 0.0882 | 0.0311 |
| FinNSP `neg_subject` | full | 0.1984 | **0.1546** | 0.0616 |
| CFSC-ABSA | full | 0.0638 | **0.0607** | 0.0339 |

FinBERT-Laya is generally better on NLL / Brier:

| Task | NLL (mm / fb-laya / fb-cls) | Brier (mm / fb-laya / fb-cls) |
|---|---|---|
| FinFE full | 0.6467 / **0.5878** / 0.6244 | 0.3516 / **0.3259** / 0.3285 |
| FinFE strict | 0.6233 / **0.5328** / 0.5713 | 0.3437 / **0.2916** / 0.3066 |
| FinNSP `neg_has` | **0.1799** / 0.1851 / 0.1517 | 0.0873 / 0.0815 / **0.0740** |
| FinNSP `neg_subject` | 0.3514 / 0.3113 / **0.2821** | 0.0952 / **0.0850** / 0.0991 |
| CFSC-ABSA full | 0.2658 / 0.2424 / **0.2157** | 0.1170 / **0.1119** / 0.1126 |

The `neg_has` ECE of 0.0882 is on the high side, but note that this task's noul temperature was fitted on **only 34** calibration samples (with a warning in `train_report.json`), so it is inherently noisy.

### 3.4 Selective accuracy

Taking the top X% by confidence and auto-processing:

| Task | @50 (mm / fb-laya / fb-cls) | @80 | @90 |
|---|---|---|---|
| FinFE | 89.50 / **91.39** / 90.99 | 83.79 / **85.09** / 85.02 | 81.35 / **81.96** / 82.40 |
| FinNSP `neg_has` | **99.60** / **99.60** / 100.0 | **99.00** / 98.75 / 99.75 | **98.22** / 98.00 / 98.00 |
| FinNSP `neg_subject` | 99.23 / 98.46 / **99.23** | **99.52** / 97.58 / 98.07 | **99.14** / 97.85 / 97.42 |
| CFSC-ABSA | 98.65 / **99.47** / 99.83 | 98.11 / **98.18** / 98.32 | 96.58 / **96.68** / 96.38 |

On FinFE, FinBERT-Laya beats the mmBERT version at all three coverage levels, consistent with its strict-protocol accuracy advantage.

### 3.5 Per-class recall

| Task | Class | Support | mmBERT-Laya | FinBERT-Laya | FinBERT classifier |
|---|---|---:|---:|---:|---:|
| FinFE | negative | 574 | 0.780 | 0.791 | **0.819** |
| FinFE | neutral | 560 | **0.696** | 0.611 | 0.639 |
| FinFE | positive | 886 | 0.832 | **0.880** | 0.872 |
| CFSC | negative | 2,146 | **0.948** | 0.944 | 0.943 |
| CFSC | neutral | 1,032 | **0.797** | 0.751 | 0.749 |
| CFSC | positive | 5,134 | 0.955 | **0.960** | 0.958 |
| FinNSP `neg_has` | true | 259 | **0.961** | 0.950 | 0.958 |
| FinNSP `neg_has` | false | 241 | 0.938 | **0.954** | 0.954 |

**`neutral` remains the weak spot across the board** (0.61–0.70). FinBERT-Laya's neutral recall is lower than the mmBERT version's (0.611 vs. 0.696), but its positive is 4.8 pt higher, leaving overall F1 only 0.006 apart. Neutral boundaries hinge on subtle semantic distinctions, and swapping the encoder genuinely changes how such samples are judged.

---

## 4. Deployment value: the main payoff of the swap

| | mmBERT-Laya | **FinBERT-Laya** | Change | FinBERT classifier |
|---|---:|---:|---:|---:|
| Parameters | 321.9M | **117.2M** | **−63.6%** | 102.3M |
| Weight size (F16) | 614.0 MB | **223.6 MB** | **−63.6%** | 390 MB (fp32) / 195 MB (F16) |
| Latency p50 | 18.64 ms | **8.66 ms** | **−53.5%** | 5.93 ms |
| Latency p95 | 19.34 ms | **8.92 ms** | **−53.9%** | 5.98 ms |
| Throughput batch=1 | 53.0 QPS | **113.0 QPS** | **×2.13** | 145.3 QPS |
| Peak GPU memory | 1499 MB | **667 MB** | **−55.5%** | 486 MB |

> On the size figure: this counts only the weight file and excludes `checkpoint_latest/` (the mid-training snapshot holding the same weights as the top-level `model.safetensors`). Earlier figures of 1228 MB / 447 MB included that snapshot and were wrong; they have been corrected.

**FinBERT-Laya delivers 2.13× the throughput at 1/2.75 the parameters, and is only 1.46× slower than the FinBERT classifier** — while matching all four question types takes four classifier models (about 780 MB), versus one 223.6 MB FinBERT-Laya.

### Training cost

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| Steps | 3,068 | 3,068 (identical) |
| Wall time | 32 min | **23 min** (−28%) |
| GPU memory (sampled) | 9.2 GB | **~3.8 GB** (−59%) |
| Final-epoch loss | 0.2018 | 0.3047 (different starting points; not directly comparable) |
| FinFE dev accuracy | 0.4386 → 0.7807 | 0.4386 → 0.7802 |

**Starting losses are not comparable**: the mmBERT version starts from the official laya checkpoint (loss 0.3983), while the FinBERT version's decision head is randomly initialized (loss 1.3957). So compare endpoints here, not the size of the drop.

---

## 5. Conclusions

| Question | Answer |
|---|---|
| Can FinBERT be swapped in as Laya's encoder? | **Yes, with no changes to the laya source.** The only prerequisite is a matching hidden_size (768) |
| Does quality drop? | **Essentially not.** FinFE strict +3.25 pt, neg_has +0.20 pt, CFSC −0.36 pt, neg_subject −1.16 pt (long-tail noise) |
| Is it worth doing? | **Yes.** Parameters −63.6%, latency −53.5%, GPU memory −55.5% at unchanged accuracy |
| When should you pick it? | When you want Laya's "one weight set, many tasks, confidence-based abstention" but find 322M too heavy |
| When should you not? | Chasing maximum single-task accuracy and unbothered by multiple models → still use the FinBERT classifier; long-tail entity recognition (`neg_subject`) → use mmBERT-Laya |

### Recommended setup

```
FinBERT-Laya (117M / 223.6 MB / 8.66 ms)
  ├─ Fits: multiple tasks coexisting, abstention thresholds needed, tight GPU memory, latency-sensitive
  └─ When to replace it: single-task accuracy insufficient → FinBERT classifier
                         long-tail entity recall insufficient → mmBERT-Laya
```

---

## 6. Limitations and caveats

1. **512-token truncation.** FinBERT's positional-encoding ceiling is 512, while 5.00% of FinNSP training samples exceed it (p99 = 1,851, max = 6,777), so tail information in those long texts is lost. CFSC/FinFE are unaffected (p99 of 436 / 60 respectively). Fully solving this would require a sliding window or long-text chunking — but BERT simply lacks mmBERT's 8192 context, which is an architectural trade-off.
2. **Unequal starting points.** The mmBERT version continues from the official laya checkpoint; the FinBERT version's decision head is randomly initialized. This is unavoidable when swapping encoders — there is no corresponding laya checkpoint for FinBERT. So **compare endpoints only, never convergence speed**.
3. **Single seed.** Every number is one run at `seed=0`. Differences within 1 pt (CFSC −0.36 pt, neg_has +0.20 pt) should not be treated as significant. The +3.25 pt FinFE strict gap is larger but still worth re-checking with multiple seeds.
4. **`neg_has` ECE of 0.0882 is weak.** The noul temperature was fitted on only 34 calibration samples and is inherently unstable.
5. **Do not over-read `neg_subject`'s 0.7003 vs. 0.8129.** The gap concentrates in two extremely small classes: 摩尔龙 (1 sample) and 钱宝 (4 samples).
6. **Untried directions**: a bare `bert-base-chinese` encoder as a control, using FinBERT's `pooler_output` instead of `last_hidden_state[:, 0]`, raising `head_layers` from 2 to 3 to recover the depth difference, and giving FinBERT-Laya more epochs to see whether it closes the 0.5 pt CFSC gap.

---

## 7. Reproduction

```bash
# 0) Environment: laya env (torch 2.6.0+cu126 / transformers 4.57.6 / laya 0.4.15)
export LAYA_ROOT=/data/laya-fin
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$LAYA_ROOT/hf_cache"

# 1) Build the base checkpoint (FinBERT encoder + random decision head), about 1 minute
python FinBertLaya/scripts/make_fb_base.py

# 2) Dry run to confirm data parsing (expect 53341 lines / 55562 questions / skipped {})
laya-train --data "$LAYA_ROOT/laya/all_train.jsonl" --eval "$LAYA_ROOT/laya/finfe_eval.jsonl" \
  --out "$LAYA_ROOT/fb_ckpt" --base "$LAYA_ROOT/fb_base" --dry-run

# 3) Train (23 minutes)
bash FinBertLaya/scripts/train.sh

# 4) Evaluate the three test sets and recompute metrics
bash FinBertLaya/scripts/eval.sh

# 5) Deployment metrics (run in the laya env)
python common/deploy_metrics.py fb_laya
```

See [`../FinBertLaya/README.md`](../FinBertLaya/README.md) for the list of artifacts.
