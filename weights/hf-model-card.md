---
language:
- zh
license: mit
library_name: pytorch
pipeline_tag: text-classification
tags:
- finance
- sentiment-analysis
- chinese
- laya
- decision-model
- calibration
- aspect-based-sentiment
datasets:
- CFLEB
metrics:
- accuracy
- f1
- calibration-error
base_model:
- yiyanghkust/finbert-tone-chinese
---

# FinBERT-Laya (117M)

Chinese financial sentiment as a decision problem. **A third of the parameters, the same accuracy as the 322M version.**

Sentiment analysis is reframed as *decisions over a closed option set*: the model does not generate text, it returns a **probability distribution over the options**. Probabilities are therefore calibrated by construction, and a confidence threshold can decide when to abstain.

| | |
|---|---|
| Encoder | `yiyanghkust/finbert-tone-chinese` (BertModel, 102.27 M) |
| Decision head | Laya native (14.97 M) |
| **Total parameters** | **117.24 M** |
| Weight size | 223.6 MB (fp16, 235 tensors) |
| Context length | 512 |
| Task coverage | Sentiment 3-way · aspect-based sentiment · negative-subject identification (**one model**) |
| Latency | 8.66 ms p50 (RTX 4090, batch=1) |
| GPU memory | 667 MB |

## Why 117M

Laya builds its decision head from `encoder.config.hidden_size`. mmBERT-base and FinBERT are both 768, so the head shape is identical (14.97 M — not a single parameter differs). The 205 M saved lives entirely in the encoder embedding:

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| Vocabulary | 256,000 | 21,128 |
| Embedding parameters | 196.61 M | **16.23 M** |
| Encoder body | 110.4 M (22 layers) | 85.0 M (12 layers) |
| Decision head | 14.97 M | 14.97 M |
| Total | 321.9 M | **117.2 M** |

Chinese financial tasks never exercise a 256k multilingual vocabulary; those 196.6 M parameters are pure redundancy.

## Usage

```bash
pip install "transformers>=4.48,<5" laya==0.4.15
```

```python
import laya

router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "zhanghanshuo/FinLaya")   # pass the Hub ID directly

questions = {
    "sentiment_social": {
        "type": "choice",
        "instructions": "这段股吧/雪球社交文本对股市或个股的情绪倾向是什么？",
        "criteria": {
            "negative": "看空、恐慌、抱怨、亏损、割肉、骂声等消极情绪",
            "neutral":  "中性陈述、疑问、无明确多空倾向",
            "positive": "看多、乐观、期待上涨等积极情绪",
        },
    }
}

ans = router.predict("试仓500股，看看门道！", questions, model=name)
print(ans["sentiment_social"])          # predicted label + probability distribution
```

Or download first, then register from a local path:

```bash
huggingface-cli download zhanghanshuo/FinLaya --local-dir finbert-laya
```

```python
name = router.register("finbert-laya", "./finbert-laya")
```

`criteria` must match the training-time definition **exactly**. The complete set is in `questions.json` in this repository.

### Input format per task

| Task | qid | Type | `state` format |
|---|---|---|---|
| Financial social sentiment | `sentiment_social` | choice(3) | plain text |
| Aspect-based sentiment | `sentiment_absa` | choice(3) | `方面词：X\n语句：……【X】……` |
| Negative subject — presence | `neg_has` | noul | `候选实体：A；B\n文本：……` |
| Negative subject — which | `neg_subject` | choice(21) | same as above; emits one of 20 entities or `以上都不是` |

## Training

RLCD recipe (soft-label cross-entropy `w_rps=1.0` + spherical regularization `w_sph=0.75`, 4 RL samples, sigma 0.4→0.1), 4 epochs / 3,068 steps / 23 minutes / ~3.8 GB GPU memory. Label smoothing 0.05, encoder lr 2.5e-5, decision-head lr 1e-4.

Fitted automatically after training:

| Item | Value |
|---|---|
| choice temperature | 1.3531 |
| noul temperature | 1.9008 |
| abstention threshold `choice:3-5` | 0.5968 |
| abstention threshold `noul:2` | 0.8953 |
| abstention threshold `choice:11+` | 0.7937 |

## Evaluation

All three datasets come from [CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications). The **strict protocol** removes samples whose text SHA256 matches the training set exactly.

| Task | n (full / strict) | Accuracy | Macro-F1 | ECE |
|---|---|---|---|---|
| FinFE social sentiment | 2,020 / 923 | 78.02 / **80.28** | 0.7629 / 0.7371 | 0.0485 / 0.0710 |
| FinNSP negative presence | 500 / 499 | 95.20 / 95.19 | 0.9520 / 0.9518 | 0.0882 / 0.0882 |
| FinNSP negative subject | 259 / 259 | 95.75 / 95.75 | 0.4669 / 0.4669 | 0.1546 / 0.1546 |
| CFSC-ABSA aspect-level | 8,312 / 7,870 | 93.01 / 93.01 | 0.8909 / 0.8905 | 0.0607 / 0.0618 |

Against the 322M version:

| Task | Protocol | mmBERT-Laya (322M) | **This model (117M)** | Delta |
|---|---|---:|---:|---:|
| FinFE | strict | 77.03 | **80.28** | **+3.25 pt** |
| FinNSP negative presence | full | 95.00 | 95.20 | +0.20 pt |
| FinNSP negative subject | full | **96.91** | 95.75 | −1.16 pt |
| CFSC-ABSA | full | **93.37** | 93.01 | −0.36 pt |

The +3.25 pt on FinFE strict comes from the 322M version being fine-tuned from the official Laya checkpoint, which gives it a memorization edge on the 54.3% overlapping samples. Remove the overlap and that edge disappears — while this model's decision head was learned from scratch.

## Limitations

1. **512-token truncation.** 5.00% of FinNSP training samples are longer (p99=1,851, max=6,777), so tail information is lost. This is a hard BERT architectural limit.
2. **Severe long tail in negative-subject identification.** Of 259 evaluation cases, 102 are `以上都不是` and the other 157 spread over 13 entities, most with only 1–7 cases. Per-class metrics are noisy.
3. **The `noul` temperature was fitted on 34 samples**, so `neg_has` ECE is less trustworthy than the other tasks.
4. **Single seed.** One run at `seed=0`; differences within 1 pt should not be read as significant.
5. **Randomly initialized decision head.** The starting point differs from the mmBERT version — compare endpoints only, not convergence speed.

## Resources

- Code and full experiments: https://github.com/Leosann/FinLaya
- Three-model full-metric comparison: [`docs/THREE_WAY_COMPARISON.md`](https://github.com/Leosann/FinLaya/blob/main/docs/THREE_WAY_COMPARISON.md)
- Source evidence and method for the encoder swap: [`docs/FINBERT_AS_ENCODER.md`](https://github.com/Leosann/FinLaya/blob/main/docs/FINBERT_AS_ENCODER.md)

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
```
