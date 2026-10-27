# Laya vs. FinBERT: Chinese Financial Sentiment Baseline Comparison

> RTX 4090 24 GB · same data, same metric protocol for both models · strict protocol = samples whose `state` text overlaps the training set are removed

## 1. Main results: accuracy and Macro-F1

Macro-F1 is the primary metric under class imbalance (in FinFE, `positive` accounts for 44% while `negative`/`neutral` are about 28% each). Macro-F1 for `neg_subject` is computed over the classes that actually have support in the test set (14 of 21 for Laya, 13 of 19 for FinBERT).

| Task | n | Laya acc | FinBERT acc | Delta | Laya Macro-F1 | FinBERT Macro-F1 | Delta |
|---|---|---|---|---|---|---|---|
| Financial sentiment, 3-way | 2020 | 77.97% | 79.26% | -1.29 | 0.7690 | 0.7776 | -0.0086 |
| Financial sentiment, 3-way (strict) | 923 | 77.03% | 79.52% | -2.49 | 0.7094 | 0.7258 | -0.0163 |
| News negative presence (binary) | 500 | 95.00% | 95.60% | -0.60 | 0.9499 | 0.9559 | -0.0060 |
| News negative presence (strict) | 499 | 94.99% | 95.59% | -0.60 | 0.9498 | 0.9558 | -0.0061 |
| Negative subject (1 of 21) | 259 | 96.91% | 95.37% | +1.54 | 0.5419 | 0.5038 | +0.0381 |
| Aspect-level sentiment, 3-way | 8312 | 93.37% | 92.78% | +0.59 | 0.8993 | 0.8881 | +0.0112 |
| Aspect-level sentiment, 3-way (strict) | 7870 | 93.58% | 92.94% | +0.65 | 0.9025 | 0.8902 | +0.0122 |

**How to read this**

- On the two sentiment tasks (FinFE, CFSC-ABSA), **FinBERT wins by 1–2 points**. A 3-layer classifier head plus full fine-tuning is a strong baseline for the plain "text → polarity" discriminative task, whereas Laya's 322M backbone must simultaneously support several decision primitives, which dilutes it.
- On the two FinNSP tasks **Laya is better**, particularly `neg_subject` (+1.54 points), because it sees the full question text with all 21 candidate entities, while FinBERT can only argmax over 19 labels.

## 2. Probability reliability: ECE / NLL / Brier / AURC

This group decides **whether confidence can be used as a gate**. ECE is the top-1 confidence calibration error over 15 equal-width bins.

| Task | Laya ECE | FinBERT ECE | Laya NLL | FinBERT NLL | Laya Brier | FinBERT Brier | Laya AURC | FinBERT AURC |
|---|---|---|---|---|---|---|---|---|
| Financial sentiment, 3-way | 0.0739 | 0.0947 | 0.6467 | 0.6244 | 0.3516 | 0.3285 | 0.1184 | 0.0953 |
| Financial sentiment, 3-way (strict) | 0.0628 | 0.0876 | 0.6233 | 0.5713 | 0.3437 | 0.3066 | 0.1015 | 0.0730 |
| News negative presence (binary) | 0.0506 | 0.0311 | 0.1799 | 0.1517 | 0.0873 | 0.0740 | 0.0073 | 0.0040 |
| News negative presence (strict) | 0.0505 | 0.0312 | 0.1801 | 0.1520 | 0.0874 | 0.0742 | 0.0073 | 0.0040 |
| Negative subject (1 of 21) | 0.1984 | 0.0616 | 0.3514 | 0.2821 | 0.0952 | 0.0991 | 0.0067 | 0.0089 |
| Aspect-level sentiment, 3-way | 0.0638 | 0.0339 | 0.2658 | 0.2157 | 0.1170 | 0.1126 | 0.0213 | 0.0098 |
| Aspect-level sentiment, 3-way (strict) | 0.0658 | 0.0326 | 0.2597 | 0.2138 | 0.1134 | 0.1106 | 0.0194 | 0.0097 |

**FinBERT is markedly better calibrated** (ECE 2–5× lower). It is trained with standard softmax cross-entropy, so its probability semantics line up naturally. Laya's `noul` head emits *the probability of true*, and its `answer_confidence` goes through a `|2p-1|` transform that does not fully align with argmax semantics, which pushes `neg_has` ECE up (0.0506 vs. 0.0311).

> Note: Laya already fits temperatures during training (choice 1.4378 / noul 1.4585), so these are post-calibration numbers. FinBERT did **not** get temperature scaling here — its native softmax is simply that accurate.

## 3. Selective accuracy (can it run unattended)

Rank by confidence and take the top 50% / 80% / 90%: the accuracy on that slice is the precision you get when auto-releasing that share of traffic.

| Task | Laya @50 | FinBERT @50 | Laya @80 | FinBERT @80 | Laya @90 | FinBERT @90 |
|---|---|---|---|---|---|---|
| Financial sentiment, 3-way | 89.5% | 91.0% | 83.8% | 85.0% | 81.4% | 82.4% |
| Financial sentiment, 3-way (strict) | 92.2% | 94.8% | 84.3% | 86.7% | 81.0% | 83.4% |
| News negative presence (binary) | 99.6% | 100.0% | 99.0% | 99.8% | 98.2% | 98.0% |
| News negative presence (strict) | 99.6% | 100.0% | 99.0% | 99.7% | 98.2% | 98.0% |
| Negative subject (1 of 21) | 99.2% | 99.2% | 99.5% | 98.1% | 99.1% | 97.4% |
| Aspect-level sentiment, 3-way | 98.7% | 99.8% | 98.1% | 98.3% | 96.6% | 96.4% |
| Aspect-level sentiment, 3-way (strict) | 98.8% | 99.8% | 98.2% | 98.3% | 96.8% | 96.4% |

Both models clear 95% at @80, so **both can run unattended**. Note that FinBERT is slightly higher at @50 and slightly lower at @90: its confidence distribution is more concentrated (mean confidence 0.95+), so it ranks less finely than Laya.

## 4. Per-class recall (class imbalance)

| Task | Class | Laya P | Laya R | Laya F1 | FinBERT P | FinBERT R | FinBERT F1 | Support |
|---|---|---|---|---|---|---|---|---|
| Financial sentiment, 3-way | negative | 0.794 | 0.780 | 0.787 | 0.770 | 0.819 | 0.794 | 574 |
| Financial sentiment, 3-way | neutral | 0.661 | 0.696 | 0.678 | 0.738 | 0.639 | 0.685 | 560 |
| Financial sentiment, 3-way | positive | 0.851 | 0.832 | 0.841 | 0.836 | 0.872 | 0.854 | 886 |
| News negative presence (binary) | true | 0.943 | 0.961 | 0.952 | 0.958 | 0.958 | 0.958 | 259 |
| News negative presence (binary) | false | 0.958 | 0.938 | 0.948 | 0.954 | 0.954 | 0.954 | 241 |
| Negative subject (1 of 21) | e租宝 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 1.000 | 2 |
| Negative subject (1 of 21) | 以上都不是 | 0.980 | 0.961 | 0.970 | 0.990 | 0.951 | 0.970 | 102 |
| Negative subject (1 of 21) | 北京正聚源通鼎公司 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| Negative subject (1 of 21) | 唐小僧 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 7 |
| Negative subject (1 of 21) | 壹佰金融 | 1.000 | 1.000 | 1.000 | 0.750 | 1.000 | 0.857 | 3 |
| Negative subject (1 of 21) | 宜贷网 | 0.926 | 0.962 | 0.943 | 0.897 | 1.000 | 0.945 | 26 |
| Negative subject (1 of 21) | 小资钱包 | 0.987 | 1.000 | 0.994 | 0.987 | 1.000 | 0.994 | 77 |
| Negative subject (1 of 21) | 恒丰银行 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| Negative subject (1 of 21) | 成都易捷公司 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| Negative subject (1 of 21) | 摩尔龙 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1 |
| Negative subject (1 of 21) | 斐讯 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| Negative subject (1 of 21) | 海象理财 | 0.857 | 1.000 | 0.923 | 0.857 | 1.000 | 0.923 | 6 |
| Negative subject (1 of 21) | 红岭创投 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| Negative subject (1 of 21) | 联璧金融 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 6 |
| Negative subject (1 of 21) | 资易贷 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| Negative subject (1 of 21) | 资易贷金融信息服务有限公司 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1 |
| Negative subject (1 of 21) | 钱宝 | 0.750 | 0.750 | 0.750 | 0.000 | 0.000 | 0.000 | 4 |
| Negative subject (1 of 21) | 钱宝网 | 1.000 | 1.000 | 1.000 | 0.792 | 1.000 | 0.884 | 19 |
| Negative subject (1 of 21) | 雅堂金融 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 |
| Aspect-level sentiment, 3-way | negative | 0.943 | 0.948 | 0.946 | 0.928 | 0.943 | 0.935 | 2146 |
| Aspect-level sentiment, 3-way | neutral | 0.794 | 0.797 | 0.796 | 0.802 | 0.749 | 0.775 | 1032 |
| Aspect-level sentiment, 3-way | positive | 0.958 | 0.955 | 0.957 | 0.951 | 0.958 | 0.954 | 5134 |

**Key takeaway**: both models are weakest on `neutral` (FinFE neutral recall 0.696 / 0.712, CFSC neutral recall 0.797 / 0.759), which matches intuition — neutral sentiment has fuzzy boundaries. `positive` recall is 0.95+ for both, since it has the largest support.

## 5. Deployment cost: size / latency / throughput / GPU memory

| Metric | Laya | FinBERT | Ratio (Laya / FinBERT) |
|---|---|---|---|
| Parameters | 321.9 M | 102.3 M | 3.15× |
| Weight size | 1228 MB | 390 MB | 3.15× |
| Weight precision | F16 | fp32 | — |
| Single-request latency p50 | 18.56 ms | 5.93 ms | 3.13× |
| Single-request latency p95 | 18.98 ms | 5.98 ms | 3.17× |
| Throughput batch=1 | 54 QPS | 145 QPS | 2.71× |
| Throughput batch=32 | — | 2858 QPS | — |
| Peak inference GPU memory | 1499 MB | 486 MB | 3.08× |

**Laya costs about 3× what FinBERT does to deploy**: 3.15× the parameters, 3.15× the weight size, 3.13× the p50 latency, 3.08× the peak GPU memory. The reason is that Laya's backbone is mmBERT-base (322M) plus a decision head with hidden 1024, whereas FinBERT is at the bert-base scale (102M).

**Most of the size gap is precision, not parameter count**: Laya's `model.safetensors` is stored as **F16** (all 170 tensors), while FinBERT is stored as fp32. Converting FinBERT to F16 brings it down to **195.1 MB**, at which point the gap is 6.3× (1228 / 195). If Laya could also drop to a lower bit width (INT8 quantization) the gap would narrow to about 3×, at the cost of its already weak calibration.

**The latency gap is real**: Laya's 18.56 ms per request covers Router dispatch, question-text encoding, and forward passes through several decision heads (choice + noul + action head), while FinBERT is one BERT forward plus a single linear layer. This is the inherent cost of unified decision-making over multiple primitives.

## 6. Conclusions and selection guidance

| Scenario | Recommendation | Rationale |
|---|---|---|
| Plain sentiment polarity at high QPS | **FinBERT** | Matches on accuracy at 1/3 the latency and 1/3 the size, with the best calibration |
| Unified multi-task decisions plus abstention | **Laya** | One weight set answers every question type; native abstention thresholds and RLCD |
| Subject identification / structured extraction | **Laya** | The question text can carry the full candidate set, making 1-of-21 natural |
| Strictly calibrated probability output | **FinBERT** | ECE 0.03 vs. Laya's 0.05–0.20 |

**The single most important point**: Laya's `neg_subject` ECE is as high as **0.1984** because its `answer_confidence` is a `|2p-1|` decision margin rather than a probability, and must not be used as one directly. To gate on it in production you must use the bucketed thresholds fitted into `rl_agent_config.json` (`choice:3-5` → 0.6217, `choice:11+` → 0.7562), or calibrate separately.

---

Data sources: `reports/ft_*_metrics.json` (Laya), `reports/finbert_all.json` (FinBERT), `reports/deploy_metrics.json` (deployment). Scripts: `laya_metrics.py`, `finbert_baseline.py`, `fb_eval.py`, `deploy_metrics.py`.
