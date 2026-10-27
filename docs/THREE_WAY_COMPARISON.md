# Three-model full-metric comparison

> FinLaya-mmBERT (322M) · FinLaya-FinBERT (117M) · FinBERT classifier (102M)
> Same data, same metrics, same RTX 4090. All numbers come from `FinLaya-mmBERT/reports/`, `FinLaya-FinBERT/reports/`, `FinBert/reports/`, and `common/reports/deploy_metrics.json`.

---

## 0. Model configuration

| | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT classifier |
|---|---|---|---|
| Encoder | ModernBERT-base (mmBERT) | BERT-base (FinBERT) | BERT-base (FinBERT) |
| Encoder parameters | 307.0 M | 102.3 M | 102.3 M |
| ├ embedding | **196.6 M** (vocab 256,000) | **16.2 M** (vocab 21,128) | 16.2 M |
| └ transformer body | 110.4 M (22 layers × 768) | 85.0 M (12 layers × 768) | 85.0 M |
| Decision head | 15.0 M (Laya native) | **15.0 M (identical)** | 4 separate classifiers (≈0.6 M each) |
| **Total parameters** | **321.9 M** | **117.2 M** | 102.3 M |
| Context length | 8,192 | 512 | 512 |
| Weight dtype | F16 | F16 | fp32 (can convert to F16) |
| Output | Distribution over options + abstention | Distribution over options + abstention | softmax probabilities |
| Task coverage | **1 model covers 4 question types** | **1 model covers 4 question types** | **4 tasks need 4 models** |
| Cost of adding a class | Edit the question's `criteria` | Edit the question's `criteria` | Must retrain |

---

## 1. Accuracy (full protocol)

| Task | n | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT | Best |
|---|---:|---:|---:|---:|---|
| FinFE social sentiment | 2,020 | 77.97 | 78.02 | **79.26** | FinBERT |
| FinNSP negative presence | 500 | 95.00 | 95.20 | **95.60** | FinBERT |
| FinNSP negative subject | 259 | **96.91** | 95.75 | 95.37 | FinLaya-mmBERT |
| CFSC-ABSA aspect-level | 8,312 | **93.37** | 93.01 | 92.78 | FinLaya-mmBERT |

**Spread**: across the four tasks the range between best and worst model is only **1.54 pt** (FinNSP negative subject) down to **0.59 pt** (CFSC). In other words, **swapping the encoder barely affects accuracy**.

---

## 2. Accuracy (strict protocol, training-set overlap removed)

| Task | n (after removal) | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT | Best |
|---|---:|---:|---:|---:|---|
| FinFE social sentiment | 923 (−1,097) | 77.03 | **80.28** | 79.52 | **FinLaya-FinBERT ⭐** |
| FinNSP negative presence | 499 (−1) | 94.99 | 95.19 | **95.59** | FinBERT |
| FinNSP negative subject | 259 (−0) | **96.91** | 95.75 | 95.37 | FinLaya-mmBERT |
| CFSC-ABSA aspect-level | 7,870 (−442) | **93.58** | 93.01 | 92.94 | FinLaya-mmBERT |

**On FinFE, FinLaya-FinBERT beats FinLaya-mmBERT by 3.25 pt** — the largest difference in the whole comparison. The reason: 54.3% of the FinFE evaluation set duplicates training text, and FinLaya-mmBERT continues from the official laya checkpoint, giving it a memorization edge on "seen" samples. Remove the overlap and that edge vanishes, while FinLaya-FinBERT's decision head was learned from scratch and generalizes better.

---

## 3. Macro-F1

| Task | Protocol | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | full | **0.7690** | 0.7629 | 0.7776 |
| FinFE | strict | 0.7094 | **0.7371** | 0.7258 |
| FinNSP negative presence | full | 0.9499 | 0.9520 | **0.9559** |
| FinNSP negative presence | strict | 0.9498 | 0.9518 | **0.9558** |
| FinNSP negative subject | full | 0.5419 | 0.4669 | 0.5038 |
| FinNSP negative subject (14 classes with support) | full | **0.8129** | 0.7003 | 0.6838 |
| CFSC-ABSA | full | **0.8993** | 0.8909 | 0.8881 |
| CFSC-ABSA | strict | **0.9025** | 0.8905 | 0.8902 |

> The raw Macro-F1 for `FinNSP negative subject` (0.5419 / 0.4669 / 0.5038) is dragged down heavily by zero-support classes: 7 of the 21 classes never appear in the test set and are forced to 0. **Only the 14 classes with support** give a comparable number — see the `macro_f1_present` row. FinBERT's vocabulary has only 19 classes (it lacks 「北京资易贷金融信息服务有限公司」 and 「易捷金融」); its 14-class figure is over classes with support.

---

## 4. Probability calibration

### 4.1 ECE (lower is better)

| Task | Protocol | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | full | 0.0739 | **0.0485** | 0.0947 |
| FinFE | strict | **0.0628** | 0.0710 | 0.0876 |
| FinNSP negative presence | full | 0.0506 | 0.0882 | **0.0311** |
| FinNSP negative subject | full | 0.1984 | 0.1546 | **0.0616** |
| CFSC-ABSA | full | 0.0638 | 0.0607 | **0.0339** |
| CFSC-ABSA | strict | 0.0658 | 0.0618 | **0.0326** |

### 4.2 NLL (lower is better)

| Task | Protocol | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | full | 0.6467 | **0.5878** | 0.6244 |
| FinFE | strict | 0.6233 | **0.5328** | 0.5713 |
| FinNSP negative presence | full | **0.1799** | 0.1851 | 0.1517 |
| FinNSP negative subject | full | 0.3514 | 0.3113 | **0.2821** |
| CFSC-ABSA | full | 0.2658 | 0.2424 | **0.2157** |
| CFSC-ABSA | strict | 0.2597 | 0.2414 | **0.2138** |

### 4.3 Brier (multiclass, lower is better)

| Task | Protocol | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | full | 0.3516 | **0.3259** | 0.3285 |
| FinFE | strict | 0.3437 | **0.2916** | 0.3066 |
| FinNSP negative presence | full | 0.0873 | 0.0815 | **0.0740** |
| FinNSP negative subject | full | 0.0952 | **0.0850** | 0.0991 |
| CFSC-ABSA | full | 0.1170 | **0.1119** | 0.1126 |
| CFSC-ABSA | strict | 0.1134 | 0.1113 | **0.1106** |

### 4.4 AURC (area under the risk–coverage curve, lower is better)

| Task | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|
| FinFE | 0.1184 | 0.0995 | **0.0953** |
| FinNSP negative presence | **0.0073** | 0.0097 | 0.0040 |
| FinNSP negative subject | **0.0067** | 0.0180 | 0.0089 |
| CFSC-ABSA | 0.0213 | 0.0119 | **0.0098** |

### 4.5 Mean confidence

| Task | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|
| FinFE | 0.8517 | 0.8287 | **0.8840** |
| FinNSP negative presence | 0.9174 | 0.8805 | **0.9852** |
| FinNSP negative subject | 0.7745 | 0.8172 | **0.9080** |
| CFSC-ABSA | 0.8883 | 0.8902 | **0.9616** |

> **The FinBERT classifier has the highest confidence overall (0.88–0.99) yet the lowest ECE**, meaning its high confidence is earned. FinLaya-FinBERT sits lower (0.83–0.89) but has better ECE, meaning it is more conservative. Different strategies; both are usable.

---

## 5. Selective accuracy (top X% by confidence, auto-processed)

### 5.1 Full protocol

| Task | Coverage | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | @50% | 89.50 | **91.39** | 90.99 |
| FinFE | @80% | 83.79 | **85.09** | 85.02 |
| FinFE | @90% | 81.35 | 81.96 | **82.40** |
| FinNSP negative presence | @50% | 99.60 | 99.60 | **100.00** |
| FinNSP negative presence | @80% | 99.00 | 98.75 | **99.75** |
| FinNSP negative presence | @90% | **98.22** | 98.00 | 98.00 |
| FinNSP negative subject | @50% | 99.23 | 98.46 | **99.23** |
| FinNSP negative subject | @80% | **99.52** | 97.58 | 98.07 |
| FinNSP negative subject | @90% | **99.14** | 97.85 | 97.42 |
| CFSC-ABSA | @50% | 98.65 | 99.47 | **99.83** |
| CFSC-ABSA | @80% | 98.11 | **98.18** | 98.32 |
| CFSC-ABSA | @90% | 96.58 | **96.68** | 96.38 |

### 5.2 Strict protocol

| Task | Coverage | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | @50% | 92.21 | **95.02** | 94.81 |
| FinFE | @80% | 84.28 | **87.94** | 86.72 |
| FinFE | @90% | 80.99 | **84.00** | 83.39 |
| FinNSP negative presence | @50% | 99.60 | 99.60 | **100.00** |
| FinNSP negative presence | @80% | 99.00 | 98.75 | **99.75** |
| FinNSP negative presence | @90% | **98.22** | 98.00 | 98.00 |
| FinNSP negative subject | @50% | 99.23 | 98.46 | **99.23** |
| FinNSP negative subject | @80% | **99.52** | 97.58 | 98.07 |
| FinNSP negative subject | @90% | **99.14** | 97.85 | 97.42 |
| CFSC-ABSA | @50% | 98.75 | 99.49 | **99.82** |
| CFSC-ABSA | @80% | 98.22 | 98.27 | **98.33** |
| CFSC-ABSA | @90% | **96.82** | 96.75 | 96.43 |

**Under the strict protocol, FinFE @50 reaches 95.02%** — meaning that after removing overlapping samples, FinLaya-FinBERT can fully automate half the traffic at 95% accuracy.

---

## 6. Per-class recall (full protocol)

### 6.1 FinFE (3 classes)

| Class | Support | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|---:|
| negative | 574 | 0.780 | 0.791 | **0.819** |
| neutral | 560 | **0.696** | 0.611 | 0.639 |
| positive | 886 | 0.832 | **0.880** | 0.872 |

### 6.2 FinNSP negative presence (2 classes)

| Class | Support | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|---:|
| true | 259 | **0.961** | 0.950 | 0.958 |
| false | 241 | 0.938 | **0.954** | 0.954 |

### 6.3 FinNSP negative subject (14 classes with support)

| Entity | Support | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|---:|
| 以上都不是 | 102 | **0.961** | **0.961** | 0.951 |
| 小资钱包 | 77 | **1.000** | **1.000** | **1.000** |
| 宜贷网 | 26 | 0.962 | **1.000** | **1.000** |
| 钱宝网 | 19 | **1.000** | 0.947 | **1.000** |
| 唐小僧 | 7 | **1.000** | **1.000** | **1.000** |
| 海象理财 | 6 | **1.000** | **1.000** | **1.000** |
| 联璧金融 | 6 | **1.000** | **1.000** | **1.000** |
| 红岭创投 | 4 | **1.000** | **1.000** | **1.000** |
| 钱宝 | 4 | **0.750** | 0.250 | 0.000 |
| 壹佰金融 | 3 | **1.000** | **1.000** | **1.000** |
| e租宝 | 2 | **1.000** | **1.000** | **1.000** |
| 恒丰银行 | 1 | 0.000 | 0.000 | 0.000 |
| 摩尔龙 | 1 | **1.000** | 0.000 | 0.000 |
| 资易贷金融信息服务有限公司 | 1 | 0.000 | 0.000 | 0.000 |

> All three models score a perfect 1.000 on 8 entities. The differences concentrate in two extremely small classes — `钱宝` (4 samples) and `摩尔龙` (1 sample) — plus `钱宝网` (19 samples). **Nearly all of the 0.8129 vs. 0.7003 vs. 0.6838 gap comes from those 5 samples**, and should not be read as a capability difference.

### 6.4 CFSC-ABSA (3 classes)

| Class | Support | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---:|---:|---:|---:|
| negative | 2,146 | **0.948** | 0.944 | 0.943 |
| neutral | 1,032 | **0.797** | 0.751 | 0.749 |
| positive | 5,134 | 0.955 | **0.960** | 0.958 |

**`neutral` is the weak spot across the board** (FinFE 0.611–0.696, CFSC 0.749–0.797). FinLaya-mmBERT leads by 4.6–4.8 pt on CFSC neutral, which is the main reason it leads on Macro-F1 for that task.

---

## 7. Deployment metrics

Single RTX 4090, batch=1 serial, weight size counting only the weight file (excluding the `checkpoint_latest/` snapshot).

| | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT (single task) | FinBERT (4-task equivalent) |
|---|---:|---:|---:|---:|
| Parameters | 321.9 M | **117.2 M** | 102.3 M | 409.2 M |
| Weight size | 614.0 MB | **223.6 MB** | 390.2 MB (fp32) / 195 MB (F16) | 1,560 MB (fp32) / **780 MB** (F16) |
| Latency p50 | 18.64 ms | **8.66 ms** | 5.93 ms | 5.93 ms (parallel) / 23.7 ms (4 serial) |
| Latency p95 | 19.34 ms | **8.92 ms** | 5.98 ms | 5.98 ms (parallel) |
| Throughput batch=1 | 53.0 QPS | **113.0 QPS** | 145.3 QPS | 36.3 QPS (4 serial) |
| Throughput batch=32 | — | — | 2,857.7 QPS | — |
| Peak GPU memory | 1,499 MB | **667 MB** | 486 MB | 4 × 486 = 1,944 MB |
| Models needed | **1** | **1** | 4 | 4 |

### Change relative to FinLaya-mmBERT

| | FinLaya-FinBERT | FinBERT (4-task equivalent) |
|---|---:|---:|
| Parameters | **−63.6%** | +27.1% |
| Weight size | **−63.6%** | +27.0% (F16) |
| Latency | **−53.5%** | +27.1% (serial) |
| Throughput | **×2.13** | −31.5% (serial) |
| GPU memory | **−55.5%** | +29.7% |

> **Key conclusion**: only FinLaya-FinBERT simultaneously achieves "one model covers every task" and "smaller and faster than the original". The FinBERT classifier is fastest per single task, but four tasks need four models, making its equivalent cost higher than FinLaya-mmBERT's.

---

## 8. Training cost

| | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT classifier |
|---|---|---|---|
| Models to train | 1 | 1 | **4** |
| Optimizer steps | 3,068 | 3,068 | — |
| Training data | 55,562 questions | 55,562 questions | Per task (33,184 / 16,157 / 4,000) |
| Wall time | 32 min | **23 min** | About 5.4 min (218 + 63 + 28 + 16 s) |
| GPU memory | 9.2 GB | **~3.8 GB** | ~0.5 GB |
| Final-epoch loss | 0.2018 | 0.3047 | absa 0.1371 / finfe 0.3593 / nsp_has 0.0429 / nsp_subject 0.2516 |
| Starting point | Official laya ckpt (loss 0.3983) | Decision head **randomly initialized** (loss 1.3957) | No ImageNet; pure FinBERT pretraining |
| Calibration | Automatic temperature fitting + abstention-threshold fitting | Automatic temperature fitting + abstention-threshold fitting | None (raw softmax) |

> **Loss is not comparable across models**: all three start from different points. FinLaya-mmBERT starts from an already fine-tuned checkpoint, while FinLaya-FinBERT's decision head and FinBERT's classifiers are random. Compare endpoint metrics only (§1–§5), never the size of the loss drop.

### Inference config produced automatically

| | FinLaya-mmBERT | FinLaya-FinBERT |
|---|---|---|
| choice temperature | 1.4378 | 1.3531 |
| noul temperature | 1.4585 | 1.9008 |
| Abstention threshold `choice:3-5` | 0.6217 | 0.5968 |
| Abstention threshold `noul:2` | 0.6661 | 0.8953 |
| Abstention threshold `choice:11+` | 0.7562 | 0.7937 |

---

## 9. Head-to-head: who wins where

| Dimension | Best | Note |
|---|---|---|
| FinFE accuracy (full) | **FinBERT** 79.26 | Ahead by 1.24 pt |
| FinFE accuracy (strict) | **FinLaya-FinBERT** 80.28 | Ahead by 0.76–3.25 pt |
| FinNSP negative presence | **FinBERT** 95.60 | Ahead by 0.40 pt |
| FinNSP negative subject | **FinLaya-mmBERT** 96.91 | Ahead by 1.16 pt, driven by long-tail samples |
| CFSC-ABSA accuracy | **FinLaya-mmBERT** 93.37 / 93.58 | Ahead by 0.36–0.57 pt |
| CFSC-ABSA Macro-F1 | **FinLaya-mmBERT** 0.8993 / 0.9025 | Ahead by 0.008–0.012 |
| neutral recall (CFSC) | **FinLaya-mmBERT** 0.797 | Ahead by 4.6 pt |
| ECE (FinFE full) | **FinLaya-FinBERT** 0.0485 | Ahead by 0.025 |
| ECE (FinNSP/CFSC) | **FinBERT** 0.0311 / 0.0339 | Ahead by 0.02–0.13 |
| NLL | **FinBERT** on 3 of 4 tasks | FinFE goes to FinLaya-FinBERT |
| Brier | Mixed | FinLaya-FinBERT wins on FinFE and negative subject |
| AURC | **FinBERT** 2/4, FinLaya-mmBERT 2/4 | FinLaya-FinBERT 0/4 |
| Selective accuracy @50 | **FinLaya-FinBERT** (FinFE strict 95.02) | — |
| Parameters / size | **FinBERT** (single task) / **FinLaya-FinBERT** (all tasks) | — |
| Latency / throughput | **FinBERT** (single task) | — |
| GPU memory | **FinBERT** 486 MB | — |
| Long context | **FinLaya-mmBERT** 8,192 | FinBERT only reaches 512 |
| Training speed | **FinBERT** 5.4 min (but trains 4 models) | FinLaya-FinBERT trains 1 in 23 min |
| Cost of adding a class | **The Laya pair** (edit the question text) | FinBERT must retrain |

---

## 10. Selection decision tree

```
Need one weight set covering several question types + confidence/abstention?
├─ Yes
│  ├─ Tight on GPU memory/latency, or strict long-tail entity requirements?
│  │  ├─ Long-tail entities matter → FinLaya-mmBERT (neg_subject mF1 0.8129, 8192 context)
│  │  └─ Resource-sensitive       → FinLaya-FinBERT (117M / 223.6 MB / 8.66 ms, on par elsewhere)
│  └─ Neither is a concern → Either works; FinLaya-FinBERT is leaner
└─ No, just want the strongest single task
   └─ FinBERT classifier (FinFE 79.26, 5.93 ms, but one model per task)
```

### In one line each

- **FinBERT**: fastest and most accurate per single task, but needs 4 models — 780 MB and 1,944 MB of GPU memory in equivalent terms.
- **FinLaya-FinBERT**: one 223.6 MB model covers everything, at accuracy on par with the 322M version and better on FinFE under the strict protocol.
- **FinLaya-mmBERT**: the only option that handles 8,192-token inputs, and the strongest on long-tail entities.

---

## 11. Data-reliability caveats

1. **The FinFE full protocol has 54.3% training-set overlap.** The full numbers (77.97 / 78.02 / 79.26) **must not be cited as generalization ability**; use the strict protocol (77.03 / 80.28 / 79.52).
2. **FinNSP negative subject has only 259 cases**: 102 are `以上都不是` and the remaining 157 spread over 13 entities, most with 1–7 samples. Per-class recall fluctuates wildly, and the `macro_f1_present` gap is driven mainly by 5 samples.
3. **The `neg_has` noul temperature was fitted on 34 samples** (both Laya models carry the warning `"fitted on only 34 calibration items"`), so its ECE is less trustworthy than other tasks.
4. **Single seed.** Every number is one run at `seed=0`. Differences within 1 pt (CFSC 0.36–0.57 pt, FinNSP 0.20–0.40 pt) should not be treated as significant; the 3.25 pt FinFE strict gap is larger but still worth re-checking with multiple seeds.
5. **FinBERT's 195 MB is a measured F16 conversion.** The conversion script `common/to_fp16.py` was written afterwards and has not been re-run on the server; the 390.2 MB fp32 figure is the reliably reproducible one.
