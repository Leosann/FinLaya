# FinBERT — 判别式基线

用 `yiyanghkust/finbert-tone-chinese`（102M，中文金融语料预训练）做对照基线。为了和 Laya 公平比较，**每个决策任务单独微调一个分类头**，四个任务合计训练 4 个独立模型。

## 任务映射

Laya 的三种决策原语到 FinBERT 分类头：

| FinBERT 任务名 | 数据集 | qid | 原语 | 类别数 | 训练/评测 |
|---|---|---|---:|---:|---:|
| `absa` | CFSC-ABSA | `sentiment_absa` | `choice` | 3 | 33,184 / 8,312 |
| `finfe` | FinFE | `sentiment_social` | `choice` | 3 | 16,157 / 2,020 |
| `nsp_has` | FinNSP | `neg_has` | `noul` | 2 | 4,000 / 500 |
| `nsp_subject` | FinNSP | `neg_subject` | `choice` | 19 | 4,000 / 259 |

> `nsp_subject` 是 19 类而非 Laya 的 21 类：FinBERT 的词表直接从**训练集**里抽标签，而两个实体（「北京资易贷金融信息服务有限公司」「易捷金融」）只出现在评测集，训练集样本为零，所以没进词表。这是判别式路线的固有局限——**新增实体必须重训**，而 Laya 只需改题面里的 `criteria`。

## 目录结构

```
FinBert/
├─ scripts/
│  ├─ setup_finbert.sh          独立 conda 环境（与 laya 环境隔离）
│  ├─ finbert_baseline.py       [核心] 四任务训练 + 评测 + 全套指标
│  └─ fb_eval.py                分批评测入口，增量写 reports/finbert_all.json
├─ logs/
│  └─ finbert_train.log         四个任务的实际训练输出
└─ reports/
   └─ finbert_all.json          全量 + 严格口径两套指标（各类别 P/R/F1、ECE、Brier、NLL、选择性准确率、延迟、显存）
```

## 运行

```bash
# 1) 环境（python 3.11 + torch 2.6.0+cu126 + transformers>=4.40,<5 + sklearn）
bash scripts/setup_finbert.sh

ENV_PY=/data/miniconda/envs/finbert/bin/python

# 2) 训练四个分类头
$ENV_PY scripts/finbert_baseline.py train

# 3) 评测 —— 必须分批，一次跑全部会被 OOM killer 杀掉且不留 Traceback
$ENV_PY scripts/fb_eval.py absa
$ENV_PY scripts/fb_eval.py finfe
$ENV_PY scripts/fb_eval.py nsp_has
$ENV_PY scripts/fb_eval.py nsp_subject
```

`fb_eval.py` 每个任务算两套 tag：`full` 与 `strict`（剔除与训练集 state 哈希重叠的样本），增量写入 `reports/finbert_all.json`，所以分批跑不会互相覆盖。

## 训练超参

```python
epochs=3, batch_size=32, lr=2e-5, max_len=256, seed=42
AdamW(weight_decay=0.01) + 线性 warmup(10%) + fp16 GradScaler + clip_grad_norm(1.0)
```

实测耗时（RTX 4090）：

| 任务 | 耗时 | 末 epoch loss |
|---|---:|---:|
| `absa` | 218 s | 0.1371 |
| `finfe` | 63 s | 0.3593 |
| `nsp_has` | 28 s | 0.0429 |
| `nsp_subject` | 16 s | 0.2516 |

## 评测指标实现

| 指标 | 实现 |
|---|---|
| Macro-F1 | 只统计**测试集中有样本**的类别（`macro_f1_present`），零支持类别不参与平均 |
| 各类别 P/R/F1 | 逐类输出，附 support |
| ECE | 15 bin 等宽分箱，`|acc - conf|` 加权求和 |
| Brier | 多分类 `Σ(p_i - y_i)² / n`；另有 top-1 版 `(p_top - 1)²` |
| NLL | `-log p(correct)` 均值 |
| AURC | 风险-覆盖曲线下面积，衡量「置信度排序是否可靠」 |
| 选择性准确率 | 按置信度降序取前 X%，在该子集上的准确率（X = 50 / 80 / 90） |
| 延迟 / 吞吐 | `latency_p50_ms` / `latency_p95_ms` / `throughput_qps`，batch=64 预热后测 |
| 显存 | `torch.cuda.max_memory_allocated()` |

## 结果摘要

| 任务 | n | acc | mF1 | ECE | 严格 acc | 严格 n |
|---|---:|---:|---:|---:|---:|---:|
| FinFE | 2,020 | 79.26% | 0.7776 | 0.0947 | 79.52% | 923 |
| FinNSP `neg_has` | 500 | 95.60% | 0.9559 | 0.0311 | 95.59% | 499 |
| FinNSP `neg_subject` | 259 | 95.37% | 0.5038¹ | 0.0616 | 95.37% | 259 |
| CFSC-ABSA | 8,312 | 92.78% | 0.8881 | 0.0339 | 92.94% | 7,870 |

¹ 19 类里 5 类零支持；只看有样本的 14 类为 **0.6838**。

完整数值见 `reports/finbert_all.json`，横向对照见 [../docs/COMPARE_Laya_vs_FinBERT.md](../docs/COMPARE_Laya_vs_FinBERT.md)。

## 注意

- **必须用独立 conda 环境**。`laya` 要求 `transformers>=4.48,<5`，FinBERT 分支用的是 `>=4.40,<5`，两者装在一起会互相顶掉依赖。
- **训练好的权重不在本仓库**（四个头合计约 1.5 GB）。`reports/finbert_all.json` 里保留了每次评测的 `model_dir_mb` 与 `n_params`，可复现体积结论。
- `TASKS` 映射写死在 `finbert_baseline.py` 顶部；新增任务只需加一行 `(train_file, eval_file, qid)`。
