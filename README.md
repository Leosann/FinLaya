<div align="center">

# FinLaya

**中文金融情感决策模型 —— 用 1/3 的参数量，拿到同样的准确率**

把情感分析变成「在选项上做决策」，让模型直接输出可校准的概率，并按置信度决定是否弃权。

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Laya](https://img.shields.io/badge/Laya-0.4.15-6E56CF)](https://github.com/laya-ai/laya)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)
[![Models](https://img.shields.io/badge/Models-3%20released-blue)](weights/)

[核心发现](#核心发现) · [实验结果](#实验结果) · [快速开始](#快速开始) · [模型下载](#模型下载) · [文档](#文档)

</div>

---

## 这是什么

[Laya](https://github.com/laya-ai/laya) 是一个**决策模型**（System-1），不是自回归生成模型。它不写一段文字，而是在给定「状态 + 带选项的问题」后直接输出**选项上的概率分布**。这带来三个工程上很实在的好处：

- **概率天然可比** —— 输出就是 softmax 分布，可以直接算 ECE / Brier / NLL，也能按阈值决定"这条自动处理、那条转人工"。
- **答案空间封闭** —— 没有幻觉、没有格式漂移，评测就是准确率，不需要解析。
- **单次前向** —— 一次 forward 一次决策，延迟低且稳定。

本项目把三个中文金融情感数据集（CFSC-ABSA / FinFE / FinNSP）改造成 Laya 的决策任务，训练并对照了三个模型，并且回答了一个具体问题：

> **Laya 的 mmBERT encoder 太重了（322M），能不能换成 102M 的 FinBERT，而不掉精度？**

答案是：**能，而且不用改 laya 一行源码。**

## 核心发现

### 一、换 encoder 是官方支持的，不需要魔改

三处源码依据（`laya 0.4.15`）：

```python
# common.py:582  决策头按 encoder 的 hidden_size 动态构建
d = encoder.config.hidden_size
nhead = max(1, d // 64)
layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)

# common.py:694  build_model 用通用 AutoModel，cfg["encoder"] 接受任意 HF ID
enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")

# train_cli.py:45  --base 的 dest 就是 model_dir，接受任意路径
parser.add_argument("--base", "--model", dest="model_dir", ...)
```

唯一硬前提是 **hidden_size 一致**。mmBERT-base 与 FinBERT 都是 **768**，所以决策头形状完全相同（14.97M 参数，一个不差）。

### 二、省下来的 205M 全是多语言词表的 embedding

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| 词表大小 | 256,000 | 21,128 |
| **embedding 参数** | **196.61 M** | **16.23 M** |
| encoder 主体 | 110.4 M（22 层） | 85.0 M（12 层） |
| 决策头 | 14.97 M | **14.97 M（完全相同）** |
| **合计** | **321.9 M** | **117.2 M** |

中文金融任务用不到 256k 的多语言词表，这 196.6M 是纯浪费。

### 三、精度不掉，FinFE 严格口径还反超

| 任务 | 口径 | mmBERT-Laya (322M) | **FinBERT-Laya (117M)** | FinBERT 分类头 (102M) |
|---|---|---:|---:|---:|
| FinFE 社交情绪 | 全量 | 77.97 | 78.02 | **79.26** |
| FinFE 社交情绪 | 严格 | 77.03 | **80.28** | 79.52 |
| FinNSP 有无负面 | 全量 | 95.00 | 95.20 | **95.60** |
| FinNSP 负面主体 | 全量 | **96.91** | 95.75 | 95.37 |
| CFSC-ABSA 方面级 | 全量 | **93.37** | 93.01 | 92.78 |

**FinFE 严格口径 +3.25 pt** 是全部对比里最大的差异。原因：FinFE 评测集 54.3% 的样本与训练集文本完全重复，mmBERT-Laya 从 laya 官方 checkpoint 续训、在"见过"的样本上有记忆优势；剔除后优势消失，而 FinBERT-Laya 的决策头是从零学的，泛化更好。

### 四、部署收益

| | mmBERT-Laya | **FinBERT-Laya** | 变化 |
|---|---:|---:|---:|
| 参数量 | 321.9 M | **117.2 M** | **−63.6%** |
| 权重体积 | 614.0 MB | **223.6 MB** | **−63.6%** |
| 延迟 p50 | 18.64 ms | **8.66 ms** | **−53.5%** |
| 吞吐 (batch=1) | 53.0 QPS | **113.0 QPS** | **×2.13** |
| 显存峰值 | 1499 MB | **667 MB** | **−55.5%** |

## 三个模型

| 模型 | encoder | 决策头 | 参数量 | 定位 |
|---|---|---|---:|---|
| **FinLaya** | mmBERT 322M | Laya 原生 15M | 321.9 M | 主实验，准确率基准，唯一支持 8192 长上下文 |
| **FinBert** | FinBERT 102M | 4 个独立分类头 | 102.3 M | 判别式基线，单任务最快最准，但需 4 个模型 |
| **FinBertLaya** | FinBERT 102M | Laya 原生 15M | **117.2 M** | 换 encoder，一个模型覆盖 4 种题，延迟减半 |

## 实验结果

### 完整准确率与 Macro-F1（全量 / 严格）

严格口径 = 剔除与训练集文本 SHA256 完全相同的样本（FinFE 剔除 1,097/2,020，CFSC 剔除 442/8,312，FinNSP 剔除 1/500）。

| 任务 | mmBERT-Laya | FinBERT-Laya | FinBERT |
|---|---|---|---|
| FinFE 准确率 | 77.97 / 77.03 | 78.02 / **80.28** | **79.26** / 79.52 |
| FinFE Macro-F1 | **0.7690** / 0.7094 | 0.7629 / **0.7371** | 0.7776 / 0.7258 |
| FinNSP 有无负面 准确率 | 95.00 / 94.99 | 95.20 / 95.19 | **95.60** / **95.59** |
| FinNSP 有无负面 Macro-F1 | 0.9499 / 0.9498 | 0.9520 / 0.9518 | **0.9559** / **0.9558** |
| FinNSP 负面主体 准确率 | **96.91** | 95.75 | 95.37 |
| FinNSP 负面主体 Macro-F1（有样本 14 类） | **0.8129** | 0.7003 | 0.6838 |
| CFSC-ABSA 准确率 | **93.37** / **93.58** | 93.01 / 93.01 | 92.78 / 92.94 |
| CFSC-ABSA Macro-F1 | **0.8993** / **0.9025** | 0.8909 / 0.8905 | 0.8881 / 0.8902 |

### 概率校准（越低越好）

| 任务 | 指标 | mmBERT-Laya | FinBERT-Laya | FinBERT |
|---|---|---:|---:|---:|
| FinFE | ECE | 0.0739 | **0.0485** | 0.0947 |
| FinFE | NLL | 0.6467 | **0.5878** | 0.6244 |
| FinFE | Brier | 0.3516 | **0.3259** | 0.3285 |
| FinNSP 有无负面 | ECE | 0.0506 | 0.0882 | **0.0311** |
| FinNSP 负面主体 | ECE | 0.1984 | 0.1546 | **0.0616** |
| CFSC-ABSA | ECE | 0.0638 | 0.0607 | **0.0339** |
| CFSC-ABSA | NLL | 0.2658 | 0.2424 | **0.2157** |

### 选择性准确率（按置信度取前 X% 自动处理，全量）

| 任务 | 覆盖 | mmBERT-Laya | FinBERT-Laya | FinBERT |
|---|---|---:|---:|---:|
| FinFE | @50% | 89.50 | **91.39** | 90.99 |
| FinFE | @80% | 83.79 | **85.09** | 85.02 |
| FinFE | @90% | 81.35 | 81.96 | **82.40** |
| FinNSP 有无负面 | @50% | 99.60 | 99.60 | **100.0** |
| FinNSP 负面主体 | @50% | 99.23 | 98.46 | **99.23** |
| CFSC-ABSA | @50% | 98.65 | 99.47 | **99.83** |

严格口径下 FinFE 的 @50 更突出：**mmBERT 92.21 / FinBERT-Laya 95.02 / FinBERT 94.81**。

> 完整版（含各类别逐类召回、AURC、mean confidence、训练成本、选型决策树）见 [`docs/THREE_WAY_COMPARISON.md`](docs/THREE_WAY_COMPARISON.md)。

## 快速开始

### 安装

两套环境必须隔离，`laya` 与 `finbert` 的依赖区间不同。

```bash
# Laya 环境
conda create -y -p /data/miniconda/envs/laya python=3.11
PY=/data/miniconda/envs/laya/bin/python
$PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install "transformers>=4.48,<5" laya==0.4.15

# FinBERT 环境
bash FinBert/scripts/setup_finbert.sh
```

### 环境变量

脚本没有硬编码路径，全部走环境变量且带默认值（默认值即本文实验的服务器布局）：

| 变量 | 默认值 | 含义 |
|---|---|---|
| `LAYA_ROOT` | `/data/laya-fin` | 实验根目录 |
| `LAYA_DATA` | `${LAYA_ROOT}/laya` | 转换好的 JSONL |
| `LAYA_CKPT` | `${LAYA_ROOT}/ckpt` | mmBERT 版训练产物 |
| `FB_CKPT` | `${LAYA_ROOT}/fb_ckpt` | FinBERT-Laya 训练产物 |
| `LAYA_REPORTS` | `${LAYA_ROOT}/reports` | 评测报告 |
| `LAYA_ENV` | `/data/miniconda/envs/laya` | laya conda 环境 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |

### 数据准备

```bash
python datasets/to_laya_format.py      # 原始数据 → Laya JSONL
python datasets/validate_laya_format.py # 校验 qid 定义一致性（退出码 0 = 通过）
```

### 训练 FinLaya（mmBERT encoder）

```bash
bash FinLaya/scripts/train.sh dryrun   # 先空跑，确认 53,341 行 / 55,562 题能被解析
bash FinLaya/scripts/train.sh train    # 4 epoch
```

实测：3,068 步，loss `0.3983 → 0.3103 → 0.2565 → 0.2018`，**32 分钟**，显存 9.2 GB。

### 训练 FinBertLaya（FinBERT encoder）

```bash
# 1) 造 base checkpoint：FinBERT encoder + 随机初始化决策头（约 1 分钟）
python FinBertLaya/scripts/make_fb_base.py

# 2) 训练（超参与 mmBERT 版逐项一致）
bash FinBertLaya/scripts/train.sh
```

实测：3,068 步，**23 分钟**，显存 ~3.8 GB。

### 训练 FinBert（分类头基线）

```bash
ENV_PY=/data/miniconda/envs/finbert/bin/python
$ENV_PY FinBert/scripts/finbert_baseline.py train
```

实测：`absa` 218s · `finfe` 63s · `nsp_has` 28s · `nsp_subject` 16s，合计约 5.4 分钟。

### 评测

```bash
# FinLaya
bash FinLaya/scripts/eval_ft.sh
bash FinLaya/scripts/eval_ft.sh finfe_eval              # 只跑一个

# FinBertLaya
bash FinBertLaya/scripts/eval.sh

# FinBert（必须分批，一次跑全部会被 OOM killer 杀掉且不留 Traceback）
$ENV_PY FinBert/scripts/fb_eval.py absa
$ENV_PY FinBert/scripts/fb_eval.py finfe
```

复算全套指标（Macro-F1 / 各类召回 / NLL / Brier / 选择性准确率 / 严格口径）：

```bash
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl --strict
```

> **注意**：`laya_metrics.py` 自己会写 `<report>_metrics.json`，**不要**把它的 stdout 重定向到那两个文件 —— 会把刚写好的 JSON 覆盖成人类可读文本。

### 部署指标

三个模型的测量必须分进程跑（依赖不同 env 的 torch）：

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
```

## 模型下载

**FinBERT-Laya（117.2M）** 已随仓库提供，分 3 片存放（超过 GitHub 单文件 100MB 限制）：

```bash
cd weights/finbert-laya
bash restore.sh        # 合并 + SHA256 校验
```

合并后即为 `laya.load()` 认的标准布局，可直接 `Router.register(name, path)`。详见 [`weights/finbert-laya/README.md`](weights/finbert-laya/README.md)。

**FinBERT 四个分类头**（fp32 各 391 MB，合计 1.56 GB）不入库，但重训只要 5.4 分钟，命令见上文。

## 数据集

三个数据集均来自 **[CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)**（中文金融语言理解评测基准）。

| 数据集 | 任务 | 原始文件 | 体积 | 条数 |
|---|---|---|---:|---:|
| **CFSC-ABSA** | 方面级情感分析 | `train.txt` / `test.txt` | 14.90 + 3.72 MB | 33,184 / 8,312 |
| **FinFE** | 金融社交文本情绪 | `train_list.json` / `eval_list.json` / `problem_list.json` | 1.09 + 0.14 + 0.17 MB | 16,157 / 2,020 / 2,020 |
| **FinNSP** | 金融负面主体识别 | `train_list.json` / `eval_list.json` / `problem_list.json` | 3.48 + 0.46 + 0.46 MB | 4,000 / 500 / 500 |

合计 **24.42 MB / 8 个文件**，逐个 SHA256 校验通过（清单见 [`datasets/README.md`](datasets/README.md)）。数据本身不入库，用 `datasets/to_laya_format.py` 重新生成。

> `problem_list.json` 是无标签测试输入，不参与本项目评测；评测用 `eval_list.json`。
>
> **FinFE 的 54.3% 重叠**是原数据集本身的划分问题，不是转换引入的。所以所有结论都同时给出全量与严格口径。

## 处理后的样例

转换产物是 9 个 JSONL（120,034 行 / 125,235 题）。三种任务的实际样子：

**FinFE —— 三分类 choice**

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

**CFSC-ABSA —— state 里带方面词**

```json
{"state": "方面词：长谷投资\n语句：问董秘：请问长谷投资是否是你旗下的公司？【长谷投资】",
 "questions": {"sentiment_absa": {
   "type": "choice",
   "instructions": "句中针对指定方面词的情感倾向是什么？",
   "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}},
 "expected": {"sentiment_absa": "neutral"},
 "meta": {"source": "CFSC-ABSA"}}
```

**FinNSP —— 一行两题，choice + noul**

```json
{"state": "候选实体：恒昌汇财；纳觅财务\n文本：【网传恒昌汇财被查，投资者恐慌】...",
 "questions": {
   "neg_has":     {"type": "noul", "instructions": "文本中是否提及了负面主体？"},
   "neg_subject": {"type": "choice", "instructions": "文本中的负面主体是哪个？",
                   "criteria": {"以上都不是": "...", "恒昌汇财": "...", "...": "..."}}},
 "expected": {"neg_has": true, "neg_subject": "以上都不是"},
 "meta": {"source": "FinNSP"}}
```

> **关键约束**：同一 `qid` 的 `questions` 定义必须跨行完全一致。`laya/train.py` 遇到冲突只 `warnings.warn` 并保留第一条，后续定义被静默丢弃 —— 这是最容易踩的坑。`validate_laya_format.py` 就是为守住这条而写的。

## 项目结构

```
FinLaya/          mmBERT encoder + Laya 决策头（322M）主实验
  ├─ scripts/     train.sh · baseline.sh · eval_ft.sh
  ├─ configs/     rl_agent_config.json · questions.json · train_report.json
  ├─ logs/        train_run · dryrun · baseline · eval_ft
  └─ reports/     eval/（ft_*）· baseline/（base_*）
FinBertLaya/      FinBERT encoder + Laya 决策头（117M）
  ├─ scripts/     make_fb_base.py · train.sh · eval.sh
  ├─ configs/     fb_rl_agent_config.json · fb_train_report.json
  └─ reports/     fb_*
FinBert/          FinBERT encoder + 四任务分类头（102M）
  ├─ scripts/     setup_finbert.sh · finbert_baseline.py · fb_eval.py
  └─ reports/     finbert_all.json
common/           三模型共用
  ├─ laya/        run_eval.py · laya_metrics.py · check_leak.py · strict_acc.py
  ├─ deploy_metrics.py · to_fp16.py · make_compare.py
  └─ reports/     deploy_metrics.json
datasets/         数据下载说明 + 转换/校验脚本
weights/          已训练的 FinBERT-Laya 权重（分片 + 合并脚本）
docs/             三份对照报告
```

## 文档

| 文档 | 内容 |
|---|---|
| [`FinLaya/REPORT.md`](FinLaya/REPORT.md) | Laya 单模型主实验：结论、数据、训练过程、踩坑、数字速查 |
| [`docs/COMPARE_Laya_vs_FinBERT.md`](docs/COMPARE_Laya_vs_FinBERT.md) | Laya vs FinBERT 分类头两方对照 |
| [`docs/FINBERT_AS_ENCODER.md`](docs/FINBERT_AS_ENCODER.md) | 换 encoder 专题：源码依据、做法、三方对比、局限 |
| [`docs/THREE_WAY_COMPARISON.md`](docs/THREE_WAY_COMPARISON.md) | 三模型全指标对照：含逐类召回、AURC、训练成本、选型决策树 |

## 已知局限

1. **FinFE 全量口径有 54.3% 训练集重叠**，77.97 / 78.02 / 79.26 不能当泛化能力引用，请以严格口径为准。
2. **FinNSP 负面主体长尾严重**。259 条里 102 条是「以上都不是」，剩 157 条散在 13 个实体，多数只有 1~7 条。`0.8129 / 0.7003 / 0.6838` 的差距几乎全来自「钱宝」（4 条）与「摩尔龙」（1 条）。
3. **`noul` 温度在 34 条样本上拟合**（`train_report.json` 带警告），`neg_has` 的 ECE 可信度低于其他任务。
4. **FinBERT-Laya 的 512 截断**。FinNSP 有 5.00% 训练样本超长（p99=1,851，max=6,777），尾部信息会丢。这是 BERT 与 mmBERT（8192）的架构性差距。
5. **两个 Laya 的起点不对等**。FinBERT-Laya 决策头随机初始化（loss 1.3957），mmBERT-Laya 从官方 checkpoint 续训（0.3983）。只能比终点，不能比收敛速度。
6. **单 seed**。1 pt 以内的差距不显著；FinFE 严格口径的 3.25 pt 差距较大，仍建议多 seed 复核。

## 引用

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

代码 MIT。数据集遵循 CFLEB 原始协议，详见 [`datasets/README.md`](datasets/README.md)。
