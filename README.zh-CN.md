<div align="center">

<img src="assets/logo-wordmark-260.png" alt="FinLaya" height="72" align="middle" />

# FinLaya：中文金融情感决策模型 —— 用 1/3 的参数量，拿到同等精度

把情感分析重构为「在闭合选项集上的决策」，模型直接输出可校准的概率，并可按置信度决定是否弃权。

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Models-FFD21E)](https://huggingface.co/zhanghanshuo/FinLaya)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Laya](https://img.shields.io/badge/Laya-0.4.15-6E56CF)](https://github.com/laya-ai/laya)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)

[English](README.md) · [模型权重](https://huggingface.co/zhanghanshuo/FinLaya) · [关键结论](#关键结论) · [实验结果](#实验结果) · [快速开始](#快速开始) · [文档](#文档)

</div>

---

## 项目概述

[Laya](https://github.com/laya-ai/laya) 是一类**决策模型**（System-1），与自回归生成模型的核心区别在于输出形态：给定「状态 + 带选项的问题」，它直接给出**选项上的概率分布**，而不生成自由文本。这一形态在工程上带来三项可验证的性质：

- **概率可校准** —— 输出即 softmax 分布，可直接计算 ECE / Brier / NLL，也可按阈值切分「自动处理 / 转人工」。
- **输出空间封闭** —— 无幻觉、无格式漂移，评测指标即准确率，无需解析后处理。
- **单次前向** —— 一次 forward 完成一次决策，延迟低且稳定。

本项目将三个中文金融情感数据集（CFSC-ABSA / FinFE / FinNSP）重构为 Laya 决策任务，训练并对三个模型做了统一口径的对照，旨在回答一个具体问题：

> **Laya 默认的 mmBERT encoder 参数量偏大（322M），能否替换为 102M 的 FinBERT 而不损失精度？**

**结论：可以，且无需修改 laya 任何源码。**

### 模型命名速查

`FinLaya` 是**系列名**，指两个共用同一种决策头、只有 encoder 不同的 Laya 模型：

| 本仓库内的名字 | 目录 | encoder | 参数量 | 也叫 |
|---|---|---|---:|---|
| **FinLaya-mmBERT** | `FinLaya-mmBERT/` | `jhu-clsp/mmBERT-base` | 321.9 M | 主实验、准确率基准 |
| **FinLaya-FinBERT** | `FinLaya-FinBERT/` | `yiyanghkust/finbert-tone-chinese` | 117.2 M | encoder 替换方案，**Hugging Face 上的那个** |

两个容易混淆的点：

- Hugging Face 仓库叫 [`zhanghanshuo/FinLaya`](https://huggingface.co/zhanghanshuo/FinLaya)，但里面放的是 **FinLaya-FinBERT**，不是 mmBERT 版。仓库名早于拆分成两个变体。
- 第三个模型 **FinBERT**（`FinBert/`）是完全不同的架构——四个独立分类头，没有 Laya 决策头。它是判别式基线，不是 FinLaya 的变体。

## 关键结论

### 一、encoder 替换属于框架原生能力，非侵入式改造

`laya 0.4.15` 的三处源码依据：

```python
# common.py:582  决策头按 encoder 的 hidden_size 动态构建
d = encoder.config.hidden_size
nhead = max(1, d // 64)
layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout, batch_first=True, norm_first=True)

# common.py:694  build_model 使用通用 AutoModel，cfg["encoder"] 接受任意 HF ID
enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")

# train_cli.py:45  --base 的 dest 即 model_dir，接受任意路径
parser.add_argument("--base", "--model", dest="model_dir", ...)
```

唯一硬性前提是 **hidden_size 一致**。mmBERT-base 与 FinBERT 均为 **768**，因此决策头结构完全相同（14.97M 参数，逐项一致）。

### 二、参数量差异几乎全部来自多语言词表 embedding

| | FinLaya-mmBERT | FinLaya-FinBERT |
|---|---:|---:|
| 词表大小 | 256,000 | 21,128 |
| **embedding 参数** | **196.61 M** | **16.23 M** |
| encoder 主体 | 110.4 M（22 层） | 85.0 M（12 层） |
| 决策头 | 14.97 M | **14.97 M（完全相同）** |
| **合计** | **321.9 M** | **117.2 M** |

中文金融任务不涉及 256k 规模的多语言覆盖，这部分 embedding 容量为结构性冗余。

### 三、精度未退化，FinFE 严格口径下反超

| 任务 | 口径 | FinLaya-mmBERT (322M) | **FinLaya-FinBERT (117M)** | FinBERT 分类头 (102M) |
|---|---|---:|---:|---:|
| FinFE 社交情绪 | 全量 | 77.97 | 78.02 | **79.26** |
| FinFE 社交情绪 | 严格 | 77.03 | **80.28** | 79.52 |
| FinNSP 有无负面 | 全量 | 95.00 | 95.20 | **95.60** |
| FinNSP 负面主体 | 全量 | **96.91** | 95.75 | 95.37 |
| CFSC-ABSA 方面级 | 全量 | **93.37** | 93.01 | 92.78 |

**FinFE 严格口径 +3.25 pt** 是三组对比中差异最大的一项。归因：FinFE 评测集有 54.3% 的样本与训练集文本完全重复，FinLaya-mmBERT 从 laya 官方 checkpoint 续训，在已见样本上具备记忆优势；剔除重叠后该优势消失，而 FinLaya-FinBERT 的决策头为随机初始化、从零学习，泛化表现更优。

### 四、部署侧收益

| | FinLaya-mmBERT | **FinLaya-FinBERT** | 变化 |
|---|---:|---:|---:|
| 参数量 | 321.9 M | **117.2 M** | **−63.6%** |
| 权重体积 | 614.0 MB | **223.6 MB** | **−63.6%** |
| 延迟 p50 | 18.64 ms | **8.66 ms** | **−53.5%** |
| 吞吐 (batch=1) | 53.0 QPS | **113.0 QPS** | **×2.13** |
| 显存峰值 | 1499 MB | **667 MB** | **−55.5%** |

## 三个模型

| 模型 | 目录 | encoder | 决策头 | 参数量 | 定位 |
|---|---|---|---|---:|---|
| **FinLaya-mmBERT** | `FinLaya-mmBERT/` | mmBERT 322M | Laya 原生 15M | 321.9 M | 主实验，准确率基准；唯一架构上支持长上下文的（训练用 1024 token，上限 8192） |
| **FinLaya-FinBERT** | `FinLaya-FinBERT/` | FinBERT 102M | Laya 原生 15M | **117.2 M** | encoder 替换方案，单模型覆盖 4 种题型，延迟减半 |
| **FinBERT** | `FinBert/` | FinBERT 102M | 4 个独立分类头 | 102.3 M | 判别式基线，单任务最快最准，但需维护 4 个模型 |

> **命名说明。** 单独说 `FinLaya` 指的是这个模型系列，不是某个具体模型。两个 Laya 模型的唯一区别是 encoder——看结果表时请对准这一列。另外 Hugging Face 仓库 [`zhanghanshuo/FinLaya`](https://huggingface.co/zhanghanshuo/FinLaya) 里放的是 **FinLaya-FinBERT**（117.2 M），仓库名早于拆分成两个变体。

## 实验结果

### 准确率与 Macro-F1（全量 / 严格）

严格口径 = 剔除与训练集文本 SHA256 完全相同的样本（FinFE 剔除 1,097/2,020，CFSC 剔除 442/8,312，FinNSP 剔除 1/500）。

| 任务 | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
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

| 任务 | 指标 | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | ECE | 0.0739 | **0.0485** | 0.0947 |
| FinFE | NLL | 0.6467 | **0.5878** | 0.6244 |
| FinFE | Brier | 0.3516 | **0.3259** | 0.3285 |
| FinNSP 有无负面 | ECE | 0.0506 | 0.0882 | **0.0311** |
| FinNSP 负面主体 | ECE | 0.1984 | 0.1546 | **0.0616** |
| CFSC-ABSA | ECE | 0.0638 | 0.0607 | **0.0339** |
| CFSC-ABSA | NLL | 0.2658 | 0.2424 | **0.2157** |

### 选择性准确率（按置信度取前 X% 自动处理，全量）

| 任务 | 覆盖 | FinLaya-mmBERT | FinLaya-FinBERT | FinBERT |
|---|---|---:|---:|---:|
| FinFE | @50% | 89.50 | **91.39** | 90.99 |
| FinFE | @80% | 83.79 | **85.09** | 85.02 |
| FinFE | @90% | 81.35 | 81.96 | **82.40** |
| FinNSP 有无负面 | @50% | 99.60 | 99.60 | **100.0** |
| FinNSP 负面主体 | @50% | 99.23 | 98.46 | **99.23** |
| CFSC-ABSA | @50% | 98.65 | 99.47 | **99.83** |

严格口径下 FinFE 的 @50 差距更为明显：**mmBERT 92.21 / FinLaya-FinBERT 95.02 / FinBERT 94.81**。

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

### 数据准备

```bash
python datasets/to_laya_format.py       # 原始数据 → Laya JSONL
python datasets/validate_laya_format.py # 校验 qid 定义一致性（退出码 0 = 通过）
```

### 训练 FinLaya（mmBERT encoder）

```bash
bash FinLaya-mmBERT/scripts/train.sh dryrun   # 先空跑，确认 53,341 行 / 55,562 题能被解析
bash FinLaya-mmBERT/scripts/train.sh train    # 4 epoch
```

### 训练 FinLaya-FinBERT（FinBERT encoder）

```bash
# 1) 构建 base checkpoint：FinBERT encoder + 随机初始化决策头（约 1 分钟）
python FinLaya-FinBERT/scripts/make_fb_base.py

# 2) 训练（超参与 mmBERT 版逐项一致）
bash FinLaya-FinBERT/scripts/train.sh
```

### 训练 FinBert（分类头基线）

```bash
ENV_PY=/data/miniconda/envs/finbert/bin/python
$ENV_PY FinBert/scripts/finbert_baseline.py train
```

### 评测

```bash
# FinLaya
bash FinLaya-mmBERT/scripts/eval_ft.sh
bash FinLaya-mmBERT/scripts/eval_ft.sh finfe_eval              # 只跑一个

# FinLaya-FinBERT
bash FinLaya-FinBERT/scripts/eval.sh

# FinBert（必须分批，一次跑全部会被 OOM killer 杀掉且不留 Traceback）
$ENV_PY FinBert/scripts/fb_eval.py absa
$ENV_PY FinBert/scripts/fb_eval.py finfe
```

复算全套指标（Macro-F1 / 各类召回 / NLL / Brier / 选择性准确率 / 严格口径）：

```bash
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl
python common/laya/laya_metrics.py reports/ft_finfe_eval.json $LAYA_DATA/finfe_eval.jsonl --strict
```

> **注意**：`laya_metrics.py` 自身会写 `<report>_metrics.json`，**不要**把它的 stdout 重定向到同名文件 —— 会把刚写好的 JSON 覆盖成人类可读文本。

### 部署指标

三个模型的测量必须分进程执行（依赖不同 env 的 torch）：

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
```

## 模型下载

### Hugging Face（推荐）

完整权重已发布：[**huggingface.co/zhanghanshuo/FinLaya**](https://huggingface.co/zhanghanshuo/FinLaya)

```bash
pip install -U huggingface_hub
huggingface-cli download zhanghanshuo/FinLaya --local-dir weights/finlaya-finbert-hf
```

国内网络可加镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download zhanghanshuo/FinLaya --local-dir weights/finlaya-finbert-hf
```

下载目录即为 `laya.load()` 认的标准布局，可直接 `Router.register(name, path)`。

> **如果下载下来是空的**，说明 Hub 仓库里的文件目前多套了一层 `hf-ready/` 前缀 —— 上次上传漏了 `path_in_repo` 参数。在修复之前先按显式路径取：
>
> ```bash
> huggingface-cli download zhanghanshuo/FinLaya --include "hf-ready/*" --local-dir weights/finlaya-finbert-hf
> mv weights/finlaya-finbert-hf/hf-ready/* weights/finlaya-finbert-hf/ && rmdir weights/finlaya-finbert-hf/hf-ready
> ```
>
> 一步修复见 [`weights/HF_UPLOAD_GUIDE.md`](weights/HF_UPLOAD_GUIDE.md)。

### 本仓库内（GitHub）

**FinLaya-FinBERT（117.2M）** 同时随仓库提供，分 3 片存放（超过 GitHub 单文件 100 MB 限制）：

```bash
cd weights/finlaya-finbert
bash restore.sh        # 合并 + SHA256 校验
```

合并后布局与 HF 版本一致。详见 [`weights/finlaya-finbert/README.md`](weights/finlaya-finbert/README.md)。

**FinBERT 四个分类头**（fp32 各 391 MB，合计 1.56 GB）不入库，重新训练即可获得，命令见上文。

## 数据集

三个数据集均来自 **[CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)**（中文金融语言理解评测基准）。

| 数据集 | 任务 | 原始文件 | 体积 | 条数 |
|---|---|---|---:|---:|
| **CFSC-ABSA** | 方面级情感分析 | `train.txt` / `test.txt` | 14.90 + 3.72 MB | 33,184 / 8,312 |
| **FinFE** | 金融社交文本情绪 | `train_list.json` / `eval_list.json` / `problem_list.json` | 1.09 + 0.14 + 0.17 MB | 16,157 / 2,020 / 2,020 |
| **FinNSP** | 金融负面主体识别 | `train_list.json` / `eval_list.json` / `problem_list.json` | 3.48 + 0.46 + 0.46 MB | 4,000 / 500 / 500 |

合计 **24.42 MB / 8 个文件**，逐个 SHA256 校验通过（清单见 [`datasets/README.md`](datasets/README.md)）。数据本身不入库，用 `datasets/to_laya_format.py` 重新生成。

> `problem_list.json` 为无标签测试输入，不参与本项目评测；评测统一使用 `eval_list.json`。
>
> **FinFE 的 54.3% 重叠**源于原数据集自身的划分问题，非转换引入。因此所有结论同时给出全量与严格两种口径。

## 处理后的样例

转换产物为 9 个 JSONL（120,034 行 / 125,235 题）。三种任务的实际形态：

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

**CFSC-ABSA —— state 中带方面词**

```json
{"state": "方面词：长谷投资\n语句：问董秘：请问长谷投资是否是你旗下的公司？【长谷投资】",
 "questions": {"sentiment_absa": {
   "type": "choice",
   "instructions": "句中针对指定方面词的情感倾向是什么？",
   "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}},
 "expected": {"sentiment_absa": "neutral"},
 "meta": {"source": "CFSC-ABSA"}}
```

**FinNSP —— 单行两题，choice + noul**

```json
{"state": "候选实体：恒昌汇财；纳觅财务\n文本：【网传恒昌汇财被查，投资者恐慌】...",
 "questions": {
   "neg_has":     {"type": "noul", "instructions": "文本中是否提及了负面主体？"},
   "neg_subject": {"type": "choice", "instructions": "文本中的负面主体是哪个？",
                   "criteria": {"以上都不是": "...", "恒昌汇财": "...", "...": "..."}}},
 "expected": {"neg_has": true, "neg_subject": "以上都不是"},
 "meta": {"source": "FinNSP"}}
```

> **关键约束**：同一 `qid` 的 `questions` 定义必须跨行完全一致。`laya/train.py` 遇到冲突仅 `warnings.warn` 并保留第一条，后续定义被静默丢弃 —— 这是最易踩的坑。`validate_laya_format.py` 即用于守住这条约束。

## 项目结构

```
FinLaya-mmBERT/       mmBERT encoder + Laya 决策头（322M）主实验
  ├─ scripts/     train.sh · baseline.sh · eval_ft.sh
  ├─ configs/     rl_agent_config.json · questions.json · train_report.json
  ├─ logs/        train_run · dryrun · baseline · eval_ft
  └─ reports/     eval/（ft_*）· baseline/（base_*）
FinLaya-FinBERT/      FinBERT encoder + Laya 决策头（117M）已发布的权重
  ├─ scripts/     make_fb_base.py · train.sh · eval.sh
  ├─ configs/     fb_rl_agent_config.json · fb_train_report.json
  └─ reports/     fb_*
FinBert/              FinBERT encoder + 四任务分类头（102M）判别式基线
  ├─ scripts/     setup_finbert.sh · finbert_baseline.py · fb_eval.py
  └─ reports/     finbert_all.json
common/           三模型共用
  ├─ laya/        run_eval.py · laya_metrics.py · check_leak.py · strict_acc.py
  ├─ deploy_metrics.py · to_fp16.py · make_compare.py
  └─ reports/     deploy_metrics.json
datasets/         数据下载说明 + 转换/校验脚本
weights/          已训练的 FinLaya-FinBERT 权重（分片 + 合并脚本）
docs/             三份对照报告
```

## 文档

| 文档 | 内容 |
|---|---|
| [`FinLaya-mmBERT/REPORT.md`](FinLaya-mmBERT/REPORT.md) | Laya 单模型主实验：结论、数据、训练过程、踩坑、数字速查 |
| [`docs/COMPARE_Laya_vs_FinBERT.md`](docs/COMPARE_Laya_vs_FinBERT.md) | Laya vs FinBERT 分类头两方对照 |
| [`docs/FINBERT_AS_ENCODER.md`](docs/FINBERT_AS_ENCODER.md) | encoder 替换专题：源码依据、做法、三方对比、局限 |
| [`docs/THREE_WAY_COMPARISON.md`](docs/THREE_WAY_COMPARISON.md) | 三模型全指标对照：含逐类召回、AURC、训练成本、选型决策树 |

## 已知局限

1. **FinFE 全量口径存在 54.3% 训练集重叠**。77.97 / 78.02 / 79.26 不宜作为泛化能力引用，请以严格口径为准。
2. **FinNSP 负面主体任务长尾严重**。259 条中 102 条为「以上都不是」，其余 157 条分散在 13 个实体上，多数仅有 1~7 条。`0.8129 / 0.7003 / 0.6838` 的差距几乎全部来自「钱宝」（4 条）与「摩尔龙」（1 条）。
3. **`noul` 温度在 34 条样本上拟合**（`train_report.json` 带警告），`neg_has` 的 ECE 可信度低于其他任务。
4. **FinLaya-FinBERT 存在 512 截断**。FinNSP 有 5.00% 训练样本超长（p99=1,851，max=6,777），尾部信息会丢失。这是 BERT 与 mmBERT（8192）的架构性差距。
5. **两个 Laya 的起点不对等**。FinLaya-FinBERT 决策头随机初始化（loss 1.3957），FinLaya-mmBERT 从官方 checkpoint 续训（0.3983）。只能比较终点，不能比较收敛速度。
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
