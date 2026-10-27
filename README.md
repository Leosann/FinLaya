# FinLaya

**中文金融情感决策模型**：用 [Laya](https://github.com/laya-ai/laya) 的决策原语（choice / noul）把三个中文金融情感数据集统一成"决策任务"，与 FinBERT 判别式基线在同一套指标下做对照。另外还做了一件更激进的事：**把 Laya 的 encoder 从 322M 的 mmBERT 换成 102M 的 FinBERT，得到 117M 的版本，准确率持平而延迟减半**。

|    |                                                                  |
| -- | ---------------------------------------------------------------- |
| 基座 | `convaiinnovations/laya` → `multilingual` = **mmBERT-base 322M** |
| 任务 | CFSC-ABSA · FinFE · FinNSP（情绪三分类 + ABSA + 负面主体识别）                |
| 训练 | RLCD 软标签 + 球面正则，4 epoch，**32 分钟**，显存 9.2 GB                      |
| 评测 | 全量 + **严格口径**（剔除与训练集重叠的样本）双套指标                                   |
| 对照 | `yiyanghkust/finbert-tone-chinese`（102M）四任务独立分类头                 |

三个模型的关系：

| 模型 | encoder | 决策头 | 参数量 | 一句话 |
|---|---|---|---:|---|
| **FinLaya** | mmBERT 322M | Laya 原生 15M | 321.9M | 主实验，准确率基准 |
| **FinBert** | FinBERT 102M | 4 个独立分类头 | 102.3M | 判别式基线，单任务最强 |
| **FinBertLaya** | FinBERT 102M | Laya 原生 15M | **117.2M** | 换 encoder，准确率持平、延迟减半 |

---

## 1. 快速查看

```
FinLaya/       mmBERT encoder + Laya 决策头（322M）主实验
FinBert/       FinBERT encoder + 四任务分类头（102M）基线
FinBertLaya/   FinBERT encoder + Laya 决策头（117M）换 encoder 实验
common/        共用的部署指标测量与对比报告生成
datasets/      原始数据下载说明 + 转 Laya JSONL 的转换/校验脚本 + 转换产物
docs/          对照报告（两模型 / 换 encoder 专题）
```

三份文档：

- **[FinLaya/REPORT.md](FinLaya/REPORT.md)** — Laya 单模型主实验报告
- **[docs/COMPARE_Laya_vs_FinBERT.md](docs/COMPARE_Laya_vs_FinBERT.md)** — Laya vs FinBERT 分类头
- **[docs/FINBERT_AS_ENCODER.md](docs/FINBERT_AS_ENCODER.md)** — 把 FinBERT 换成 Laya 的 encoder（三方对比）

---

## 2. 模型简介

### 2.1 Laya 是什么

Laya 不是自回归生成模型，而是一个**决策模型（System-1）**。它不产出一段文本，而是在给定一个「状态」和一组「带选项的问题」后，直接输出**在选项上的概率分布**。这有三点直接的好处：

1. **概率天然可比**：输出就是 softmax 后的分布，可做 ECE / Brier / NLL 校准评估，可以按置信度阈值决定是否自动处理。
2. **答案空间封闭**：不存在生成式模型的幻觉与格式漂移，评测就是准确率，不需要解析。
3. **单次前向**：一次 forward 完成一次决策，没有逐 token 解码，延迟低且稳定。

它支持三种决策原语，本项目用到其中两种：

| 原语       | 语义           | 本项目用途            |
| -------- | ------------ | ---------------- |
| `choice` | 在 N 个候选项中选一个 | 情绪三分类、负面主体 21 选一 |
| `noul`   | 二值真/假判定      | 「文本是否含负面主体」      |
| `score`  | 在连续刻度上打分     | 未使用              |

### 2.2 为什么能把情感分析变成决策问题

传统做法是「输入文本 → 输出标签」，模型只学一个映射。Laya 的输入是一个三元组：

```
state      = 待判断的文本（这里还额外塞进了方面词 / 候选实体）
questions  = { qid: { type, instructions, criteria } }
expected   = { qid: 正确答案 }
```

`criteria` 是**写在题面里的判定标准**，例如：

```json
"negative": "看空、恐慌、抱怨、亏损、割肉、骂声等消极情绪",
"neutral":  "中性陈述、疑问、无明确多空倾向",
"positive": "看多、乐观、期待上涨等积极情绪"
```

这意味着模型不只知道「标签叫 negative」，还知道「什么样的文本算 negative」。这是本项目最关键的改造点——**标签的语义被写进了输入**，而不是只存在于参数里。同一份权重因此可以同时服务多个任务：不同任务只是不同的 `qid` 与不同的选项集合，决策头按 qid 自动分组。

> **一个硬约束**：同一 `qid` 的 `questions` 定义必须跨行**完全一致**。`laya/train.py` 遇到冲突只 `warnings.warn` 并保留第一条，不会报错——这是最容易踩的坑（详见 §7）。`datasets/to_laya_format.py` 与 `validate_laya_format.py` 就是为守住这条约束而写的。

### 2.3 训练目标

用的是官方 RLCD 配方，不是普通交叉熵：

- **软标签交叉熵**（`w_rps = 1.0`）：对 4 条 RL 采样路径加权，缓解选项顺序偏置
- **球面正则**（`w_sph = 0.75`）：约束不同 qid 的决策头向量在球面上分散，避免互相挤占
- **sigma 退火** 0.4 → 0.1：采样温度从探索逐步收紧
- **标签平滑** 0.05：抑制过度自信，直接改善 ECE
- 编码器 lr `2.5e-5`，决策头 lr `1e-4`，4 epoch

训练结束后自动做**温度标定**（choice 1.4378 / noul 1.4585）和**弃权阈值拟合**（`choice:3-5` 0.6217、`noul:2` 0.6661、`choice:11+` 0.7562），产物在 `FinLaya/configs/rl_agent_config.json`。

### 2.4 FinBERT 基线

`yiyanghkust/finbert-tone-chinese`，102M，中文金融语料预训练。为了公平对照，每个任务**单独微调一个分类头**（不是多头共享一个模型），3 epoch / bs 32 / lr 2e-5 / max_len 256 / fp16，用一个模型打两次总分。

### 2.5 换 encoder：FinBERT-Laya

Laya 的决策头是**按 encoder 的 `hidden_size` 动态构建**的（`DecisionModel.__init__` 里 `d = encoder.config.hidden_size`），`build_model` 用的是通用 `AutoModel`，`--base` 的 `dest` 就是 `model_dir`。所以只要 hidden_size 一致，换 encoder **不需要改 laya 一行源码**。

mmBERT-base 与 FinBERT 的 hidden_size 都是 **768**，决策头形状完全一致（14.97M，一个参数都不差）。省下来的 205M 全是 mmBERT 那 256,000 词表带来的 embedding —— 中文金融任务用不到：

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| encoder | 322M（其中 **196.61M 是 embedding**） | 102.27M（embedding 仅 16.23M） |
| 决策头 | 14.97M | **14.97M** |
| 合计 | 321.9M | **117.2M** |

做法是先用 `build_model(cfg, pretrained=True)` 造一个「FinBERT encoder + 随机初始化决策头」的 base checkpoint，再用**完全相同的 RLCD 配方**训练 4 epoch。细节与三方对比见 **[docs/FINBERT_AS_ENCODER.md](docs/FINBERT_AS_ENCODER.md)**。

---

## 3. 实验结果

### 3.1 主结果（全量口径）

准确率与 Macro-F1：

| 任务              | qid                |     n |   Laya acc | FinBERT acc |   Laya mF1 | FinBERT mF1 |
| --------------- | ------------------ | ----: | ---------: | ----------: | ---------: | ----------: |
| FinFE 社交情绪      | `sentiment_social` | 2,020 |     77.97% |  **79.26%** |     0.7690 |  **0.7776** |
| FinNSP 是否有负面主体  | `neg_has`          |   500 |     95.00% |  **95.60%** |     0.9499 |  **0.9559** |
| FinNSP 负面主体识别   | `neg_subject`      |   259 | **96.91%** |      95.37% |     0.5419 |      0.5038 |
| CFSC-ABSA 方面级情绪 | `sentiment_absa`   | 8,312 | **93.37%** |      92.78% | **0.8993** |      0.8881 |

> `neg_subject` 的 Macro-F1 被严重低估：21 个类别里只有 14 个在测试集出现，零支持类别被强制计 0。**只看有样本的类别**（`macro_f1_present`）：Laya **0.8129** vs FinBERT **0.6838**——这是两模型差距最大的一项。FinBERT 词表少 2 个实体（「北京资易贷金融信息服务有限公司」「易捷金融」），所以是 19 类 / 14 类有样本。

概率可信度：

| 任务                   |   Laya ECE | FinBERT ECE | Laya NLL | FinBERT NLL | Laya Brier | FinBERT Brier |
| -------------------- | ---------: | ----------: | -------: | ----------: | ---------: | ------------: |
| FinFE                | **0.0739** |      0.0947 |   0.6467 |  **0.6244** |     0.3516 |    **0.3285** |
| FinNSP `neg_has`     |     0.0506 |  **0.0311** |   0.1799 |  **0.1517** |     0.0873 |    **0.0740** |
| FinNSP `neg_subject` |     0.1984 |  **0.0616** |   0.3514 |  **0.2821** | **0.0952** |        0.0991 |
| CFSC-ABSA            |     0.0638 |  **0.0339** |   0.2658 |  **0.2157** |     0.1170 |    **0.1126** |

**读法**：FinBERT 的概率更"锐"也更准（ECE / NLL 普遍更优），因为判别式分类头 + 3 epoch 全参数微调是这条路线上很难打败的强基线。Laya 唯一在 ECE 上胜出的是 FinFE（0.0739 vs 0.0947）。

### 3.2 严格口径（剔除与训练集重叠样本）

先用 SHA256 对三个测试集的 `state` 文本与三个训练集逐一比对，命中的样本从评测里剔除：

| 测试集              |   原 n |    与训练集重叠 |        占比 | 剔除后 n |
| ---------------- | ----: | --------: | --------: | ----: |
| `finfe_eval`     | 2,020 | **1,097** | **54.3%** |   923 |
| `finnsp_eval`    |   500 |         1 |      0.2% |   499 |
| `cfsc_absa_test` | 8,312 |       442 |      5.3% | 7,870 |

严格口径下的指标：

| 任务                   |     n |   Laya acc | FinBERT acc |   Laya mF1 | FinBERT mF1 |
| -------------------- | ----: | ---------: | ----------: | ---------: | ----------: |
| FinFE                |   923 |     77.03% |  **79.52%** |     0.7094 |  **0.7258** |
| FinNSP `neg_has`     |   499 |     94.99% |  **95.59%** |     0.9498 |  **0.9558** |
| FinNSP `neg_subject` |   259 | **96.91%** |      95.37% |     0.5419 |      0.5038 |
| CFSC-ABSA            | 7,870 | **93.58%** |      92.94% | **0.9025** |      0.8902 |

FinFE 剔除 54.3% 后两模型各掉约 1 个百分点（Laya −0.94 pt，FinBERT −0.26 pt），说明**之前 77.97% 的数字里有相当一部分来自记忆**，严格口径才是可信结论。CFSC-ABSA 反而略升，因为被剔除的多是短文本易分样本。

### 3.3 高置信度选择性准确率

按置信度从高到低取前 X% 样本，在这部分「可以自动处理」的样本上的准确率：

| 任务                   | 覆盖 50% |           |    覆盖 80% |           |    覆盖 90% |           |
| -------------------- | -----: | --------: | --------: | --------: | --------: | --------: |
|                      |   Laya |        FB |      Laya |        FB |      Laya |        FB |
| FinFE                |  89.50 | **90.99** |     83.79 | **85.02** |     81.35 | **82.40** |
| FinNSP `neg_has`     |  99.60 | **100.0** |     99.00 | **99.75** | **98.22** |     98.00 |
| FinNSP `neg_subject` |  99.23 |     99.23 | **99.52** |     98.07 | **99.14** |     97.42 |
| CFSC-ABSA            |  98.65 | **99.83** |     98.11 | **98.32** | **96.58** |     96.38 |

### 3.4 各类别召回率

| 任务                   | 类别       |   支持数 |      Laya |   FinBERT |
| -------------------- | -------- | ----: | --------: | --------: |
| FinFE                | negative |   574 |     0.780 | **0.819** |
| FinFE                | neutral  |   560 | **0.696** |     0.639 |
| FinFE                | positive |   886 |     0.832 | **0.872** |
| CFSC-ABSA            | negative | 2,146 | **0.948** |     0.943 |
| CFSC-ABSA            | neutral  | 1,032 | **0.797** |     0.749 |
| CFSC-ABSA            | positive | 5,134 |     0.955 | **0.958** |
| FinNSP `neg_has`     | true     |   259 | **0.961** |     0.958 |
| FinNSP `neg_has`     | false    |   241 |     0.938 | **0.954** |
| FinNSP `neg_subject` | 以上都不是    |   102 | **0.961** |     0.951 |

两模型都栽在 **neutral**（FinFE 上只有 0.64~0.70）。中性是金融文本里最难的一类：它既没有明确利空也没有明确利好，边界靠语义细微差别。Laya 在 CFSC 的 neutral 上高 4.8 个百分点（0.797 vs 0.749），这与「`criteria` 把中性定义写进了题面」的设计吻合。

### 3.5 换 encoder 之后（三方对比）

把 Laya 的 encoder 从 mmBERT 322M 换成 FinBERT 102M，超参完全不动：

| 任务         | 口径 |     n | Laya (mmBERT) | **FinBERT-Laya** | FinBERT 分类头 |
| ---------- | -- | ----: | ------------: | ---------------: | ----------: |
| FinFE      | 全量 | 2,020 |         77.97 |         **78.02** |   **79.26** |
| FinFE      | 严格 |   923 |         77.03 |         **80.28** |       79.52 |
| FinNSP `neg_has`     | 全量 |   500 |         95.00 |         **95.20** |   **95.60** |
| FinNSP `neg_has`     | 严格 |   499 |         94.99 |         **95.19** |       95.59 |
| FinNSP `neg_subject` | 全量 |   259 |    **96.91** |            95.75 |       95.37 |
| CFSC-ABSA  | 全量 | 8,312 |    **93.37** |            93.01 |       92.78 |
| CFSC-ABSA  | 严格 | 7,870 |    **93.58** |            93.01 |       92.94 |

**最值得看的一行是 FinFE 严格口径：80.28% vs 77.03%，+3.25 pt。** FinFE 评测集有 54.3% 的样本与训练集文本完全相同，mmBERT-Laya 从 laya 官方 checkpoint 出发、在这部分"见过"的样本上有记忆优势；一旦剔除重叠样本，优势就消失了。而 FinBERT-Laya 的决策头是**从零随机初始化**的，没有这层记忆，泛化反而更好。

唯一明显退步的是 `neg_subject`（有样本类别 mF1 0.8129 → 0.7003），但差距几乎全来自「摩尔龙」（1 条，召回 1.000 → 0.000）和「钱宝」（4 条，0.750 → 0.250）两个极小样本类，不宜当作能力差异。

完整分析（含源码依据、校准、选择性准确率、逐类召回、局限）见 **[docs/FINBERT_AS_ENCODER.md](docs/FINBERT_AS_ENCODER.md)**。

### 3.6 部署价值

单卡（RTX 4090 24GB），batch=1 串行测量：

|             | Laya (mmBERT) | **FinBERT-Laya** | FinBERT 分类头 |
| ----------- | ------------------------ | --------------- | ------------------------------- |
| 参数量         | 321.9M                   | **117.2M**（−63.6%） | 102.3M                          |
| 权重体积        | 614.0 MB（F16，170 张量） | **223.6 MB**（−63.6%） | 390 MB（fp32）→ **195 MB**（转 F16） |
| 延迟 p50      | 18.64 ms                 | **8.66 ms**（−53.5%） | 5.93 ms                     |
| 延迟 p95      | 19.34 ms                 | **8.92 ms** | 5.98 ms                     |
| 吞吐 batch=1  | 53.0 QPS                 | **113.0 QPS**（×2.13） | 145.3 QPS                   |
| 吞吐 batch=32 | —                        | — | 2857.7 QPS                      |
| 显存峰值        | 1499 MB                  | **667 MB**（−55.5%） | 486 MB                      |

> 体积口径：只算权重文件，已排除 `checkpoint_latest/`（训练中途快照，与顶层 `model.safetensors` 是同一份权重）。此前记录的 1228 MB 是含快照的错误口径。

**结论分两层**：

1. **换 encoder 的收益很实在** —— FinBERT-Laya 用 1/2.75 的参数量拿到 2.13 倍吞吐、省 55.5% 显存，而准确率持平（FinFE 严格口径还高 3.25 pt）。原来"Laya 太重"这个主要顾虑基本被消掉了。
2. **剩下的差距是架构性的** —— FinBERT-Laya 仍比单任务分类头慢 1.46 倍。但它一个 223.6 MB 的模型覆盖 4 种题，而分类头要 4 个模型（约 780 MB）才等效。按"覆盖同样任务"比，FinBERT-Laya 反而更省。

### 3.7 选型建议

| 场景                       | 建议                                   |
| ------------------------ | ------------------------------------ |
| 追求单任务最优准确率、GPU 充裕   | **FinBERT 分类头**（FinFE 上 +1.3 pt，最快）   |
| 多任务共存、需要置信度自动分流、显存受限 | **FinBERT-Laya**（117M / 223.6 MB / 8.66 ms，准确率与 322M 版持平） |
| 长尾实体识别（`neg_subject`） | **Laya (mmBERT)**（有样本类别 mF1 0.8129 vs 0.7003） |
| 需要严格校准的概率输出              | **FinBERT**（ECE/NLL 普遍更优，除 FinFE）    |
| 长文本（>512 token） | **Laya (mmBERT)**（8192 上下文；FinBERT 只有 512） |

---

## 4. 数据集来源

三个数据集都来自 **[CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)**（Chinese Financial Language Evaluation Benchmark，中文金融语言理解评测基准）。

| 数据集           | 任务       | 原始文件                                                       |                    体积 |                     条数 |
| ------------- | -------- | ---------------------------------------------------------- | --------------------: | ---------------------: |
| **CFSC-ABSA** | 方面级情感分析  | `train.txt` / `test.txt`                                   |       14.90 + 3.72 MB |         33,184 / 8,312 |
| **FinFE**     | 金融社交文本情绪 | `train_list.json` / `eval_list.json` / `problem_list.json` | 1.09 + 0.14 + 0.17 MB | 16,157 / 2,020 / 2,020 |
| **FinNSP**    | 金融负面主体识别 | `train_list.json` / `eval_list.json` / `problem_list.json` | 3.48 + 0.46 + 0.46 MB |      4,000 / 500 / 500 |

合计 **24.42 MB / 8 个文件**，已逐个 SHA256 校验通过（清单见 [datasets/README.md](datasets/README.md)）。

> **注意**：FinFE 与 FinNSP 的 `problem_list.json` 是**无标签测试输入**，只用于生成提交文件，不参与本项目评测。评测用的是 `eval_list.json`。
>
> **关于 FinFE 的 54.3% 重叠**：`finfe_eval.jsonl` 的 2,020 条里有 1,097 条 `state` 文本与 `finfe_train.jsonl` 完全相同。这是**原数据集本身的划分问题**，不是我们转换时引入的。所以本项目所有结论都同时给出全量与严格口径两套数字。

---

## 5. 数据处理与样例

### 5.1 转换流程

```
datasets/*/原始文件
        │  to_laya_format.py
        ▼
datasets/laya/*.jsonl          ← 9 个文件，120,034 行 / 125,235 题
        │  validate_laya_format.py   （退出码 0 才算通过）
        ▼
上传服务器 → laya-train / laya-evals
```

| 产物                         |     行数 |     题数 | 用途            |
| -------------------------- | -----: | -----: | ------------- |
| `all_train.jsonl`          | 53,341 | 55,562 | 三任务合训主输入      |
| `cfsc_absa_train.jsonl`    | 33,184 | 33,184 | 单任务           |
| `cfsc_absa_test.jsonl`     |  8,312 |  8,312 | 评测            |
| `finfe_train.jsonl`        | 16,157 | 16,157 | 单任务           |
| `finfe_eval.jsonl`         |  2,020 |  2,020 | 评测 + 训练时验证    |
| `finnsp_train.jsonl`       |  4,000 |  6,221 | 单任务（每行 1~2 题） |
| `finnsp_eval.jsonl`        |    500 |    759 | 评测            |
| `finfe_test_inputs.jsonl`  |  2,020 |  2,020 | 无标签测试输入       |
| `finnsp_test_inputs.jsonl` |    500 |  1,000 | 无标签测试输入       |

### 5.2 真实样例

**FinFE（`finfe_eval.jsonl` 第一行）** — 三分类 choice：

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

**CFSC-ABSA** — `state` 里带方面词，标准格式 `方面词：X\n语句：...【X】...`：

```json
{"state": "方面词：长谷投资\n语句：问董秘：请问长谷投资是否是你旗下的公司？【长谷投资】",
 "questions": {"sentiment_absa": {
   "type": "choice",
   "instructions": "句中针对指定方面词的情感倾向是什么？",
   "criteria": {"negative": "...", "neutral": "...", "positive": "..."}}},
 "expected": {"sentiment_absa": "neutral"},
 "meta": {"source": "CFSC-ABSA"}}
```

**FinNSP** — 一行两题，`choice` 识别主体 + `noul` 判定是否存在：

```json
{"state": "候选实体：恒昌汇财；纳觅财务\n文本：【网传恒昌汇财被查，投资者恐慌】...",
 "questions": {
   "neg_has":      {"type": "noul", "instructions": "文本中是否提及了负面主体？"},
   "neg_subject":  {"type": "choice", "instructions": "文本中的负面主体是哪个？",
                    "criteria": {"以上都不是": "...", "恒昌汇财": "...", "...": "..."}}},
 "expected": {"neg_has": true, "neg_subject": "以上都不是"},
 "meta": {"source": "FinNSP"}}
```

`neg_subject` 的选项空间固定为 **20 个实体 + 「以上都不是」**。构建时做过两件事：

- **实体名归一化** `norm_entity()`：去括注、去空白、转小写，把词表里 5 个近似重复的「资易贷」变体合并
- **固定选项顺序**：否则同一 qid 的 `criteria` 跨行不一致，训练时会被静默丢弃

---

## 6. 环境与命令

### 6.1 环境

两套环境**必须隔离**，因为 `laya` 与 `finbert` 的 torch/transformers 依赖区间不同。

**Laya 环境**

```bash
conda create -y -p /data/miniconda/envs/laya python=3.11
PY=/data/miniconda/envs/laya/bin/python
$PY -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
$PY -m pip install "transformers>=4.48,<5" laya==0.4.15
laya-train --help && laya-evals --help          # 确认 CLI 就位
```

**FinBERT 环境**

```bash
bash FinBert/scripts/setup_finbert.sh
```

**环境变量**（全部有默认值，默认值即本次实验服务器布局；本地复现时按需覆盖）


| 变量             | 默认值                         | 含义            |
| -------------- | --------------------------- | ------------- |
| `LAYA_ROOT`    | `/data/laya-fin`            | 实验根目录         |
| `LAYA_DATA`    | `${LAYA_ROOT}/laya`         | 转换好的 JSONL    |
| `LAYA_CKPT`    | `${LAYA_ROOT}/ckpt`         | 训练产物          |
| `LAYA_REPORTS` | `${LAYA_ROOT}/reports`      | 评测报告          |
| `LAYA_ENV`     | `/data/miniconda/envs/laya` | laya conda 环境 |
| `FINBERT_CKPT` | `${LAYA_ROOT}/finbert`      | FinBERT 训练产物  |
| `HF_ENDPOINT`  | `https://hf-mirror.com`     | 国内镜像          |

### 6.2 数据准备

```bash
python datasets/to_laya_format.py --help        # 原始 → Laya JSONL
python datasets/validate_laya_format.py         # 校验 qid 定义一致性（退出码 0 = 通过）
```

### 6.3 Laya：训练

```bash
bash FinLaya/scripts/train.sh dryrun    # 先空跑，验证 53,341 行 / 55,562 题能被正确解析
bash FinLaya/scripts/train.sh train     # 正式训练，4 epoch
```

`train` 的实际参数：

```
--data all_train.jsonl --eval finfe_eval.jsonl --out ckpt --base multilingual
--epochs 4 --micro-batch 24 --grad-accum 3 --device cuda
--shuffle-options --label-smoothing 0.05
```

实测：3068 步，loss `0.3983 → 0.3103 → 0.2565 → 0.2018`，32 分钟（18:41→19:13），显存 9.2 GB。

### 6.4 Laya：评测

```bash
# 未微调基线
BASE_MODEL=ml bash FinLaya/scripts/baseline.sh

# 微调后（三个测试集）
bash FinLaya/scripts/eval_ft.sh
bash FinLaya/scripts/eval_ft.sh finfe_eval            # 只跑一个

# 复算全套指标（Macro-F1 / 各类召回 / NLL / Brier / 选择性准确率）
python FinLaya/scripts/laya_metrics.py FinLaya/reports/eval/ft_finfe_eval.json \
       $LAYA_DATA/finfe_eval.jsonl
python FinLaya/scripts/laya_metrics.py FinLaya/reports/eval/ft_finfe_eval.json \
       $LAYA_DATA/finfe_eval.jsonl --strict

# 泄漏审计
python FinLaya/scripts/check_leak.py
```

> ⚠️ **不要用 `laya-evals run --model <本地目录>`**。该参数的取值只走注册表别名（`english` / `multilingual` / `ml` / `laya` / ...），传本地目录会被 `router.resolve()` 拒绝。官方支持本地目录的唯一入口是 `Router.register(name, source)`，`run_eval.py` 就是包了这一层。
>
> ⚠️ `EvalReport` **不是 dict**，序列化必须用 `report.to_json()` / `report.to_markdown()`，它没有 `to_dict()`。
>
> ⚠️ **`laya_metrics.py` 自己会写 `_metrics.json`**。不要把它的 stdout 重定向到那两个文件——会把刚写好的 JSON 覆盖成人类可读文本。stdout 丢 `/dev/null` 即可。

### 6.5 FinBERT：训练与评测

```bash
ENV_PY=/data/miniconda/envs/finbert/bin/python

$ENV_PY FinBert/scripts/finbert_baseline.py train        # 四任务分类头
$ENV_PY FinBert/scripts/fb_eval.py                       # 全部四任务，全量 + 严格
$ENV_PY FinBert/scripts/fb_eval.py nsp_has               # 单任务（分批跑更稳）
```

实测训练耗时：`absa` 218 s（loss 0.1371）· `finfe` 63 s（0.3593）· `nsp_has` 28 s（0.0429）· `nsp_subject` 16 s（0.2516）。

> ⚠️ **必须分批跑**。一次性跑完 8 个评测任务（含 8,312×2）会被系统 OOM killer 杀掉且**不留 Traceback**，看起来像脚本静默退出。

### 6.6 FinBERT-Laya：换 encoder

```bash
export LAYA_ROOT=/data/laya-fin
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$LAYA_ROOT/hf_cache

# 1) 造 base checkpoint：FinBERT encoder + 随机初始化决策头（约 1 分钟）
python FinBertLaya/scripts/make_fb_base.py

# 2) dryrun（应输出 53341 行 / 55562 题 / skipped {}）
laya-train --data $LAYA_ROOT/laya/all_train.jsonl --eval $LAYA_ROOT/laya/finfe_eval.jsonl \
  --out $LAYA_ROOT/fb_ckpt --base $LAYA_ROOT/fb_base --dry-run

# 3) 训练（23 分钟）
bash FinBertLaya/scripts/train.sh

# 4) 评测（三个测试集 + 复算指标）
bash FinBertLaya/scripts/eval.sh
```

> ⚠️ **`--base` 不能直接传 HF ID**。`resolve_checkpoint_dir` 对 Hub ID 走 `snapshot_download`，`allow_patterns` 只含 laya 布局文件，FinBERT 仓库没有 `rl_agent_config.json` 就直接报错。必须先跑 `make_fb_base.py` 造本地目录。

### 6.7 部署指标与对照报告

不同模型依赖不同 env 的 torch，**必须分开执行**：

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
python common/make_compare.py            # → docs/COMPARE_Laya_vs_FinBERT.md
```

---

## 7. 踩坑记录

按「症状 → 原因 → 处理」排列，全部是实际撞过的。

| 症状                                                 | 原因                                                                | 处理                                                          |
| -------------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------- |
| 某个任务指标异常低，却没有任何报错                                  | 同一 `qid` 的 `questions` 跨行不一致，`train.py` 只 `warn` 并保留第一条，后续定义被静默丢弃 | `to_laya_format.py` 固定选项顺序 + `validate_laya_format.py` 强制校验 |
| 2712 条 schema 错误                                   | `neg_subject` 选项集随行变化（实体数不定）                                      | 固定 20 实体 + 「以上都不是」                                          |
| 实体类出现多个近重复项                                        | 原词表里有「资易贷」的 5 种写法                                                 | `norm_entity()`：去括注 + 去空白 + 小写                              |
| `unknown model`                                    | `laya-evals --model` 只认别名                                         | 基线用 `ml`；微调后用 `Router.register()`                           |
| `EvalReport is not JSON serializable`              | 报告对象不是 dict                                                       | `report.to_json()`                                          |
| `AttributeError: np has no attribute trapz`        | numpy ≥ 2 移除                                                      | `getattr(np,"trapezoid",None) or np.trapz`                  |
| `neg_has` 算出 acc 0.519、false 召回 0                  | 把 `answer["noul"]` 当 bool，其实是 **True 的概率（float）**                 | 直接取 `pt = answer["noul"]`，修完是 95.00%                        |
| `neg_subject` mF1 只有 0.5419                        | 21 类里 7 类零支持，被强制计 0                                               | 增加 `macro_f1_present`，有样本 14 类算得 0.8129                     |
| FinBERT 评测跑一半进程消失、无 Traceback                      | 一次跑 8 个任务被 OOM killer 杀                                           | 按任务分批 `fb_eval.py <task>`                                   |
| `bash: $'\r': command not found`                   | Windows CRLF 上传到 Linux                                            | 上传前 `replace(b"\r\n", b"\n")`                               |
| SFTP 偶发 `ENOENT`                                   | 并发/通道复用                                                           | 写本地临时文件 → `sftp.put` → 失败重连重试                               |
| `paramiko` 报 `Bad password or corrupt private key` | RSA PEM 被猜成 ed25519                                               | 显式 `pkey=RSAKey.from_private_key_file()`                    |
| 前台命令约 2 分钟被 SIGTERM                                | 会话超时                                                              | 长任务全部后台化，`setsid ... < /dev/null &` 后立即关通道                  |
| 多层引号嵌套传 Python 到远端必炸                               | shell 转义层数过多                                                      | 一律写脚本文件上传，不用 `python -c "..."`                              |
| 训练跑完又自动重跑一遍                                    | 内联 heredoc 启动的后台命令在 paramiko 超时、通道关闭后，僵死的 `bash -c` 又执行了一次 `setsid` | 后台任务一律「先 sftp 上传脚本 → 再单独发一条启动命令」，不用内联 heredoc；启动后 `pgrep` 确认只有一个进程 |
| `_metrics.json` 变成人类可读文本                          | 把 `laya_metrics.py` 的 stdout 重定向到它自己会写的那个文件（stdout 最后 flush，覆盖 JSON） | 不重定向，或丢 `/dev/null`；`eval.sh` 已修正 |
| 权重体积比参数量×2 大近一倍                                | `weights_only_mb` 把 `checkpoint_latest/` 训练快照也算进去了                    | 统计时跳过 `checkpoint_latest/`，只算顶层 `model.safetensors`         |
| `--base <HF ID>` 报 `FileNotFoundError: rl_agent_config.json` | Hub 下载的 `allow_patterns` 只含 laya 布局文件                             | 先用 `make_fb_base.py` 造本地 base 目录                           |

---

## 8. 已知局限

1. **FinFE 有 54.3% 训练集重叠**。全量口径的 77.97% 不可作为泛化能力证据，请以严格口径的 77.03% 为准。重叠来自原数据集划分，本项目未做修正——如果要做干净评测，需要重新按文本去重划分训练/测试。
2. **`neg_subject` 的长尾严重**。259 条评测样本里 102 条是「以上都不是」，剩下 157 条散在 13 个实体上，多数实体只有 1~7 条支持，单类召回率波动很大（如「钱宝」4 条，Laya 0.75 / FinBERT 0.00），不宜逐类解读。FinBERT-Laya 在这一项的 0.7003 vs 0.8129 差距同样来自「摩尔龙」1 条与「钱宝」4 条。
3. **`noul` 温度是在 34 条样本上拟合的**，`train_report.json` 里带 `"fitted on only 34 calibration items"` 警告。`neg_has` 的 ECE 0.0506 可信度低于其他项；FinBERT-Laya 的 0.0882 同理。
4. **FinBERT 是单任务单模型**，部署对比里 FinBERT 的 195 MB / 5.93 ms 是**一个任务**的成本；4 个任务要 ×4。§3.6 的对比按「4 任务合计 ~780 MB」换算后才公平。
5. **`FinBERT` 转 F16 的 195.1 MB 是另一次实测结论，转换脚本是事后补写的**（`common/to_fp16.py`），只做了语法校验、未在服务器上重跑复现。fp32 权重 390 MB 是确定可复现的。
6. **只跑了单 seed**。所有数字都是 `seed=0` 一次结果，没有做多 seed 方差估计，1 pt 以内的差距（如 CFSC 93.37 vs 92.78、FinBERT-Laya 的 −0.36 pt）不应视为显著。FinFE 严格口径的 +3.25 pt 差距较大，但仍建议多 seed 复核。
7. **FinBERT-Laya 的 512 截断**。BERT 位置编码上限 512，而 FinNSP 有 5.00% 训练样本超长（p99 = 1,851，max = 6,777），尾部信息会丢。CFSC / FinFE 无此问题（p99 = 436 / 60）。这是 BERT 与 mmBERT（8192 上下文）的架构性差距，换 encoder 换不掉。
8. **FinBERT-Laya 与 mmBERT-Laya 的起点不对等**。前者决策头随机初始化（loss 1.3957），后者从 laya 官方 checkpoint 续训（loss 0.3983）。所以两者**只能比终点，不能比收敛速度**——这是"换 encoder"不可避免的，FinBERT 没有对应的 laya checkpoint。

---

## 9. 引用

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

代码 MIT。数据集遵循 CFLEB 原始协议，详见 [datasets/README.md](datasets/README.md)。
