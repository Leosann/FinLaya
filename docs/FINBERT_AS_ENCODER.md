# 把 FinBERT 换成 Laya 的 encoder

> **一句话结论**：可以，而且**不用改 laya 一行源码**。换完参数少 63.6%、延迟降 53.5%、显存省 55.5%，三个任务准确率基本打平；FinFE 严格口径上反而**高出 3.25 个百分点**。

---

## 1. 为什么可以这么改

不是"魔改"，是 laya 本来就留好的口子。三处源码证据（`laya 0.4.15`）：

**① 决策头按 encoder 的 hidden_size 动态构建**（`common.py:582`）

```python
class DecisionModel(nn.Module):
    def __init__(self, encoder, head_layers=2, n_act=2, dropout=0.1, no_init=False):
        d = encoder.config.hidden_size          # ← 不硬编码
        nhead = max(1, d // 64)                 # ← 不硬编码
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, ...)
        self.head = nn.TransformerEncoder(layer, head_layers, ...)
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), ...)
```

**② `build_model` 用通用 `AutoModel`，`cfg["encoder"]` 接受任意 HF ID**（`common.py:694`）

```python
def build_model(cfg, encoder_dir=None, pretrained=True, revision=None):
    from transformers import AutoConfig, AutoModel
    ...
    enc = AutoModel.from_pretrained(cfg["encoder"], attn_implementation="sdpa")
    return DecisionModel(enc, head_layers, n_act)
```

**③ `--base` 是 `dest="model_dir"`，接受任意路径或 model ID**（`train_cli.py:45`）

```python
parser.add_argument("--base", "--model", dest="model_dir",
                    default="convaiinnovations/laya",
                    help="Base checkpoint path or model ID")
```

所以唯一的前提是 **hidden_size 一致**，好让决策头的形状不变。这里正好一致：

| | mmBERT-base（原） | FinBERT（拟） | |
|---|---|---|---|
| model_type | modernbert | bert | — |
| **hidden_size** | **768** | **768** | ✅ 一致 |
| num_hidden_layers | 22 | 12 | — |
| num_attention_heads | 12 | 12 | — |
| intermediate_size | 1152（GLU，等效 2304） | 3072 | — |
| vocab_size | 256,000 | 21,128 | — |
| max_position_embeddings | 8192 | 512 | ⚠️ 要改 max_len |

两个附加验证：
- `_apply_rope_config(ecfg)` 只对 `rope_parameters` 存在时生效，BERT config 没有这个键 → **直接 return，无副作用**
- `forward` 用 `self.encoder(...).last_hidden_state`，`BertModel` 完全支持；`position_ids` 分支只在 `option_ids is not None`（parallel_layout）时走，本项目 `option_layout: null` 不触发

### 关键数字：省下来的全是多语言词表

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| encoder | 322M 中 **196.61M 是 embedding** | 102.27M（embedding 仅 16.23M） |
| 决策头 | 14.97M | **14.97M（完全一致）** |
| 合计 | 321.9M | 117.2M |

决策头一模一样，差的全在 encoder。而 mmBERT 的 196.6M embedding 是为 256k 多语言词表付的钱——**中文金融任务用不到**。

---

## 2. 怎么做的（三步，无需改源码）

### 步骤 1：造一个 FinBERT-Laya base checkpoint

`laya-train --base` 要求目录是完整的 laya 布局（`rl_agent_config.json` + `model.safetensors` + `encoder/` + `tokenizer/`），直接传 `yiyanghkust/finbert-tone-chinese` 不行（`resolve_checkpoint_dir` 会走 `snapshot_download` 的 `allow_patterns`，找不到 `rl_agent_config.json` 就报错）。

所以 `scripts/make_fb_base.py` 做四件事：

```python
cfg = json.load(open(SRC + "/rl_agent_config.json"))   # 借 mmBERT 版的 cfg 骨架
cfg["encoder"] = "yiyanghkust/finbert-tone-chinese"
cfg["max_len"] = 512                                   # BERT 位置编码上限
cfg["temperature"] = [1.0, 1.0, 1.0]                   # 旧温度是在 mmBERT 表征上拟合的，作废
cfg.pop("abstention_thresholds", None)
cfg.pop("training", None)

model = build_model(cfg, pretrained=True)   # encoder 载 FinBERT 预训练权重，决策头随机初始化
save_checkpoint(model, tok, cfg, OUT)       # 写出 laya.load 能读的四件套
```

产物 `fb_base/`：

```
model.safetensors        223.64 MB   (fp16, 235 张量)
rl_agent_config.json      encoder=yiyanghkust/finbert-tone-chinese, max_len=512
encoder/config.json       BertConfig
tokenizer/                BertTokenizerFast, vocab.txt + tokenizer.json
```

`load_checkpoint(OUT)` 用 `strict=True` 回读通过 —— 这一步是关键验证，说明权重键与 laya 期望的完全一致。

### 步骤 2：原配方训练

超参与 mmBERT 版**逐项一致**，唯一差别是 `--base`：

```bash
laya-train --data all_train.jsonl --eval finfe_eval.jsonl \
  --out fb_ckpt --base /data/laya-fin/fb_base \
  --epochs 4 --micro-batch 24 --grad-accum 3 --device cuda \
  --shuffle-options --label-smoothing 0.05
```

dryrun 先验证：`53,341 行 / 55,562 题，skipped {}`，**与 mmBERT 版完全一致**，optimizer budget 同为 3068 步。

### 步骤 3：原口径评测

`run_eval.py`（`Router.register` 本地目录）→ `laya_metrics.py` 复算，全量与严格口径都跑。

---

## 3. 实验结果

### 3.1 准确率：三个任务打平，FinFE 严格口径反超

| 任务 | 口径 | n | mmBERT-Laya (322M) | **FinBERT-Laya (117M)** | FinBERT 分类头 (102M) |
|---|---|---:|---:|---:|---:|
| FinFE 社交情绪 | 全量 | 2,020 | 77.97% | **78.02%** | **79.26%** |
| FinFE 社交情绪 | 严格 | 923 | 77.03% | **80.28%** ⭐ | 79.52% |
| FinNSP 有无负面 | 全量 | 500 | 95.00% | **95.20%** | **95.60%** |
| FinNSP 有无负面 | 严格 | 499 | 94.99% | **95.19%** | 95.59% |
| FinNSP 负面主体 | 全量 | 259 | **96.91%** | 95.75% | 95.37% |
| CFSC-ABSA | 全量 | 8,312 | **93.37%** | 93.01% | 92.78% |
| CFSC-ABSA | 严格 | 7,870 | **93.58%** | 93.01% | 92.94% |

**最值得说的一行**：FinFE 严格口径，FinBERT-Laya **80.28%** vs mmBERT-Laya 77.03%，**+3.25 pt**。

为什么？FinFE 评测集有 54.3%（1,097/2,020）的样本与训练集文本完全相同。mmBERT-Laya 的起点是 laya 官方 checkpoint（已经见过大量公开金融语料），在这部分"见过"的样本上有记忆优势；一旦剔除重叠样本，它的优势就消失了。而 FinBERT-Laya 的决策头是**从零随机初始化**学的，没有这层记忆，泛化反而更好。

### 3.2 Macro-F1

| 任务 | 口径 | mmBERT-Laya | FinBERT-Laya | FinBERT 分类头 |
|---|---|---:|---:|---:|
| FinFE | 全量 | **0.7690** | 0.7629 | 0.7776 |
| FinFE | 严格 | 0.7094 | **0.7371** | 0.7258 |
| FinNSP `neg_has` | 全量 | 0.9499 | **0.9520** | 0.9559 |
| FinNSP `neg_subject` | 全量 | 0.5419 | 0.4669 | 0.5038 |
| FinNSP `neg_subject`（有样本 14 类） | 全量 | **0.8129** | 0.7003 | 0.6838 |
| CFSC-ABSA | 全量 | **0.8993** | 0.8909 | 0.8881 |
| CFSC-ABSA | 严格 | **0.9025** | 0.8905 | 0.8902 |

`neg_subject` 是唯一明显退步的：有样本类别 mF1 从 0.8129 掉到 0.7003。原因是长尾——259 条样本里 102 条是「以上都不是」，剩 157 条散在 13 个实体上，多数只有 1~7 条。看逐类：

| 实体 | 支持数 | mmBERT-Laya 召回 | FinBERT-Laya 召回 |
|---|---:|---:|---:|
| 摩尔龙 | 1 | 1.000 | **0.000** |
| 钱宝 | 4 | 0.750 | **0.250** |
| 钱宝网 | 19 | **1.000** | 0.947 |
| 宜贷网 | 26 | 0.962 | **1.000** |
| 以上都不是 | 102 | 0.961 | 0.961 |

差距几乎全来自「摩尔龙」1 条和「钱宝」4 条这两个极小样本类。**这类波动不宜当作能力差异解读。**

### 3.3 概率校准

| 任务 | 口径 | mmBERT ECE | FinBERT-Laya ECE | FinBERT 分类头 ECE |
|---|---|---:|---:|---:|
| FinFE | 全量 | 0.0739 | **0.0485** | 0.0947 |
| FinFE | 严格 | **0.0628** | 0.0710 | 0.0876 |
| FinNSP `neg_has` | 全量 | **0.0506** | 0.0882 | 0.0311 |
| FinNSP `neg_subject` | 全量 | 0.1984 | **0.1546** | 0.0616 |
| CFSC-ABSA | 全量 | 0.0638 | **0.0607** | 0.0339 |

NLL / Brier 上 FinBERT-Laya 普遍更好：

| 任务 | NLL (mm / fb-laya / fb-cls) | Brier (mm / fb-laya / fb-cls) |
|---|---|---|
| FinFE 全量 | 0.6467 / **0.5878** / 0.6244 | 0.3516 / **0.3259** / 0.3285 |
| FinFE 严格 | 0.6233 / **0.5328** / 0.5713 | 0.3437 / **0.2916** / 0.3066 |
| FinNSP `neg_has` | **0.1799** / 0.1851 / 0.1517 | 0.0873 / 0.0815 / **0.0740** |
| FinNSP `neg_subject` | 0.3514 / 0.3113 / **0.2821** | 0.0952 / **0.0850** / 0.0991 |
| CFSC-ABSA 全量 | 0.2658 / 0.2424 / **0.2157** | 0.1170 / **0.1119** / 0.1126 |

`neg_has` 的 ECE 0.0882 偏高，但注意这一项的 noul 温度是在**仅 34 条**校准样本上拟合的（`train_report.json` 里带警告），本身噪声就大。

### 3.4 选择性准确率

按置信度取前 X% 自动处理：

| 任务 | @50 (mm / fb-laya / fb-cls) | @80 | @90 |
|---|---|---|---|
| FinFE | 89.50 / **91.39** / 90.99 | 83.79 / **85.09** / 85.02 | 81.35 / **81.96** / 82.40 |
| FinNSP `neg_has` | **99.60** / **99.60** / 100.0 | **99.00** / 98.75 / 99.75 | **98.22** / 98.00 / 98.00 |
| FinNSP `neg_subject` | 99.23 / 98.46 / **99.23** | **99.52** / 97.58 / 98.07 | **99.14** / 97.85 / 97.42 |
| CFSC-ABSA | 98.65 / **99.47** / 99.83 | 98.11 / **98.18** / 98.32 | 96.58 / **96.68** / 96.38 |

FinFE 上 FinBERT-Laya 在三个覆盖率下都优于 mmBERT 版，这与严格口径的准确率优势一致。

### 3.5 各类别召回

| 任务 | 类别 | 支持数 | mmBERT-Laya | FinBERT-Laya | FinBERT 分类头 |
|---|---|---:|---:|---:|---:|
| FinFE | negative | 574 | 0.780 | 0.791 | **0.819** |
| FinFE | neutral | 560 | **0.696** | 0.611 | 0.639 |
| FinFE | positive | 886 | 0.832 | **0.880** | 0.872 |
| CFSC | negative | 2,146 | **0.948** | 0.944 | 0.943 |
| CFSC | neutral | 1,032 | **0.797** | 0.751 | 0.749 |
| CFSC | positive | 5,134 | 0.955 | **0.960** | 0.958 |
| FinNSP `neg_has` | true | 259 | **0.961** | 0.950 | 0.958 |
| FinNSP `neg_has` | false | 241 | 0.938 | **0.954** | 0.954 |

**neutral 依然是全场短板**（0.61~0.70）。FinBERT-Laya 的 neutral 召回比 mmBERT 版低（0.611 vs 0.696），但 positive 高 4.8 pt，整体 F1 只差 0.006。neutral 边界靠语义细微差别，换 encoder 后这类样本的判断方式确实变了。

---

## 4. 部署价值：这才是换 encoder 的主要收益

| | mmBERT-Laya | **FinBERT-Laya** | 变化 | FinBERT 分类头 |
|---|---:|---:|---:|---:|
| 参数量 | 321.9M | **117.2M** | **−63.6%** | 102.3M |
| 权重体积 (F16) | 614.0 MB | **223.6 MB** | **−63.6%** | 390 MB (fp32) / 195 MB (F16) |
| 延迟 p50 | 18.64 ms | **8.66 ms** | **−53.5%** | 5.93 ms |
| 延迟 p95 | 19.34 ms | **8.92 ms** | **−53.9%** | 5.98 ms |
| 吞吐 batch=1 | 53.0 QPS | **113.0 QPS** | **×2.13** | 145.3 QPS |
| 显存峰值 | 1499 MB | **667 MB** | **−55.5%** | 486 MB |

> 体积口径说明：只算权重文件，已排除 `checkpoint_latest/`（训练中途快照，与顶层 `model.safetensors` 是同一份权重）。此前记录过的 1228 MB / 447 MB 是含快照的错误口径，已修正。

**FinBERT-Laya 用 1/2.75 的参数量，拿到 2.13 倍的吞吐，只比 FinBERT 分类头慢 1.46 倍** —— 而分类头要 4 个模型才覆盖这 4 种题（约 780 MB），FinBERT-Laya 一个 223.6 MB 的模型全包。

### 训练成本

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| 步数 | 3,068 | 3,068（完全相同） |
| 耗时 | 32 分钟 | **23 分钟**（−28%） |
| 显存（采样） | 9.2 GB | **~3.8 GB**（−59%） |
| 末 epoch loss | 0.2018 | 0.3047（起点不同，不可直接比） |
| FinFE 验证集 acc | 0.4386 → 0.7807 | 0.4386 → 0.7802 |

**起点 loss 不可比**：mmBERT 版从 laya 官方 checkpoint 出发（loss 0.3983），FinBERT 版的决策头是随机初始化的（loss 1.3957）。所以这里只比"终点"，不比"降幅"。

---

## 5. 结论

| 问题 | 答案 |
|---|---|
| 能不能把 FinBERT 当 encoder 换进 Laya？ | **能，且不用改 laya 源码**。前提只有 hidden_size 一致（768） |
| 效果会掉吗？ | **基本不掉**。FinFE 严格口径 +3.25 pt，neg_has +0.20 pt，CFSC −0.36 pt，neg_subject −1.16 pt（长尾噪声） |
| 值得做吗？ | **值得**。参数 −63.6%、延迟 −53.5%、显存 −55.5%，准确率持平 |
| 什么场景选它？ | 需要 Laya 的"一份权重多任务 + 置信度弃权"，但觉得 322M 太重 |
| 什么场景别选？ | 追求单任务极限准确率且不在乎多模型 → 仍用 FinBERT 分类头；长尾实体识别（`neg_subject`）→ 用 mmBERT-Laya |

### 推荐配置

```
FinBERT-Laya (117M / 223.6 MB / 8.66 ms)
  ├─ 适合：多任务共存、需要弃权阈值、显存受限、延迟敏感
  └─ 换掉它的时机：单任务准确率不够 → FinBERT 分类头
                    长尾实体召回不够 → mmBERT-Laya
```

---

## 6. 局限与注意

1. **512 截断**：FinBERT 位置编码上限 512，而 FinNSP 有 5.00% 的训练样本超过 512（p99 = 1,851，max = 6,777）。这些长文本的尾部信息会丢。CFSC/FinFE 无此问题（p99 分别 436 / 60）。若要彻底解决，需要 sliding window 或长文本切分——但 BERT 本身不具备 mmBERT 的 8192 上下文能力，这是架构层面的取舍。
2. **起点不对等**：mmBERT 版从 laya 官方 checkpoint 续训，FinBERT 版决策头随机初始化。这是"换 encoder"不可避免的——FinBERT 没有对应的 laya checkpoint。所以**只能比终点，不能比收敛速度**。
3. **单 seed**：所有数字都是 `seed=0` 一次结果。1 pt 以内的差距（CFSC −0.36 pt、neg_has +0.20 pt）不应视为显著。FinFE 严格口径的 +3.25 pt 差距较大，但仍建议多 seed 复核。
4. **`neg_has` 的 ECE 0.0882 偏弱**：noul 温度只在 34 条校准样本上拟合，本身不稳。
5. **`neg_subject` 的 0.7003 vs 0.8129 不宜过度解读**：差距集中在「摩尔龙」（1 条）和「钱宝」（4 条）两个极小样本类。
6. **未尝试的方向**：换 `bert-base-chinese` 裸 encoder 做对照、用 FinBERT 的 `pooler_output` 替代 `last_hidden_state[:,0]`、把 `head_layers` 从 2 加到 3 补回深度差、以及给 FinBERT-Laya 更多 epoch 看是否追上 CFSC 的 0.5 pt 差距。

---

## 7. 复现

```bash
# 0) 环境：laya env（torch 2.6.0+cu126 / transformers 4.57.6 / laya 0.4.15）
export LAYA_ROOT=/data/laya-fin
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$LAYA_ROOT/hf_cache

# 1) 造 base checkpoint（FinBERT encoder + 随机决策头），约 1 分钟
python FinBertLaya/scripts/make_fb_base.py

# 2) dryrun 确认数据解析（应为 53341 行 / 55562 题 / skipped {}）
laya-train --data $LAYA_ROOT/laya/all_train.jsonl --eval $LAYA_ROOT/laya/finfe_eval.jsonl \
  --out $LAYA_ROOT/fb_ckpt --base $LAYA_ROOT/fb_base --dry-run

# 3) 训练（23 分钟）
bash FinBertLaya/scripts/train.sh

# 4) 评测三个测试集 + 复算指标
bash FinBertLaya/scripts/eval.sh

# 5) 部署指标（laya env 下跑）
python common/deploy_metrics.py fb_laya
```

产物清单见 [`../FinBertLaya/README.md`](../FinBertLaya/README.md)。
