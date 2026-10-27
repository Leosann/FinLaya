# FinBertLaya — FinBERT 作 encoder 的 Laya

把 Laya 的 mmBERT encoder（322M）换成 FinBERT（102M），**不改 laya 一行源码**，得到 117.2M 的决策模型。完整分析见 **[../docs/FINBERT_AS_ENCODER.md](../docs/FINBERT_AS_ENCODER.md)**。

| | |
|---|---|
| encoder | `yiyanghkust/finbert-tone-chinese`（BertModel 102.27M） |
| 决策头 | Laya 原生（14.97M，与 mmBERT 版**完全一致**） |
| 合计 | **117.24M**（mmBERT 版 321.9M，−63.6%） |
| 训练 | 4 epoch / 3,068 步 / **23 分钟** / 显存 ~3.8 GB |
| 关键结果 | FinFE 严格口径 **80.28%**（mmBERT 版 77.03%，**+3.25 pt**） |
| 部署 | 223.6 MB / 8.66 ms p50 / 113 QPS / 667 MB 显存 |

## 目录结构

```
FinBertLaya/
├─ scripts/
│  ├─ make_fb_base.py    [核心] 造 FinBERT-Laya base checkpoint
│  ├─ train.sh           RLCD 训练（超参与 mmBERT 版逐项一致）
│  └─ eval.sh            三测试集评测 + 指标复算
├─ configs/
│  ├─ fb_rl_agent_config.json   训练产出的推理配置（温度 / 弃权阈值）
│  └─ fb_train_report.json      before/after + 温度标定
├─ logs/
│  ├─ fb_train.log       训练日志（含 before/after 表）
│  └─ fb_eval.log        三测试集评测输出
└─ reports/
   ├─ fb_*_eval.json            laya-evals 原始报告
   └─ fb_*_metrics{,_strict}.json  复算指标（全量 / 严格）
```

## 为什么不用改源码

三处源码依据（`laya 0.4.15`）：

1. `DecisionModel.__init__`：`d = encoder.config.hidden_size`，`nhead = max(1, d // 64)` —— 决策头按 encoder 动态构建
2. `build_model`：用通用 `AutoModel.from_pretrained(cfg["encoder"])`，`cfg["encoder"]` 接受任意 HF ID
3. `laya-train --base` 的 `dest` 是 `model_dir`，接受任意路径

唯一硬前提是 **hidden_size 一致**。mmBERT-base 与 FinBERT 都是 **768**，所以决策头形状不变（14.97M 一模一样）。

## 运行

```bash
export LAYA_ROOT=/data/laya-fin
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=$LAYA_ROOT/hf_cache

# 1) 造 base checkpoint：encoder 载 FinBERT 预训练权重，决策头随机初始化
python scripts/make_fb_base.py

# 2) dryrun（应输出 53341 行 / 55562 题 / skipped {}）
laya-train --data $LAYA_ROOT/laya/all_train.jsonl --eval $LAYA_ROOT/laya/finfe_eval.jsonl \
  --out $LAYA_ROOT/fb_ckpt --base $LAYA_ROOT/fb_base --dry-run

# 3) 训练
bash scripts/train.sh

# 4) 评测
bash scripts/eval.sh
bash scripts/eval.sh finfe_eval      # 只跑一个

# 5) 部署指标（laya env 下）
python ../common/deploy_metrics.py fb_laya
```

## base checkpoint 里改了什么

`make_fb_base.py` 从 mmBERT 版的 `rl_agent_config.json` 复制骨架，只改四处：

| 字段 | 原值 | 新值 | 原因 |
|---|---|---|---|
| `encoder` | `jhu-clsp/mmBERT-base` | `yiyanghkust/finbert-tone-chinese` | 换 encoder |
| `max_len` | 1024 | **512** | BERT 的 `max_position_embeddings` 上限 |
| `temperature` | `[1.4378, 1.0, 1.4585]` | `[1.0, 1.0, 1.0]` | 旧温度在 mmBERT 表征上拟合，换 encoder 后失效 |
| `abstention_thresholds` | 三项 | 删除 | 同上，重训会重新拟合 |

`training` 字段整体删除，避免继承 mmBERT 那轮的训练元数据。

## 训练后自动得到

| 项 | 值 |
|---|---|
| choice 温度 | 1.3531 |
| noul 温度 | 1.9008 |
| 弃权阈值 `choice:3-5` | 0.5968 |
| 弃权阈值 `noul:2` | 0.8953 |
| 弃权阈值 `choice:11+` | 0.7937 |
| FinFE 验证集 acc | 0.4386 → **0.7802** |
| FinFE 验证集 ECE | 0.0967 → **0.0485** |

## 结果速查

| 任务 | 口径 | n | mmBERT-Laya | **FinBERT-Laya** | delta |
|---|---|---:|---:|---:|---:|
| FinFE | 全量 | 2,020 | 77.97% | **78.02%** | +0.05 pt |
| FinFE | 严格 | 923 | 77.03% | **80.28%** | **+3.25 pt** |
| FinNSP `neg_has` | 全量 | 500 | 95.00% | **95.20%** | +0.20 pt |
| FinNSP `neg_has` | 严格 | 499 | 94.99% | **95.19%** | +0.20 pt |
| FinNSP `neg_subject` | 全量 | 259 | **96.91%** | 95.75% | −1.16 pt |
| CFSC-ABSA | 全量 | 8,312 | **93.37%** | 93.01% | −0.36 pt |
| CFSC-ABSA | 严格 | 7,870 | **93.58%** | 93.01% | −0.57 pt |

## 两个坑

1. **`laya_metrics.py` 自己会写 `_metrics.json`**，不要把它的 stdout 重定向到这两个文件——那会把 JSON 覆盖成人类可读文本。`eval.sh` 里已丢 `/dev/null`。
2. **`--base` 传 HF ID 会失败**：`resolve_checkpoint_dir` 对 Hub ID 走 `snapshot_download` 且 `allow_patterns` 只含 laya 布局文件，FinBERT 仓库没有 `rl_agent_config.json` 就直接报错。必须先造本地 base 目录。

## 未保存的产物

- 训练好的权重（223.6 MB）不在本仓库，按上面步骤重跑 23 分钟即可。
- `fb_base/`（223.6 MB）同样未入库，用 `make_fb_base.py` 约 1 分钟重建。
