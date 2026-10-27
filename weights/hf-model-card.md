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

中文金融情感决策模型。**用 1/3 的参数量，拿到与 322M 版本相同的准确率。**

把情感分析改造成「在选项上做决策」：模型不生成文字，而是直接输出**选项上的概率分布**，因此概率天然可校准，可以按置信度决定是否弃权。

| | |
|---|---|
| encoder | `yiyanghkust/finbert-tone-chinese`（BertModel，102.27 M） |
| 决策头 | Laya 原生（14.97 M） |
| 总参数 | **117.24 M** |
| 权重体积 | 223.6 MB（fp16，235 个张量） |
| 上下文 | 512 |
| 覆盖任务 | 情绪三分类 · 方面级情感 · 负面主体识别（**一个模型**） |
| 延迟 | 8.66 ms p50（RTX 4090，batch=1） |
| 显存 | 667 MB |

## 为什么是 117M

Laya 的决策头按 `encoder.config.hidden_size` 动态构建，mmBERT-base 与 FinBERT 的 hidden_size 都是 768，所以决策头形状完全一致（14.97 M，一个参数都不差）。省下来的 205 M 全在 encoder 的 embedding：

| | mmBERT-Laya | FinBERT-Laya |
|---|---:|---:|
| 词表 | 256,000 | 21,128 |
| embedding 参数 | 196.61 M | **16.23 M** |
| encoder 主体 | 110.4 M（22 层） | 85.0 M（12 层） |
| 决策头 | 14.97 M | 14.97 M |
| 合计 | 321.9 M | **117.2 M** |

中文金融任务用不到 256k 多语言词表，这 196.6 M 是纯浪费。

## 用法

```bash
pip install "transformers>=4.48,<5" laya==0.4.15
```

```python
import laya

router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "Leosann/FinLaya-117M")   # 直接给 Hub ID

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
print(ans["sentiment_social"])          # 预测标签 + 概率分布
```

`criteria` 必须与训练时**完全一致**，完整定义见仓库 `FinLaya/configs/questions.json`。

### 三类任务的输入格式

| 任务 | qid | 类型 | state 格式 |
|---|---|---|---|
| 金融社交情绪 | `sentiment_social` | choice(3) | 纯文本 |
| 方面级情感 | `sentiment_absa` | choice(3) | `方面词：X\n语句：……【X】……` |
| 负面主体识别 | `neg_has` | noul | `候选实体：A；B\n文本：……` |
| | `neg_subject` | choice(21) | 同上，输出 20 实体之一或「以上都不是」 |

## 训练

RLCD 配方（软标签交叉熵 `w_rps=1.0` + 球面正则 `w_sph=0.75`，4 条 RL 采样，sigma 0.4→0.1），4 epoch / 3,068 步 / 23 分钟 / 显存 ~3.8 GB。标签平滑 0.05，编码器 lr 2.5e-5，决策头 lr 1e-4。

训练后自动拟合：

| 项 | 值 |
|---|---|
| choice 温度 | 1.3531 |
| noul 温度 | 1.9008 |
| 弃权阈值 `choice:3-5` | 0.5968 |
| 弃权阈值 `noul:2` | 0.8953 |
| 弃权阈值 `choice:11+` | 0.7937 |

## 评测

三个数据集均来自 [CFLEB](https://github.com/ssymmetry/BBT-FinCUGE-Applications)。**严格口径** = 剔除与训练集文本 SHA256 完全相同的样本。

| 任务 | n（全量/严格） | 准确率 | Macro-F1 | ECE |
|---|---|---|---|---|
| FinFE 社交情绪 | 2,020 / 923 | 78.02 / **80.28** | 0.7629 / 0.7371 | 0.0485 / 0.0710 |
| FinNSP 有无负面 | 500 / 499 | 95.20 / 95.19 | 0.9520 / 0.9518 | 0.0882 / 0.0882 |
| FinNSP 负面主体 | 259 / 259 | 95.75 / 95.75 | 0.4669 / 0.4669 | 0.1546 / 0.1546 |
| CFSC-ABSA 方面级 | 8,312 / 7,870 | 93.01 / 93.01 | 0.8909 / 0.8905 | 0.0607 / 0.0618 |

与 322M 版本对照：

| 任务 | 口径 | mmBERT-Laya (322M) | **本模型 (117M)** | 差值 |
|---|---|---:|---:|---:|
| FinFE | 严格 | 77.03 | **80.28** | **+3.25 pt** |
| FinNSP 有无负面 | 全量 | 95.00 | 95.20 | +0.20 pt |
| FinNSP 负面主体 | 全量 | **96.91** | 95.75 | −1.16 pt |
| CFSC-ABSA | 全量 | **93.37** | 93.01 | −0.36 pt |

FinFE 严格口径的 +3.25 pt 是因为 322M 版本从 Laya 官方 checkpoint 续训、在 54.3% 重叠样本上有记忆优势；剔除重叠后优势消失，而本模型的决策头是从零学的。

## 局限

1. **512 截断**：FinNSP 有 5.00% 训练样本超长（p99=1,851，max=6,777），尾部信息会丢。BERT 架构上限，无法绕过。
2. **负面主体识别长尾严重**：259 条评测样本里 102 条是「以上都不是」，剩 157 条散在 13 个实体，多数只有 1~7 条，逐类指标波动大。
3. **`noul` 温度在 34 条样本上拟合**，`neg_has` 的 ECE 可信度低于其他任务。
4. **单 seed**：`seed=0` 一次结果，1 pt 以内的差距不应视为显著。
5. **决策头随机初始化起点**：与 mmBERT 版起点不同，只能比终点不能比收敛速度。

## 相关资源

- 代码与完整实验：https://github.com/Leosann/FinLaya
- 三模型全指标对比：[`docs/THREE_WAY_COMPARISON.md`](https://github.com/Leosann/FinLaya/blob/main/docs/THREE_WAY_COMPARISON.md)
- 换 encoder 的源码依据与做法：[`docs/FINBERT_AS_ENCODER.md`](https://github.com/Leosann/FinLaya/blob/main/docs/FINBERT_AS_ENCODER.md)

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
```
