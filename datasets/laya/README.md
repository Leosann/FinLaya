# Laya 可训练格式（JSONL）

由 `to_laya_format.py` 从 `../CFSC-ABSA`、`../FinFE`、`../FinNSP` 原始数据转换而来，
`validate_laya_format.py` 按 Laya 官方源码规则校验，**0 错误**。

## 一、Laya 训练数据 schema

对齐官方 `docs/finetune.md` 与 `laya/train.py`：

```json
{"state": "文本或 JSON 状态",
 "questions": {"qid": {"type": "choice|score|noul", "instructions": "...", "criteria": {...}}},
 "expected": {"qid": 答案}}
```

- `state`：待判断的文本（也可传 dict，会被序列化）
- `questions`：键是题号，值是题型定义；**同一 qid 的定义必须跨行一致**
- `expected`：每题一个标准答案（也可用 `gold` 存教师概率分布）
  - `choice` → 填选项字符串（必须是 `criteria` 的 key）
  - `noul` → 填 `true` / `false`
  - `score` → 填档位
- **不写 `expected` 的行就是推理输入**，可直接 `predict` / `laya-evals`

## 二、文件清单

| 文件 | 行数 | 题数 | 说明 |
| --- | --- | --- | --- |
| `cfsc_absa_train.jsonl` | 33,184 | 33,184 | ABSA 训练 |
| `cfsc_absa_test.jsonl` | 8,312 | 8,312 | ABSA 测试（带标签） |
| `finfe_train.jsonl` | 16,157 | 16,157 | 社交情绪训练 |
| `finfe_eval.jsonl` | 2,020 | 2,020 | 社交情绪验证 |
| `finfe_test_inputs.jsonl` | 2,020 | 2,020 | 仅输入，无标签 |
| `finnsp_train.jsonl` | 4,000 | 6,221 | 负面主体训练（含双题） |
| `finnsp_eval.jsonl` | 500 | 759 | 负面主体验证 |
| `finnsp_test_inputs.jsonl` | 500 | 1,000 | 仅输入，无标签 |
| `all_train.jsonl` | 53,341 | 55,562 | 三份训练集合并打散（seed=42） |

`all_train.jsonl` = 33,184 + 16,157 + 4,000，随时可由脚本重新生成。

## 三、三个任务的题型设计

### 1. CFSC-ABSA → `sentiment_absa`（choice，三选一）

```json
{"state": "方面词：湖南\n语句：……【湖南】、江西增速分别提高1.4和0.7个百分点……",
 "questions": {"sentiment_absa": {"type": "choice",
   "instructions": "在给定方面词的语境下，这句话对该方面词表达的情感倾向是什么？",
   "criteria": {"positive": "…", "neutral": "…", "negative": "…"}}},
 "expected": {"sentiment_absa": "positive"}}
```

- 原数据一个方面词一条样本，这里保持粒度不变（41,496 条）
- `state` 用 `【】` 标出方面词位置，把 ABSA 的"给定方面词"条件显式喂给单向编码器；
  原始 `from`/`to` 位置若与实际子串不符会用 `find()` 兜底
- 类别分布：train 正 20,385 / 中 4,228 / 负 8,571；test 正 5,134 / 中 1,032 / 负 2,146

### 2. FinFE → `sentiment_social`（choice，三选一）

```json
{"state": "试仓500股，看看门道！",
 "questions": {"sentiment_social": {"type": "choice", "instructions": "…",
   "criteria": {"negative": "…", "neutral": "…", "positive": "…"}}},
 "expected": {"sentiment_social": "positive"}}
```

原始标签 `0/1/2` → `negative/neutral/positive`（**不是** 0/1/2 的常规顺序，转换已按官方映射）。
train 负 4,690 / 中 4,143 / 正 7,324；eval 负 574 / 中 560 / 正 886。

### 3. FinNSP → `neg_has`（noul）+ `neg_subject`（choice）

```json
{"state": "候选实体：小资钱包；恒丰银行；北京正聚源通鼎信用管理有限公司\n文本：……",
 "questions": {"neg_has": {"type": "noul", "instructions": "文中是否包含针对给定候选实体的负面消息…？"},
               "neg_subject": {"type": "choice", "instructions": "负面消息的主体是下列哪个实体？",
                               "criteria": {"小资钱包": "…", "宜贷网": "…", "以上都不是": "…"}}},
 "expected": {"neg_has": true, "neg_subject": "小资钱包"}}
```

- 原数据结构 `[id, 冗余字段, 正文, 候选实体, 是否含负面, 负面主体]`
- `neg_has`：noul，全部 4,000 条都出题
- `neg_subject`：**只在 `neg_has=true` 时出题**（2,221 条），负例不必猜主体
- 主体题的选项用**固定 20 个高频实体 + 「以上都不是」**，因为 Laya 要求同一 qid 的
  `questions` 定义跨行一致（`train.py` 导出 `questions.json` 时遇冲突只告警并保留第一条，
  动态候选项会导致推理端无法复原选项空间）
- 实体词表只从训练集统计，并按"去括注 + 去空白"归一合并近似重复项（如"资易贷"与两家全称）
- 权衡与残留问题：验证集有 **102/259（39%）** 正例主体落在词表之外，答 `以上都不是`。
  想覆盖更全可调大 `NSP_VOCAB_SIZE`，但 Laya 官方指出选项超前 20 个准确率明显下滑；
  更稳的做法是按实体域拆成多个子任务分别出 choice 题。

## 四、`meta` 字段

每行都带 `meta`，**Laya 训练时不读它**，保留是为了能追溯/回评：

- `meta.source`：`CFSC-ABSA` / `FinFE` / `FinNSP`
- `meta.split`：`train` / `test` / `eval`
- ABSA：`term`、`category`（Corporate/Stock/Market/Economy）、原始 `from`/`to`
- FinNSP：`id`、`candidates`（该样本的原始候选实体列表）、`subjects`（标准主体）

## 五、直接用于训练

```bash
pip install laya
laya-train --data all_train.jsonl --base multilingual --out ./laya-fin-sent
# 或先干跑，检查读入/跳过情况与优化步数预算，不加载权重
laya-train --data all_train.jsonl --base multilingual --dry-run
```

注意：`--base` 必须用 `multilingual` 或 `chinese` 类主干，**不要用英文 `laya`**。
中文字符在英文 checkpoint 上会掉到接近随机。单文件验证用
`laya-evals run finfe_eval.jsonl --model multilingual`。

## 六、重新生成 / 校验

```bash
python to_laya_format.py        # 重新转换，输出到 laya/
python validate_laya_format.py  # 校验，退出码 0 表示全部合法
```

校验覆盖：题型合法、`instructions` 非空、choice 标签为不重复标量、noul 答案必须是 bool、
choice 答案必须落在 criteria 内、同一 qid 的 schema 跨行一致、`expected` 题号必须已定义。
