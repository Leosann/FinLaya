# Laya 中文金融情感微调实验报告

> 基座 `convaiinnovations/laya`（multilingual / mmBERT-base, 322M）· 三任务合训 · RTX 4090 24GB
> 训练日期 2026-10-26 · 框架 `laya 0.4.15` / `torch 2.6.0+cu126` / `transformers 4.57.6`
>
> **配套文档**：`../docs/COMPARE_Laya_vs_FinBERT.md` —— 与中文金融 FinBERT 基线的完整对照
> （Macro-F1 / 各类召回 / ECE / NLL / Brier / 选择性准确率 / 延迟 / 吞吐 / 显存 / 体积）

---

## 1. 结论摘要

三个中文金融任务全部显著提升，且在**剔除训练集重叠样本的严格口径**下依然成立：

| 数据集 | 任务 | 指标 | 基线 | 微调后 | 提升 | 严格口径 | 训练集重叠 |
|---|---|---|---|---|---|---|---|
| FinFE eval | 三分类情感（2,020） | choice_accuracy | 55.79% | **77.97%** | **+22.18** | 77.03% (n=923) | 54.3% |
| FinNSP eval | 20 选 1 主体（259） | choice_accuracy | 13.51% | **96.91%** | **+83.40** | **96.91%** (n=259) | **0%** |
| FinNSP eval | 有无负面（500） | noul_accuracy | 68.00% | **95.00%** | **+27.00** | 94.99% (n=499) | 0.2% |
| CFSC-ABSA test | 方面级情感（8,312） | choice_accuracy | 68.96% | **93.37%** | **+24.41** | 93.58% (n=7,870) | 5.3% |

**校准改善同样关键**（这才是 Laya 决策模型的价值所在）：

| 数据集 | ECE 基线 | ECE 微调后 | mean_confidence 基线 → 后 |
|---|---|---|---|
| FinFE | 0.1848 | **0.0739** | 0.7427 → 0.8517 |
| FinNSP | 0.3554 | **0.0998** | 0.8438 → 0.8686 |
| CFSC-ABSA | 0.1508 | **0.0638** | 0.8404 → 0.8883 |

ECE 下降 60–72%，意味着**置信度可以直接当门槛用**：CFSC-ABSA 在 confidence@80 阈值下
selective_accuracy 达 **98.08%**（全量 93.37%），这正是一个可上线的决策引擎应有的行为。

### 1.1 与 FinBERT 基线的对照（详见 `../docs/COMPARE_Laya_vs_FinBERT.md`）

用 `yiyanghkust/finbert-tone-chinese`（102M，中文金融语料预训练）在**同一批数据、同一套指标**
下微调三个分类头作为基线：

| 任务 | Laya acc | FinBERT acc | 谁更好 |
|---|---|---|---|
| FinFE 三分类 | 77.97% | **79.26%** | FinBERT +1.29 |
| FinNSP 有无负面 | 95.00% | **95.60%** | FinBERT +0.60 |
| FinNSP 负面主体（21/19 选 1） | **96.91%** | 95.37% | Laya +1.54 |
| CFSC-ABSA 三分类 | **93.37%** | 92.78% | Laya +0.59 |

- **FinBERT 在纯极性分类上略胜**（1–1.3 个点），且校准明显更好（CFSC ECE 0.0339 vs 0.0638；
  FinNSP 0.0311 vs 0.0506）
- **Laya 在「带候选集的结构化决策」上更好**（主体识别 +1.54），因为它能读完整题面
- **部署成本 Laya 约为 FinBERT 的 3 倍**：321.9M vs 102.3M 参数、18.56ms vs 5.93ms 延迟、
  1499MB vs 486MB 峰值显存

**注意**：Laya 的 `neg_subject` ECE 高达 **0.1984**，因为它的 `answer_confidence` 是
`|2p-1|` 形式的决策边际而非概率。做门控必须用 `rl_agent_config.json` 里拟合好的分桶阈值
（`choice:3-5`→0.6217、`choice:11+`→0.7562、`noul:2`→0.6661），不能直接比阈值。

---

## 2. 数据

### 2.1 来源与规模

| 数据集 | 训练 | 测试/验证 | 来源 |
|---|---|---|---|
| CFSC-ABSA | 33,184 | 8,312 | `github.com/Ya-dongLi/CFSC`（`CFSC-ABSA/json格式/`） |
| FinFE | 16,157 | 2,020 | `github.com/supersymmetry-technologies/BBT-FinCUGE-Applications` |
| FinNSP | 4,000 | 500 | 同上 |
| **合计** | **53,341** | 10,832 | 24.42 MB 原始数据，逐字节校验通过 |

### 2.2 转成 Laya 决策格式

Laya 的样本是「状态 + 若干待答问题 + 每题答案」，一个样本可以挂多道题：

```json
{"state": "…【方面词】…",
 "questions": {"qid": {"type": "choice|noul", "instructions": "…", "criteria": {…}}},
 "expected": {"qid": "答案"}}
```

三个任务的映射设计：

| 原任务 | qid | 题型 | 选项 / 答案 |
|---|---|---|---|
| CFSC-ABSA（方面级极性） | `sentiment_absa` | choice | positive / neutral / negative |
| FinFE（金融情绪） | `sentiment_social` | choice | negative / neutral / positive（注意 0,1,2 顺序非直觉） |
| FinNSP（新闻有无负面） | `neg_has` | noul | true / false |
| FinNSP（负面主体） | `neg_subject` | choice | 固定 20 高频实体 + 「以上都不是」 |

**关键设计决策**：`neg_subject` 原本想每题动态生成候选实体，但 `laya/train.py` 在导出
`questions.json` 时遇到同名 qid 定义冲突**只 `warnings.warn` 并保留第一条**，推理端将无法
复原选项空间。因此改为**固定 20 实体 + 「以上都不是」**的词表，且词表只从训练集统计（避免
验证集泄漏），并用 `norm_entity()`（去括注、去空白、小写）合并「资易贷」等 5 个近似重复写法。

**遗留局限**：该词表导致 FinNSP 验证集 259 条正例中约 102 条（39%）的主体落在词表外，
只能答「以上都不是」。这是任务设计上的取舍——换取 schema 一致性与推理端可复原。
实测 `neg_subject` 仍达 96.91%，说明模型学会了「不在词表就是以上都不是」这个信号。

### 2.3 泄漏审计（重要）

用 state 文本 SHA256 比对训练集与测试集：

```
finfe_eval       rows=2020   与训练集重叠=1097 (54.3%)   ← FinFE 官方切分本身有重叠
finnsp_eval      rows=500    与训练集重叠=1    (0.2%)    ← 干净
cfsc_absa_test   rows=8312   与训练集重叠=442  (5.3%)    ← 轻度
```

**处理**：剔除重叠样本重算（见 §1 严格口径列）。FinFE 重叠 54% 但严格口径仅降 0.94 个点
（77.97% → 77.03%），说明模型学到的是可迁移的规则而非记忆答案；CFSC 严格口径反而略高
（93.58%），因为重叠样本里参数拟合得并不比其他样本更好。

---

## 3. 训练配置与过程

### 3.1 配置

```bash
laya-train \
  --data  laya/all_train.jsonl \      # 53,341 行 / 55,562 题
  --eval  laya/finfe_eval.jsonl \     # 2020（注意：含泄漏，日志会明确标注）
  --out   ckpt \
  --base  multilingual \              # mmBERT-base 322M
  --epochs 4 \
  --micro-batch 24 --grad-accum 3 \   # 有效 batch 72（默认配方 8x8=64）
  --device cuda \
  --shuffle-options \                 # choice 题选项顺序随机，防位置偏置
  --label-smoothing 0.05
```

`--loss rlcd`（默认，未显式传）：RLCD 目标 = 软标签交叉熵项（w_rps=1.0）+ 球面/正则项
（w_sph=0.75），配 4 条 RL 采样、sigma 从 0.4 退火到 0.1。这才是「System-1 决策引擎」的
训练方式——直接拟合决策分布，而非只做分类 argmax。

`laya-train` 训练结束后**自动做两件额外的事**：
1. **温度校准**：choice temperature 1.4378、noul 1.4585（在 10% 校准集上拟合）
2. **弃权阈值拟合**：`choice:3-5` → 0.6217、`choice:11+` → 0.7562、`noul:2` → 0.6661
   （按选项数量分桶，target_error=0.10）

这是 ECE 大幅改善的直接原因，无需手工调。

### 3.2 过程

| 项目 | 数值 |
|---|---|
| 优化步数 | 3,068（4 epoch × 767 步） |
| 吞吐 | ~5 步/s ≈ 360 item/s |
| 显存占用 | 9.2 GB / 24 GB（4090 余量充足，batch 还能翻倍） |
| GPU 利用率 | 81–93% |
| 总耗时 | 约 32 分钟（18:41 → 19:13） |

各 epoch 平均 loss：

```
epoch 1/4  mean loss 0.3983
epoch 2/4  mean loss 0.3103
epoch 3/4  mean loss 0.2565
epoch 4/4  mean loss 0.2018
```

单调下降，无发散。**单个 step loss 出现负值是正常的**——RLCD 是「似然项 − 正则/KL 项」的
形式，当模型对标签的置信度高于惩罚项时整体可为负（epoch 3 step 2000 是 -0.2701）。

### 3.3 训练内评测（不可作为泛化证据）

`train_report.json` 自带 before/after，但里面明确写了：

```json
"eval_mode": "overlapping_eval",
"is_held_out": false,
"note": "Evaluation set contains inputs that overlap training data (1090) and
         calibration data (7) (total 1097/2020 items); metrics do not reflect
         strictly independent held-out evaluation."
```

| 指标 | Before | After | Delta |
|---|---|---|---|
| Accuracy | 0.5604 | 0.7807 | +0.2203 |
| Loss | 1.1470 | 0.6462 | −0.5008 |
| ECE | 0.1821 | 0.0717 | −0.1104 |
| Brier (multi) | 0.6107 | 0.3513 | −0.2594 |
| Mean Confidence | 0.7425 | 0.8513 | +0.1088 |

数字与独立评测（77.97%）吻合，但**不要**引用为泛化性能——引用 §1 的表。

---

## 4. 产物清单

### 4.1 服务器 `/data/laya-fin/`

```
ckpt/
  model.safetensors            644 MB   ← 微调后权重（bf16/fp32 混合导出）
  rl_agent_config.json         5.4 KB   ← 温度、弃权阈值、题面定义
  questions.json               4.5 KB   ← 3 个 qid 的完整定义（推理端据此复原选项空间）
  train_report.json            2.8 KB   ← 训练超参 + before/after + 校准/弃权结果
  tokenizer/                   33.6 MB
  checkpoint_latest/           运行时快照
laya/                          9 个 JSONL（102 MB）
reports/                       base_*.json（zero-shot）+ ft_*.json（微调后）
train.sh / baseline.sh / eval_ft.sh / run_eval.py / check_leak.py / strict_acc.py（后四个现已移至 common/laya/）
```

### 4.2 本地仓库结构（`FinLaya/`）

配置、tokenizer、全部评测报告（JSON + Markdown）、训练日志都在 `FinLaya/` 下：

```
FinLaya/
├─ README.md                   目录导航 + 运行顺序 + 数字速查
├─ REPORT.md                   本文
├─ scripts/                    train.sh · baseline.sh · eval_ft.sh
../common/laya/                 run_eval.py · laya_metrics.py · check_leak.py · strict_acc.py
├─ configs/                    rl_agent_config.json · questions.json · train_report.json
├─ logs/                       train_run / dryrun / baseline / eval_ft
├─ reports/eval/               ft_*.json + ft_*_metrics{,_strict}.json
├─ reports/baseline/           base_*.json
└─ tokenizer/                  33.6 MB
```

**权重 644 MB 未拉回**——如需本地推理，从服务器 `/data/laya-fin/ckpt/` 取。
两模型横向对照已移至 `docs/COMPARE_Laya_vs_FinBERT.md`。

---

## 5. 踩坑记录（复用价值高）

### 5.1 `laya-evals` 的 `--model` 只认别名

```bash
laya-evals run x.jsonl --model convaiinnovations/laya
# laya-evals: unknown model '...'; choose one of
#   ['english','multilingual','typed-decisions'] (or an alias: [...])
laya-evals run x.jsonl --model /data/laya-fin/ckpt    # 同样被拒
```

`router.resolve()` 只查注册表（`english`/`multilingual`/`typed-decisions` 及别名
`ml`/`multi`/`laya`/`default`…）。**基线评测用 `--model ml`**；**微调后评测必须走 Python API**：

```python
router = laya.Router(device="cuda", preload=False)
name = router.register("fin-laya", "/data/laya-fin/ckpt")   # 官方支持本地目录
report = evaluate(RouterRunner(router), Dataset.from_jsonl(tmp), batch_size=16, on_error="skip")
json.dump(report.to_json(), f)
```

注意 `evaluate(runner, dataset, ...)` 是位置参数，且返回值 `EvalReport` **不是 dict**——
用 `report.to_json()` / `report.to_markdown()`，没有 `to_dict()`。

### 5.2 `--dry-run` 显示的 batch 不是最终 batch

dryrun 输出 `micro_batch 8 x grad_accum 8` 是**未显式传参时的默认配方**，不是自动下调。
显式传 `--micro-batch 24 --grad-accum 3` 后就是 72（显存才 9.2GB）。

### 5.3 服务器网络

| 目标 | 可达 | 备注 |
|---|---|---|
| PyPI（北外源） | ✅ | 阿里云源装 `transformers` 会卡死，换北外 |
| `hf-mirror.com` | ✅ | `HF_ENDPOINT` + `HF_HOME` 后权重正常拉取（647MB） |
| `github.com` / `raw.githubusercontent.com` / `ghproxy.net` | ❌ | 全部不通 |

之前从本地探测到 `/api/models/convaiinnovations/laya/tree/main` 返回 403，但服务器上
`snapshot_download` **可以正常拉**，两者不矛盾（API 端点与 CDN 路径不同）。

### 5.4 其他

- SFTP 偶发 `ENOENT`（`put`/`putfo`/`chmod` 都可能报）→ 「写本地临时文件 + `sftp.put` +
  重连重试」在**内联单进程**里稳定成功
- Windows CRLF 导致 `bash: $'\r': command not found` → 上传前 `replace(b"\r\n", b"\n")`
- 前台命令约 2 分钟被 SIGTERM → 长任务一律后台 + 轮询远端日志
- `nohup … &` 会占住 SSH 通道导致 TimeoutError → 用 `setsid … < /dev/null & exit 0` 后立即关 channel

---

## 6. 下一步建议

1. **重训 FinFE 用干净切分**：把 `finfe_train` 里与 `eval_list` 重叠的 1,097 条剔除后重训，
   得到可正式引用的 held-out 数字（当前严格口径 77.03% 已是良好估计，但样本只剩 923 条）。
2. **拉高 batch**：显存只用了 9.2GB，`micro-batch 48 × grad-accum 2`（有效 96）应能再快 1.3×，
   同时略改善 RLCD 的梯度估计质量。
3. **扩 `neg_subject` 词表**：当前 39% 验证正例落在 20 词表外。若业务允许动态选项，
   可改成一题一 qid（如 `neg_subject#01`）绕开 schema 一致性约束。
4. **用弃权阈值做线上门控**：`rl_agent_config.json` 里已拟合好分桶阈值，直接接进服务端；
   CFSC-ABSA 在 confidence@80 下准确率 98.08%，剩下 20% 转人工。
5. **导出 ONNX**：`laya-evals run --onnx` 支持，32ms → 目标 <10ms（4090 上 p50 已有 1.4ms
   的吞吐数字 `cost_per_decision_p50_ms`，但含 batch 分摊，真实单条延迟看 `latency_p50_ms`）。

---

## 7. 数字速查

```
基线（zero-shot，--model ml）
  FinFE       choice_acc 0.5579  ece 0.1848  conf 0.7427
  FinNSP      choice_acc 0.1351  noul_acc 0.6800  ece 0.3554
  CFSC-ABSA   choice_acc 0.6896  ece 0.1508  conf 0.8404

微调后
  FinFE       choice_acc 0.7797  ece 0.0739  sel@80 0.8379
  FinNSP      choice_acc 0.9691  noul_acc 0.9500  ece 0.0998  sel@80 0.9671
  CFSC-ABSA   choice_acc 0.9337  ece 0.0638  sel@80 0.9808

严格口径（剔除与训练集 state 重叠的样本）
  FinFE       0.7703 (n=923)
  FinNSP      0.9691 (n=259) / noul 0.9499 (n=499)
  CFSC-ABSA   0.9358 (n=7870)
```
