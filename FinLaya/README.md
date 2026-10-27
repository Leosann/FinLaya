# FinLaya — Laya 决策模型的中文金融微调

本目录是 Laya 侧的全部代码、配置、日志与评测报告。主实验结论见 **[REPORT.md](REPORT.md)**，两模型横向对照见 **[../docs/COMPARE_Laya_vs_FinBERT.md](../docs/COMPARE_Laya_vs_FinBERT.md)**。

## 目录结构

```
FinLaya/
├─ REPORT.md                    主实验报告（结论 / 数据 / 训练过程 / 踩坑 / 数字速查）
├─ scripts/                     可执行脚本，按运行顺序排列
│  ├─ train.sh                  dryrun | train | eval | all
│  ├─ baseline.sh               未微调基座 zero-shot 基线
│  ├─ eval_ft.sh                微调后评测（转调 run_eval.py）
│  ├─ run_eval.py               [核心] Router.register 本地 ckpt + evaluate
│  ├─ laya_metrics.py           从报告复算全套指标，支持 --strict
│  ├─ check_leak.py             训练/测试集 state 重叠审计
│  └─ strict_acc.py             按报告 cases 行序对齐复算严格口径
├─ configs/
│  ├─ rl_agent_config.json      训练自动产出的推理配置（温度 / 弃权阈值 / 题面定义）
│  ├─ questions.json            4 个 qid 的完整定义
│  └─ train_report.json         before/after + 温度标定 + 弃权阈值
├─ logs/                        train_run / dryrun / baseline / eval_ft
├─ reports/
│  ├─ eval/                     ft_* 评测报告 + 复算指标（全量 & 严格）
│  └─ baseline/                 base_* 未微调基线报告
└─ tokenizer/                   ckpt 配套 tokenizer（3 个文件）
```

## 运行顺序

```bash
# 0) 数据（在仓库根的 datasets/ 下生成后上传）
python ../datasets/to_laya_format.py
python ../datasets/validate_laya_format.py

# 1) 空跑，确认 53,341 行 / 55,562 题能被正确解析
bash scripts/train.sh dryrun

# 2) 训练（4 epoch，4090 上约 32 分钟，显存 9.2GB）
bash scripts/train.sh train

# 3) 评测三个测试集
bash scripts/eval_ft.sh

# 4) 复算全套指标（全量 + 严格口径）
DATA=${LAYA_DATA:-/data/laya-fin/laya}
for f in finfe_eval finnsp_eval cfsc_absa_test; do
  python scripts/laya_metrics.py reports/eval/ft_${f}.json ${DATA}/${f}.jsonl
  python scripts/laya_metrics.py reports/eval/ft_${f}.json ${DATA}/${f}.jsonl --strict
done

# 5) 重叠审计（可选，结论已写在 REPORT.md §2）
python scripts/check_leak.py
```

## 两个必须知道的坑

1. **`laya-evals run --model <本地目录>` 一定失败**。该参数只走注册表别名（`english` / `multilingual` / `ml` / `laya` / `default` / `typed-decisions`），传路径会被 `router.resolve()` 拒。本地 ckpt 只能通过官方 `Router.register(name, source)` 注册，`run_eval.py` 封装了这一层。
2. **`EvalReport` 不是 dict**。序列化用 `report.to_json()` / `report.to_markdown()`，没有 `to_dict()`。

## 关键数字速查

| 任务 | n（全量/严格） | acc | mF1 | ECE | 严格 acc |
|---|---|---:|---:|---:|---:|
| FinFE `sentiment_social` | 2020 / 923 | 77.97% | 0.7690 | 0.0739 | 77.03% |
| FinNSP `neg_has` | 500 / 499 | 95.00% | 0.9499 | 0.0506 | 94.99% |
| FinNSP `neg_subject` | 259 / 259 | 96.91% | 0.5419¹ | 0.1984 | 96.91% |
| CFSC `sentiment_absa` | 8312 / 7870 | 93.37% | 0.8993 | 0.0638 | 93.58% |

¹ 21 类里 7 类零支持被强制计 0；只看有样本的 14 类为 **0.8129**。

## 环境变量

脚本里没有硬编码路径，全部走环境变量且带默认值（默认值即本次实验的服务器布局）：

| 变量 | 默认值 |
|---|---|
| `LAYA_ROOT` | `/data/laya-fin` |
| `LAYA_DATA` | `${LAYA_ROOT}/laya` |
| `LAYA_CKPT` | `${LAYA_ROOT}/ckpt` |
| `LAYA_REPORTS` | `${LAYA_ROOT}/reports` |
| `LAYA_ENV` | `/data/miniconda/envs/laya` |
| `HF_ENDPOINT` | `https://hf-mirror.com` |
| `EVAL_BATCH` | `16` |

## 未保存的产物

- **训练好的权重（644 MB）不在本仓库**，只保留了 `configs/` 与 `tokenizer/`。重训按上面第 2 步即可。
- 日志与报告里出现的 `/data/laya-fin/...` 是当时服务器上的真实路径，保留原样以便与 `logs/` 逐行对应。
