# FinBERT-Laya 权重

FinBERT encoder + Laya 决策头，117.24M 参数，fp16。在 CFSC-ABSA / FinFE / FinNSP 三任务上训练完成。

## 为什么是分片的

`model.safetensors` 有 223.6 MB，超过 GitHub 单文件 100 MB 的限制，而本机没有 git-lfs，且 LFS 依赖的 HTTPS 端点在当前网络下不可达。所以切成 3 片随仓库提交，合并只需一行命令。

| 文件 | 大小 |
|---|---:|
| `model.safetensors.part-00` | 80 MB |
| `model.safetensors.part-01` | 80 MB |
| `model.safetensors.part-02` | 64 MB |
| 合并后 | 223.6 MB |

## 合并

```bash
cd weights/finbert-laya
bash restore.sh
```

输出：

```
[*] 合并分片 ...
[*] 合并完成：224M
[*] 校验 SHA256 ... OK
```

校验值：`745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3`

只校验不合并：`bash restore.sh --check`

## 目录内容

```
finbert-laya/
├─ model.safetensors.part-00/01/02   分片权重（fp16，235 个张量）
├─ rl_agent_config.json              encoder / max_len / 温度 / 弃权阈值
├─ encoder/config.json               BertConfig（hidden 768，12 层，vocab 21128）
├─ tokenizer/                        BertTokenizerFast（vocab.txt + tokenizer.json）
├─ SHA256SUMS                        合并后权重校验
└─ restore.sh                        合并脚本
```

合并后即为 `laya.load()` 认的标准布局，可直接 `Router.register(name, path)`。

## 加载

```python
import laya

router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "weights/finbert-laya")
ans = router.predict("试仓500股，看看门道！", questions, model=name)
```

`questions` 定义见 [`FinLaya/configs/questions.json`](../../FinLaya/configs/questions.json)。

## 训练时自动拟合的推理参数

| 项 | 值 |
|---|---|
| choice 温度 | 1.3531 |
| noul 温度 | 1.9008 |
| 弃权阈值 `choice:3-5` | 0.5968 |
| 弃权阈值 `noul:2` | 0.8953 |
| 弃权阈值 `choice:11+` | 0.7937 |

## 实测性能

单卡 RTX 4090，batch=1 串行：

| | 值 |
|---|---:|
| 参数量 | 117.24 M |
| 权重体积 | 223.6 MB |
| 延迟 p50 / p95 | 8.66 / 8.92 ms |
| 吞吐 | 113.0 QPS |
| 显存峰值 | 667 MB |

准确率（全量 / 严格口径）：

| 任务 | 准确率 | Macro-F1 |
|---|---|---|
| FinFE 社交情绪 | 78.02 / **80.28** | 0.7629 / 0.7371 |
| FinNSP 有无负面 | 95.20 / 95.19 | 0.9520 / 0.9518 |
| FinNSP 负面主体 | 95.75 / 95.75 | 0.4669（有样本 14 类 0.7003） |
| CFSC-ABSA 方面级 | 93.01 / 93.01 | 0.8909 / 0.8905 |

完整对比见 [`docs/THREE_WAY_COMPARISON.md`](../../docs/THREE_WAY_COMPARISON.md)。

## 上传到 HuggingFace

完整权重的合并版已经组装好（硬链接，不额外占空间），可直接拖拽上传：

```
weights/hf-ready/          9 个文件 / 224.2 MB
```

上传步骤、CLI 命令、上传后自查清单见 [`../HF_UPLOAD_GUIDE.md`](../HF_UPLOAD_GUIDE.md)。

## FinBERT 四个分类头的权重

**不在本仓库**。四个头是 fp32 各 391 MB（合计 1.56 GB），放进来会让仓库超过 1 GB。但重训只要 5.4 分钟：

```bash
bash FinBert/scripts/setup_finbert.sh
/data/miniconda/envs/finbert/bin/python FinBert/scripts/finbert_baseline.py train
```

复现出的指标与 `FinBert/reports/finbert_all.json` 一致。
