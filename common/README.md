# common — 三个模型共用的测量与汇总

这里放不归属单一模型的脚本：部署指标测量、对照报告生成。

## 目录结构

```
common/
├─ deploy_metrics.py     统一口径测参数量 / 权重体积 / 延迟 / 吞吐 / 显存
├─ to_fp16.py            把 checkpoint 转 float16 并报体积（复现 195.1 MB 那一项）
├─ make_compare.py       把 Laya vs FinBERT 两模型指标汇总成 Markdown 对照报告
└─ reports/
   ├─ deploy_metrics.json  三个模型的部署指标实测结果（finbert / laya / fb_laya）
   ├─ deploy_fb.log        FinBERT 侧的测量输出
   └─ deploy_laya.log      Laya 侧的测量输出
```

## 为什么必须分进程跑

`deploy_metrics.py` 的三个子命令依赖不同 conda 环境的 torch，
同一个进程里连续执行会互相干扰（甚至静默给出错误的显存/延迟数字）。**必须分开调用**：

```bash
/data/miniconda/envs/finbert/bin/python common/deploy_metrics.py finbert
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py laya
/data/miniconda/envs/laya/bin/python     common/deploy_metrics.py fb_laya
```

三次都写同一个 `reports/deploy_metrics.json`，各自只更新自己那个键（`finbert` / `laya` / `fb_laya`），不会互相覆盖。

`laya` 与 `fb_laya` 走的是同一条 `bench_laya()` 推理路径，只是 `--base` 的 encoder 不同，所以口径天然可比：

```python
bench_laya()                                      # mmBERT 版，ckpt = LAYA_CKPT
bench_laya(FB_LAYA_CKPT, model_key="fin-laya-fb") # FinBERT 版，ckpt = FB_LAYA_CKPT
```

## 测量口径

| 指标 | FinBERT 实现 | Laya 实现 |
|---|---|---|
| 参数量 | `sum(p.numel())` | 读 safetensors header（`struct.unpack("<Q")`）累加各张量形状 |
| 权重体积 | 遍历目录累加 `.safetensors/.bin/.pt` | 同上 |
| 延迟 p50 / p95 | `transformers` 前向，batch=1，200 次取分位，先预热 | `Router.predict`，单条，100 次取分位 |
| 吞吐 | batch=1 测 200 次；batch=32 测 30 次 | batch=1 |
| 显存峰值 | `torch.cuda.max_memory_allocated()`（batch=32） | 同上（单条） |

两次测量都在 RTX 4090 24GB 上，其余进程空闲。

## 实测结果

| | Laya (mmBERT) | FinBERT-Laya | FinBERT 分类头 |
|---|---:|---:|---:|
| 参数量 | 321.9 M | **117.2 M** | 102.3 M |
| 权重体积 | 614.0 MB（170 张量全 F16） | **223.6 MB**（235 张量全 F16） | 390 MB（fp32）→ 195 MB（F16） |
| 延迟 p50 | 18.64 ms | **8.66 ms** | 5.93 ms |
| 延迟 p95 | 19.34 ms | **8.92 ms** | 5.98 ms |
| 吞吐 batch=1 | 53.0 QPS | **113.0 QPS** | 145.3 QPS |
| 吞吐 batch=32 | — | — | 2857.7 QPS |
| 显存峰值 | 1499 MB | **667 MB** | 486 MB |

> 体积口径：`weights_only_mb` / `dir_size_mb` 默认跳过 `checkpoint_latest/`。那是训练中途的快照，和顶层 `model.safetensors` 是同一份权重，算进去会让体积接近翻倍（Laya 会虚报成 1228 MB、FinBERT-Laya 虚报成 447 MB）。

> FinBERT 转 F16 后 195.1 MB 是另一轮单独实测的结论，当时用的转换代码没有留下来。
> 现在补了 `to_fp16.py` 按同样做法重写，可复现这一项：
>
> ```bash
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --check    # 先看 dtype 分布
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --dry-run  # 只估体积
> /data/miniconda/envs/finbert/bin/python common/to_fp16.py <model_dir> --out <dir>
> ```
>
> 该脚本只做了语法校验，**未在服务器上重跑复现**；fp32 的 390 MB 是确定可复现的。

## 生成对照报告

```bash
python common/make_compare.py       # → docs/COMPARE_Laya_vs_FinBERT.md
```

数据来源：

| 文件 | 内容 |
|---|---|
| `FinLaya/reports/eval/ft_*_metrics.json` | Laya 全量口径 |
| `FinLaya/reports/eval/ft_*_metrics_strict.json` | Laya 严格口径 |
| `FinBert/reports/finbert_all.json` | FinBERT 全量 + 严格口径 |
| `common/reports/deploy_metrics.json` | 两者部署指标 |

路径可用 `LAYA_REPORTS` 覆盖，默认取本目录下的 `reports/`。
