# FinBERT-Laya weights

FinBERT encoder + Laya decision head, 117.24M parameters, fp16. Trained on CFSC-ABSA / FinFE / FinNSP.

## Why the file is split

`model.safetensors` is 223.6 MB, over GitHub's 100 MB per-file limit. This machine has no git-lfs, and the HTTPS endpoints LFS depends on are unreachable from the current network. So the file is committed as 3 parts; merging takes one command.

| File | Size |
|---|---:|
| `model.safetensors.part-00` | 80 MB |
| `model.safetensors.part-01` | 80 MB |
| `model.safetensors.part-02` | 64 MB |
| Merged | 223.6 MB |

## Merging

```bash
cd weights/finbert-laya
bash restore.sh
```

Output:

```
[*] merging parts ...
[*] merged: 224M
[*] verifying SHA256 ... OK
```

Expected value: `745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3`

Verify only, without merging: `bash restore.sh --check`

## Contents

```
finbert-laya/
├─ model.safetensors.part-00/01/02   weight shards (fp16, 235 tensors)
├─ rl_agent_config.json              encoder / max_len / temperatures / abstention thresholds
├─ encoder/config.json               BertConfig (hidden 768, 12 layers, vocab 21128)
├─ tokenizer/                        BertTokenizerFast (vocab.txt + tokenizer.json)
├─ SHA256SUMS                        checksum of the merged weights
└─ restore.sh                        merge script
```

Once merged, the directory is the standard layout `laya.load()` expects, so `Router.register(name, path)` works directly.

## Loading

```python
import laya

router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "weights/finbert-laya")
ans = router.predict("<Chinese financial sentence>", questions, model=name)
```

The `questions` definitions live in [`FinLaya/configs/questions.json`](../../FinLaya/configs/questions.json).

## Inference parameters fitted during training

| Item | Value |
|---|---|
| choice temperature | 1.3531 |
| noul temperature | 1.9008 |
| Abstention threshold `choice:3-5` | 0.5968 |
| Abstention threshold `noul:2` | 0.8953 |
| Abstention threshold `choice:11+` | 0.7937 |

## Measured performance

Single RTX 4090, batch=1 serial:

| | Value |
|---|---:|
| Parameters | 117.24 M |
| Weight size | 223.6 MB |
| Latency p50 / p95 | 8.66 / 8.92 ms |
| Throughput | 113.0 QPS |
| Peak GPU memory | 667 MB |

Accuracy (full / strict protocol):

| Task | Accuracy | Macro-F1 |
|---|---|---|
| FinFE social sentiment | 78.02 / **80.28** | 0.7629 / 0.7371 |
| FinNSP negative presence | 95.20 / 95.19 | 0.9520 / 0.9518 |
| FinNSP negative subject | 95.75 / 95.75 | 0.4669 (0.7003 over the 14 classes with support) |
| CFSC-ABSA aspect-level | 93.01 / 93.01 | 0.8909 / 0.8905 |

Full comparison in [`docs/THREE_WAY_COMPARISON.md`](../../docs/THREE_WAY_COMPARISON.md).

## Uploading to HuggingFace

The merged weights are already assembled for upload (hard links, no extra disk use), so they can be dragged in directly:

```
weights/hf-ready/          9 files / 224.2 MB
```

Steps, CLI commands, and a post-upload checklist are in [`../HF_UPLOAD_GUIDE.md`](../HF_UPLOAD_GUIDE.md).

## FinBERT's four classifier weights

**Not in this repository.** Each head is 391 MB in fp32 (1.56 GB total), which would push the repo past 1 GB. Retraining takes only 5.4 minutes:

```bash
bash FinBert/scripts/setup_finbert.sh
/data/miniconda/envs/finbert/bin/python FinBert/scripts/finbert_baseline.py train
```

The resulting metrics match `FinBert/reports/finbert_all.json`.
