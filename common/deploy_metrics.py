# -*- coding: utf-8 -*-
"""统一口径测量两套模型的部署指标：参数量、权重体积、显存、单条延迟、吞吐。

FinBERT：直接用 transformers 前向，batch=1 与 batch=32 各测一次。
Laya：走 Router 的 predict，同样 batch=1 与批量 predict_batch。

用法: python deploy_metrics.py
"""
import json
import os
import statistics
import sys
import time

import torch

ROOT = os.environ.get("LAYA_ROOT", "/data/laya-fin")
CKPT = os.environ.get("LAYA_CKPT", os.path.join(ROOT, "ckpt"))
FB_LAYA_CKPT = os.environ.get("FB_LAYA_CKPT", os.path.join(ROOT, "fb_ckpt"))
FB = os.environ.get("FINBERT_CKPT", os.path.join(ROOT, "finbert"))
REPORTS = os.environ.get("LAYA_REPORTS", os.path.join(ROOT, "reports"))
OUT = os.path.join(REPORTS, "deploy_metrics.json")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def dir_size_mb(d, skip=("checkpoint_latest",)):
    """目录总大小。默认跳过 `checkpoint_latest/`——那是训练中途的快照，
    和顶层 model.safetensors 是同一份权重，算进去会让体积翻倍。"""
    tot = 0
    for r, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot / 1048576.0


def weights_only_mb(d, skip=("checkpoint_latest",)):
    """只算权重文件（.safetensors/.bin/.pt），跳过训练快照目录。"""
    tot = 0
    for r, dirs, fs in os.walk(d):
        dirs[:] = [x for x in dirs if x not in skip]
        for f in fs:
            if f.endswith((".safetensors", ".bin", ".pt")):
                tot += os.path.getsize(os.path.join(r, f))
    return tot / 1048576.0


def bench_finbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    d = os.path.join(FB, "absa")
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
    n_param = sum(p.numel() for p in model.parameters())

    text = "公司上半年净利润同比增长百分之三十，业绩超出市场预期。" * 3
    res = {"n_params": n_param, "n_params_M": n_param / 1e6,
           "weights_mb": weights_only_mb(d), "dir_mb": dir_size_mb(d)}

    # batch=1 延迟（含 tokenize，模拟真实请求）
    lat = []
    with torch.no_grad():
        for _ in range(20):     # warmup
            enc = tok([text], truncation=True, max_length=256, padding=True, return_tensors="pt")
            model(**{k: v.cuda() for k, v in enc.items()})
        torch.cuda.synchronize()
        for _ in range(200):
            t0 = time.time()
            enc = tok([text], truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(**enc)
            torch.cuda.synchronize()
            lat.append((time.time() - t0) * 1000)
    res["latency_p50_ms"] = statistics.median(lat)
    res["latency_p95_ms"] = sorted(lat)[int(0.95 * len(lat))]
    res["latency_mean_ms"] = statistics.mean(lat)
    res["qps_batch1"] = 1000.0 / statistics.mean(lat)

    # batch=32 吞吐
    texts = [text] * 32
    with torch.no_grad():
        for _ in range(5):
            enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            model(**{k: v.cuda() for k, v in enc.items()})
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(30):
            enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                model(**enc)
        torch.cuda.synchronize()
        dt = time.time() - t0
    res["qps_batch32"] = 32 * 30 / dt
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")
        model(**{k: v.cuda() for k, v in enc.items()})
    res["peak_mem_mb_batch32"] = torch.cuda.max_memory_allocated() / 1048576.0
    res["weight_dtype"] = "fp32"
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return res


def bench_laya(ckpt=None, model_key="fin-laya"):
    """测一个 Laya 格式 checkpoint 的部署指标。

    `ckpt` 缺省用 mmBERT 版（LAYA_CKPT）；测 FinBERT-Laya 时传 FB_LAYA_CKPT。
    两者都是同一套 laya 推理路径，所以口径天然可比。
    """
    import laya
    from laya.evals import Dataset

    ckpt = ckpt or CKPT
    # 用 finfe_eval 前几条构造 state/questions
    ds_path = os.path.join(ROOT, "laya", "finfe_eval.jsonl")
    rows = []
    with open(ds_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["model"] = model_key
            rows.append(d)
            if len(rows) >= 64:
                break

    router = laya.Router(device="cuda", preload=False)
    name = router.register(model_key, ckpt)

    # 预热
    router.predict(rows[0]["state"], rows[0]["questions"], model=name)
    torch.cuda.reset_peak_memory_stats()

    lat = []
    for i in range(100):
        r = rows[i % len(rows)]
        t0 = time.time()
        router.predict(r["state"], r["questions"], model=name)
        torch.cuda.synchronize()
        lat.append((time.time() - t0) * 1000)

    res = {
        "ckpt": ckpt,
        "weights_mb": weights_only_mb(ckpt),
        "dir_mb": dir_size_mb(ckpt),
        "latency_p50_ms": statistics.median(lat),
        "latency_p95_ms": sorted(lat)[int(0.95 * len(lat))],
        "latency_mean_ms": statistics.mean(lat),
        "qps_batch1": 1000.0 / statistics.mean(lat),
    }

    # 遍历 checkpoint 里的 safetensors 头拿参数量（不加载权重，读 header）
    import struct
    p = os.path.join(ckpt, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    tot = 0
    for k, v in header.items():
        if k == "__metadata__":
            continue
        shape = v.get("shape") or []
        if not shape:
            continue
        num = 1
        for s in shape:
            num *= s
        tot += num
    res["n_params"] = tot
    res["n_params_M"] = tot / 1e6
    # dtype 统计
    dtypes = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        dtypes[v.get("dtype", "?")] = dtypes.get(v.get("dtype", "?"), 0) + 1
    res["weight_dtypes"] = dtypes
    res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1048576.0
    return res


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    out = {}
    if os.path.exists(OUT):
        try:
            out = json.load(open(OUT, "r", encoding="utf-8"))
        except Exception:
            out = {}
    if part in ("finbert", "all"):
        print(">>> bench finbert", flush=True)
        out["finbert"] = bench_finbert()
        print(json.dumps(out["finbert"], indent=1), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if part in ("laya", "all"):
        print(">>> bench laya (mmBERT encoder)", flush=True)
        out["laya"] = bench_laya()
        print(json.dumps(out["laya"], indent=1), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if part in ("fb_laya", "all"):
        print(">>> bench fb_laya (FinBERT encoder)", flush=True)
        out["fb_laya"] = bench_laya(FB_LAYA_CKPT, model_key="fin-laya-fb")
        print(json.dumps(out["fb_laya"], indent=1), flush=True)
        json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("DEPLOY_DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
