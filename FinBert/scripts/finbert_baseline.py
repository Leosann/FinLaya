# -*- coding: utf-8 -*-
"""FinBERT 中文基线：三个任务各自的分类头微调 + 全套指标评测。

对应 Laya 的三种决策原语：
  sentiment_absa   (CFSC-ABSA)  -> 3 分类 choice
  sentiment_social (FinFE)      -> 3 分类 choice
  neg_has          (FinNSP)     -> 2 分类 noul
  neg_subject      (FinNSP)     -> 21 分类 choice（20 实体 + 以上都不是）

训练：head + 全参数微调，CE 损失，AdamW，线性 warmup。
评测：Macro-F1 / 各类别 P/R/F1 / ECE(15 bin) / Brier / NLL / 选择性准确率 / 延迟 / 吞吐 / 显存 / 体积。
严格口径：与训练集 state 哈希重叠的样本单独标记，另出一套剔除后的指标。

用法:
  python finbert_baseline.py train
  python finbert_baseline.py eval
  python finbert_baseline.py all
"""
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np

ROOT = os.environ.get("LAYA_ROOT", "/data/laya-fin")
DATA = os.environ.get("LAYA_DATA", os.path.join(ROOT, "laya"))
OUT = os.environ.get("FINBERT_CKPT", os.path.join(ROOT, "finbert"))
REPORTS = os.environ.get("LAYA_REPORTS", os.path.join(ROOT, "reports"))
MODEL_NAME = "yiyanghkust/finbert-tone-chinese"

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TASKS = {
    # name: (train_file, eval_file, qid, label_source)
    "absa": ("cfsc_absa_train.jsonl", "cfsc_absa_test.jsonl", "sentiment_absa"),
    "finfe": ("finfe_train.jsonl", "finfe_eval.jsonl", "sentiment_social"),
    "nsp_has": ("finnsp_train.jsonl", "finnsp_eval.jsonl", "neg_has"),
    "nsp_subject": ("finnsp_train.jsonl", "finnsp_eval.jsonl", "neg_subject"),
}


def sha(t):
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()


def load_rows(path, qid):
    """返回 [(text, label_str)]，只保留有 expected[qid] 的行。"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            exp = (d.get("expected") or {}).get(qid)
            if exp is None:
                continue
            text = d["state"]["text"] if isinstance(d["state"], dict) else d["state"]
            label = "true" if exp is True else ("false" if exp is False else str(exp))
            rows.append((text, label))
    return rows


def build_label_space(task):
    """标签空间只从训练集统计（避免测试集泄漏进标签定义）；noul 固定为 true/false。"""
    tr_file, _, qid = TASKS[task]
    labels = sorted({lab for _, lab in load_rows(os.path.join(DATA, tr_file), qid)})
    if qid == "neg_has":
        labels = ["false", "true"]          # 固定，防止训练集单类时崩掉
    return labels


def train_task(task, epochs=3, bs=32, lr=2e-5, max_len=256, seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset as TorchDS
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tr_file, _, qid = TASKS[task]
    rows = load_rows(os.path.join(DATA, tr_file), qid)
    labels = build_label_space(task)
    lab2id = {l: i for i, l in enumerate(labels)}
    print("[%s] train rows=%d labels=%d" % (task, len(rows), len(labels)), flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels), ignore_mismatched_sizes=True).cuda()

    class DS(TorchDS):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            t, l = self.items[i]
            return t, lab2id[l]

    def collate(batch):
        texts = [b[0] for b in batch]
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        enc = tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = ys
        return enc

    dl = DataLoader(DS(rows), batch_size=bs, shuffle=True, collate_fn=collate, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = len(dl) * epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    t0 = time.time()
    for ep in range(epochs):
        tot = n = 0
        for step, batch in enumerate(dl):
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.cuda.amp.autocast(dtype=torch.float16):
                out = model(**batch)
            scaler.scale(out.loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            opt.zero_grad(set_to_none=True)
            tot += float(out.loss)
            n += 1
            if (step + 1) % 100 == 0:
                print("  [%s] ep%d step %d loss %.4f" % (task, ep + 1, step + 1, tot / n), flush=True)
        print("  [%s] epoch %d mean loss %.4f  (%.0fs)" % (task, ep + 1, tot / n, time.time() - t0), flush=True)

    d = os.path.join(OUT, task)
    os.makedirs(d, exist_ok=True)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    json.dump({"labels": labels, "qid": qid, "max_len": max_len, "epochs": epochs,
               "base": MODEL_NAME, "train_rows": len(rows)},
              open(os.path.join(d, "label_map.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("[%s] saved -> %s (%.0fs)" % (task, d, time.time() - t0), flush=True)
    del model
    torch.cuda.empty_cache()


def softmax_np(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def ece_score(probs, correct, n_bins=15):
    """按 max-prob（top-1 置信度）分箱的标准 ECE。"""
    conf = probs.max(axis=1)
    pred_ok = correct.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1]) if i else (conf >= bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(pred_ok[m].mean() - conf[m].mean())
    return float(ece)


def brier_multiclass(probs, y):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def brier_top1(probs, y):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    ok = (pred == y).astype(float)
    return float(((conf - ok) ** 2).mean())


def nll_score(probs, y):
    p = np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


def aurc(probs, y):
    """风险-覆盖曲线下面积（越低越好）。"""
    conf = probs.max(axis=1)
    ok = (probs.argmax(axis=1) == y).astype(float)
    # kind="stable" 与 common/laya/laya_metrics.py 保持一致：置信度并列时，
    # 默认 quicksort 的顺序随 numpy 版本变化，会让 aurc 不可复现。
    order = np.argsort(-conf, kind="stable")
    ok = ok[order]
    cov = np.arange(1, len(ok) + 1) / len(ok)
    risk = 1.0 - np.cumsum(ok) / np.arange(1, len(ok) + 1)
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 改了名
    return float(_trapz(risk, cov))


def selective_at(probs, y, target_cov):
    """给定覆盖率目标，返回 (该覆盖率下的准确率, 实际阈值)。"""
    conf = probs.max(axis=1)
    ok = (probs.argmax(axis=1) == y).astype(float)
    k = max(1, int(round(target_cov * len(conf))))
    order = np.argsort(-conf, kind="stable")[:k]   # 同上：截断处并列时顺序必须稳定
    return float(ok[order].mean()), float(conf[order].min())


def eval_task(task, tag, exclude_hashes=None, batch_size=64):
    """评测一个任务。exclude_hashes 非空时只保留不在其中的样本（严格口径）。"""
    import torch
    from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                                 precision_recall_fscore_support)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tr_file, ev_file, qid = TASKS[task]
    d = os.path.join(OUT, task)
    meta = json.load(open(os.path.join(d, "label_map.json"), "r", encoding="utf-8"))
    labels, lab2id, max_len = meta["labels"], None, meta["max_len"]
    lab2id = {l: i for i, l in enumerate(labels)}

    rows = load_rows(os.path.join(DATA, ev_file), qid)
    if exclude_hashes is not None:
        keep = [(t, l) for (t, l) in rows if sha(t) not in exclude_hashes]
        dropped = len(rows) - len(keep)
        rows = keep
    else:
        dropped = 0

    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).cuda().eval()
    n_param = sum(p.numel() for p in model.parameters())
    size_mb = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) / 1048576.0

    y = np.array([lab2id[l] for _, l in rows])
    probs_all = np.zeros((len(rows), len(labels)), dtype=np.float64)
    lat = []

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            enc = tok([t for t, _ in chunk], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt")
            enc = {k: v.cuda() for k, v in enc.items()}
            torch.cuda.synchronize()
            t0 = time.time()
            logits = model(**enc).logits.float()
            torch.cuda.synchronize()
            dt = time.time() - t0
            lat.extend([dt / len(chunk)] * len(chunk))
            probs_all[i:i + len(chunk)] = softmax_np(logits.cpu().numpy())

    peak_mem = torch.cuda.max_memory_allocated() / 1048576.0
    del model
    torch.cuda.empty_cache()

    pred = probs_all.argmax(axis=1)
    correct = (pred == y)
    acc = float(correct.mean())
    macro_f1 = float(f1_score(y, pred, average="macro", labels=list(range(len(labels))), zero_division=0))
    weighted_f1 = float(f1_score(y, pred, average="weighted", labels=list(range(len(labels))), zero_division=0))
    p, r, f, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(labels))), zero_division=0)
    per_class = {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(sup[i])}
                 for i in range(len(labels))}
    cm = confusion_matrix(y, pred, labels=list(range(len(labels))))
    # macro-F1 只在测试集中真实出现的类别上算，与 Laya 侧口径一致
    present = [i for i in range(len(labels)) if sup[i] > 0]
    macro_f1_present = float(f1_score(y, pred, average="macro", labels=present, zero_division=0)) if present else 0.0

    lat_sorted = sorted(lat)
    res = {
        "task": task, "tag": tag, "qid": qid, "n": len(rows), "dropped_overlap": dropped,
        "labels": labels,
        "accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "macro_f1_present": macro_f1_present,
        "n_classes": len(labels), "n_classes_present": len(present),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "ece": ece_score(probs_all, correct), "brier": brier_multiclass(probs_all, y),
        "brier_top1": brier_top1(probs_all, y), "nll": nll_score(probs_all, y),
        "aurc": aurc(probs_all, y),
        "mean_confidence": float(probs_all.max(axis=1).mean()),
        "latency_p50_ms": lat_sorted[len(lat_sorted) // 2] * 1000,
        "latency_p95_ms": lat_sorted[int(0.95 * len(lat_sorted))] * 1000,
        "throughput_qps": len(rows) / sum(lat),
        "peak_gpu_mem_mb": peak_mem,
        "model_dir_mb": size_mb,
        "n_params": n_param,
    }
    # 选择性准确率
    for cov in (0.5, 0.8, 0.9):
        a, th = selective_at(probs_all, y, cov)
        res["selective_acc@%d" % int(cov * 100)] = a
        res["selective_threshold@%d" % int(cov * 100)] = th
    return res


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(REPORTS, exist_ok=True)
    torch_epochs = int(os.environ.get("FB_EPOCHS", "3"))
    torch_bs = int(os.environ.get("FB_BS", "32"))

    if mode in ("train", "all"):
        for task in TASKS:
            train_task(task, epochs=torch_epochs, bs=torch_bs)

    if mode in ("eval", "all"):
        # 训练集 state 哈希：用于严格口径
        train_hashes = set()
        for f in ("cfsc_absa_train.jsonl", "finfe_train.jsonl", "finnsp_train.jsonl"):
            with open(os.path.join(DATA, f), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    dd = json.loads(line)
                    s = dd["state"]["text"] if isinstance(dd["state"], dict) else dd["state"]
                    train_hashes.add(sha(s))
        print("train unique states: %d" % len(train_hashes), flush=True)

        allres = {}
        for task in TASKS:
            r_full = eval_task(task, "full")
            r_clean = eval_task(task, "strict_no_overlap", exclude_hashes=train_hashes)
            allres[task] = {"full": r_full, "strict": r_clean}
            for k, r in (("full", r_full), ("strict", r_clean)):
                print("[%s/%s] n=%d acc=%.4f macroF1=%.4f ece=%.4f nll=%.4f brier=%.4f "
                      "sel@80=%.4f lat_p50=%.1fms qps=%.0f mem=%.0fMB"
                      % (task, k, r["n"], r["accuracy"], r["macro_f1"], r["ece"], r["nll"],
                         r["brier"], r["selective_acc@80"], r["latency_p50_ms"],
                         r["throughput_qps"], r["peak_gpu_mem_mb"]), flush=True)
        json.dump(allres, open(os.path.join(REPORTS, "finbert_all.json"), "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print("FINBERT_DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
