# -*- coding: utf-8 -*-
"""从 laya-evals 报告复算全套指标（Macro-F1 / 各类召回 / NLL / Brier / 选择性准确率），
并支持剔除训练集重叠样本的严格口径。

用法: python laya_metrics.py <report.json> <dataset.jsonl> [--strict]
"""
import hashlib
import json
import os
import sys

import numpy as np

DATA = os.environ.get("LAYA_DATA", "/data/laya-fin/laya")


def sha(t):
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()


def train_hashes():
    out = set()
    for f in ("cfsc_absa_train.jsonl", "finfe_train.jsonl", "finnsp_train.jsonl"):
        p = os.path.join(DATA, f)
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                s = d["state"]["text"] if isinstance(d["state"], dict) else d["state"]
                out.add(sha(s))
    return out


def report_hashes(dataset):
    """按 cases 顺序还原每行的 state 哈希（同一行多题会重复出现）。"""
    hs = []
    with open(dataset, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s = d["state"]["text"] if isinstance(d["state"], dict) else d["state"]
            h = sha(s)
            for _ in (d.get("expected") or {}):
                hs.append(h)
    return hs


def ece_score(probs, correct, n_bins=15):
    conf = probs.max(axis=1)
    ok = correct.astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = ((conf >= bins[i]) & (conf <= bins[i + 1])) if i == 0 else ((conf > bins[i]) & (conf <= bins[i + 1]))
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(ece)


def brier_mc(probs, y):
    oh = np.zeros_like(probs)
    oh[np.arange(len(y)), y] = 1
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def nll_s(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def aurc_s(probs, y):
    conf = probs.max(axis=1)
    ok = (probs.argmax(axis=1) == y).astype(float)
    # kind="stable"：报告里的概率是 4 位小数，置信度并列很常见（FinFE 2020 条里有 1051 条重复）。
    # 默认的 quicksort 不稳定，并列顺序随 numpy 版本/平台变化，会让 aurc 和 selective_acc 不可复现。
    o = np.argsort(-conf, kind="stable")
    ok = ok[o]
    cov = np.arange(1, len(ok) + 1) / len(ok)
    risk = 1.0 - np.cumsum(ok) / np.arange(1, len(ok) + 1)
    _trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2 改了名
    return float(_trapz(risk, cov))


def selective_at(probs, y, cov):
    conf = probs.max(axis=1)
    ok = (probs.argmax(axis=1) == y).astype(float)
    k = max(1, int(round(cov * len(conf))))
    o = np.argsort(-conf, kind="stable")[:k]   # 同上：并列时必须稳定排序，否则截断处的样本会漂移
    return float(ok[o].mean()), float(conf[o].min())


def confusion_and_prf(y, pred, classes):
    """不依赖 sklearn 的混淆矩阵 + 每类 P/R/F1。"""
    C = len(classes)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    per = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(cm[i, :].sum())}
    macro_f1 = float(np.mean([per[c]["f1"] for c in classes]))
    # 只在测试集中真实出现过的类别上算 macro-F1（零支持类别强制计 0 会系统性压低均值）
    present = [c for c in classes if per[c]["support"] > 0]
    macro_f1_present = float(np.mean([per[c]["f1"] for c in present])) if present else 0.0
    # weighted F1
    tot = sum(per[c]["support"] for c in classes)
    wf1 = float(sum(per[c]["f1"] * per[c]["support"] for c in classes) / tot) if tot else 0.0
    return cm.tolist(), per, macro_f1, wf1, macro_f1_present, len(present)


def main():
    if len(sys.argv) < 3:
        print("用法: python laya_metrics.py <report.json> <dataset.jsonl> [--strict]")
        return 2
    report_path, dataset = sys.argv[1], sys.argv[2]
    strict = "--strict" in sys.argv
    if not os.path.exists(report_path):
        print("找不到报告文件 %s" % report_path)
        return 1
    if not os.path.exists(dataset):
        print("找不到数据集 %s" % dataset)
        return 1
    rep = json.load(open(report_path, "r", encoding="utf-8"))
    if "cases" not in rep:
        # 最常见的原因是误把 *_metrics.json 当成 laya-evals 的原始报告传进来
        print("%s 里没有 cases 字段 —— 需要 laya-evals 的原始报告，不是 *_metrics.json" % report_path)
        return 1
    cases = rep["cases"]
    hs = report_hashes(dataset)
    if len(hs) != len(cases):
        print("对齐失败: cases=%d hashes=%d" % (len(cases), len(hs)))
        return 1
    tr = train_hashes() if strict else None

    by_q = {}
    for h, c in zip(hs, cases):
        if strict and h in tr:
            continue
        by_q.setdefault(c["qid"], []).append((h, c))

    out = {}
    for qid, items in sorted(by_q.items()):
        ans0 = items[0][1]["answer"]
        preds = None
        # noul 题：概率是 {true: p, false: 1-p} 形式或直接 confidence
        if ans0.get("type") == "noul":
            # laya 的 noul 答案：answer["noul"] 是 True 的概率（不是 bool！），
            # confidence / answer_confidence 是 |2p-1| 形式的决策置信度。
            classes = ["true", "false"]
            probs, y, preds = [], [], []
            for _, c in items:
                a = c["answer"]
                p = a.get("probabilities") or {}
                if "true" in p and "false" in p:
                    pt = float(p["true"])
                else:
                    pt = float(a.get("noul", 0.5))       # 直接用 noul 概率
                pt = min(max(pt, 0.0), 1.0)
                probs.append([pt, 1 - pt])
                y.append(0 if c["expected"] is True else 1)
                preds.append(0 if pt >= 0.5 else 1)
        else:
            classes = sorted({c["expected"] for _, c in items} | set(ans0.get("probabilities", {}).keys()))
            probs, y = [], []
            for _, c in items:
                p = c["answer"].get("probabilities") or {}
                probs.append([float(p.get(k, 0.0)) for k in classes])
                y.append(classes.index(c["expected"]))
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum(axis=1, keepdims=True)
        y = np.array(y)
        pred = np.array(preds) if preds else probs.argmax(axis=1)
        correct = pred == y
        cm, per, mf1, wf1, mf1_present, n_present = confusion_and_prf(y, pred, classes)
        r = {
            "qid": qid, "n": len(items), "classes": classes,
            "accuracy": float(correct.mean()), "macro_f1": mf1, "weighted_f1": wf1,
            "macro_f1_present": mf1_present, "n_classes_present": n_present,
            "n_classes": len(classes),
            "per_class": per, "confusion_matrix": cm,
            "ece": ece_score(probs, correct), "brier": brier_mc(probs, y), "nll": nll_s(probs, y),
            "aurc": aurc_s(probs, y), "mean_confidence": float(probs.max(axis=1).mean()),
        }
        for cov in (0.5, 0.8, 0.9):
            a, th = selective_at(probs, y, cov)
            r["selective_acc@%d" % int(cov * 100)] = a
            r["selective_threshold@%d" % int(cov * 100)] = th
        out[qid] = r
        print("== %s (%s) n=%d ==" % (qid, "严格口径" if strict else "全量", len(items)))
        print("   acc=%.4f  macroF1=%.4f (有样本类=%.4f, %d/%d 类)  wF1=%.4f  ece=%.4f  nll=%.4f  brier=%.4f  aurc=%.4f"
              % (r["accuracy"], r["macro_f1"], mf1_present, n_present, len(classes),
                 r["weighted_f1"], r["ece"], r["nll"], r["brier"], r["aurc"]))
        for c in classes:
            d = per[c]
            print("   %-12s P=%.3f R=%.3f F1=%.3f n=%d" % (c, d["precision"], d["recall"], d["f1"], d["support"]))
        print("   sel@50=%.4f sel@80=%.4f sel@90=%.4f"
              % (r["selective_acc@50"], r["selective_acc@80"], r["selective_acc@90"]))
    suffix = "_strict" if strict else ""
    outp = report_path.replace(".json", "_metrics%s.json" % suffix)
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("-> %s" % outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
