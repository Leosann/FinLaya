# -*- coding: utf-8 -*-
"""剔除与训练集重叠的样本后，重算各测试集的准确率（严格口径）。

读 laya-evals 报告里的 cases（每例含 expected / answer / correct），
再和训练集 state 哈希比对，只保留「训练时没见过」的样本。
"""
import hashlib
import json
import os
import sys

DATA = os.environ.get("LAYA_DATA", "/data/laya-fin/laya")
REPORTS = os.environ.get("LAYA_REPORTS", "/data/laya-fin/reports")


def sha(text):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_states(path):
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            s = d.get("state") or {}
            text = s.get("text") if isinstance(s, dict) else s
            out.add(sha(text))
    return out


def main():
    train = set()
    for f in ("cfsc_absa_train", "finfe_train", "finnsp_train"):
        train |= load_states(os.path.join(DATA, f + ".jsonl"))
    print("train unique state hashes: %d\n" % len(train))

    # 报告 cases 的顺序与数据集行序一致，按顺序取 state
    for name in ("finfe_eval", "finnsp_eval", "cfsc_absa_test"):
        ds_path = os.path.join(DATA, name + ".jsonl")
        rp_path = os.path.join(REPORTS, "ft_%s.json" % name)
        if not os.path.exists(rp_path):
            print("%s: 报告缺失，跳过" % name)
            continue
        rows, hs = [], []
        with open(ds_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                s = d.get("state") or {}
                hs.append(sha(s.get("text") if isinstance(s, dict) else s))
        rep = json.load(open(rp_path, "r", encoding="utf-8"))
        cases = rep["cases"]
        if len(cases) != len(hs):
            print("%s: cases=%d 与 行=%d 不匹配，跳过" % (name, len(cases), len(hs)))
            continue

        # 按 qid 分组统计（finnsp 有主体题 + noul 两题）
        stat = {}
        for h, c in zip(hs, cases):
            qid = c.get("qid")
            seen = h in train
            d = stat.setdefault(qid, {"n": 0, "ok": 0, "n_clean": 0, "ok_clean": 0})
            d["n"] += 1
            d["ok"] += 1 if c.get("correct") else 0
            if not seen:
                d["n_clean"] += 1
                d["ok_clean"] += 1 if c.get("correct") else 0
        print("---- %s (含泄漏 / 严格口径) ----" % name)
        for qid, d in sorted(stat.items()):
            a = 100.0 * d["ok"] / d["n"] if d["n"] else 0
            b = 100.0 * d["ok_clean"] / d["n_clean"] if d["n_clean"] else 0
            print("  %-18s 全量 %6.2f%% (n=%-5d)   去泄漏 %6.2f%% (n=%-5d, 剔除 %d)"
                  % (qid, a, d["n"], b, d["n_clean"], d["n"] - d["n_clean"]))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
