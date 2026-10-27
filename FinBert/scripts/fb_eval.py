# -*- coding: utf-8 -*-
"""FinBERT 评测入口（独立脚本，避免命令行引号转义问题）。

用法:
  python fb_eval.py                 # 全部四个任务，全量 + 严格口径
  python fb_eval.py nsp_has         # 只跑一个任务（调试用）
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import finbert_baseline as F  # noqa: E402


def main():
    tasks = sys.argv[1:] or list(F.TASKS)
    out_path = os.path.join(F.REPORTS, "finbert_all.json")

    # 训练集 state 哈希（严格口径）
    import hashlib
    tr = set()
    for f in ("cfsc_absa_train.jsonl", "finfe_train.jsonl", "finnsp_train.jsonl"):
        p = os.path.join(F.DATA, f)
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                s = d["state"]["text"] if isinstance(d["state"], dict) else d["state"]
                tr.add(hashlib.sha256(s.encode("utf-8")).hexdigest())
    print("train unique states: %d" % len(tr), flush=True)

    allres = {}
    if os.path.exists(out_path):
        allres = json.load(open(out_path, "r", encoding="utf-8"))

    for task in tasks:
        print(">>> eval %s" % task, flush=True)
        r_full = F.eval_task(task, "full")
        print("[%s/full] n=%d acc=%.4f macroF1=%.4f (present %.4f) ece=%.4f nll=%.4f "
              "brier=%.4f sel@80=%.4f lat=%.2fms qps=%.0f mem=%.0fMB size=%.0fMB"
              % (task, r_full["n"], r_full["accuracy"], r_full["macro_f1"],
                 r_full["macro_f1_present"], r_full["ece"], r_full["nll"], r_full["brier"],
                 r_full["selective_acc@80"], r_full["latency_p50_ms"],
                 r_full["throughput_qps"], r_full["peak_gpu_mem_mb"],
                 r_full["model_dir_mb"]), flush=True)
        r_clean = F.eval_task(task, "strict_no_overlap", exclude_hashes=tr)
        print("[%s/strict] n=%d acc=%.4f macroF1=%.4f (present %.4f) ece=%.4f nll=%.4f "
              "brier=%.4f sel@80=%.4f"
              % (task, r_clean["n"], r_clean["accuracy"], r_clean["macro_f1"],
                 r_clean["macro_f1_present"], r_clean["ece"], r_clean["nll"],
                 r_clean["brier"], r_clean["selective_acc@80"]), flush=True)
        allres[task] = {"full": r_full, "strict": r_clean}
        json.dump(allres, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print("[%s] dumped" % task, flush=True)
    print("FB_EVAL_DONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
