# -*- coding: utf-8 -*-
"""用 laya 的 Python API 评测微调后的 checkpoint。

为什么不用 laya-evals CLI：它的 --model 只认注册表别名（english / multilingual / ...），
本地目录会被 router.resolve 拒掉。但 Router.register(name, source) 官方支持注册本地目录
（docstring 原话：a local directory holding model.safetensors and rl_agent_config.json --
a fine-tune exported by the training recipe, for instance），所以走 API。

用法: python run_eval.py <dataset.jsonl> <ckpt_dir> <out.json> [model-name]
"""
import json
import os
import sys
import tempfile


def main():
    dataset, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else "fin-laya"
    batch_size = int(os.environ.get("EVAL_BATCH", "16"))

    import laya
    from laya.evals import Dataset, evaluate
    from laya.evals_cli import RouterRunner

    router = laya.Router(device="cuda", preload=False)
    name = router.register(model_name, ckpt, description="CFSC-ABSA + FinFE + FinNSP fine-tune")
    print("[eval] registered %r -> %s" % (name, ckpt), flush=True)

    # 把每行的 model 字段改成注册名（顶层与 expected 都保留原样）
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    n = 0
    with open(dataset, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d["model"] = name
            tmp.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    tmp.close()
    print("[eval] %d rows" % n, flush=True)

    ds = Dataset.from_jsonl(tmp.name)
    report = evaluate(RouterRunner(router), ds, batch_size=batch_size, on_error="skip",
                      config={"dataset": os.path.basename(dataset), "model": name,
                              "checkpoint": ckpt, "device": "cuda"})
    # EvalReport 不是 dict：report.to_json() 才是可序列化的 payload（to_dict 不存在）
    payload = report.to_json()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write(report.to_markdown())

    ov = payload.get("overall", {})
    print("[eval] === %s ===" % os.path.basename(dataset), flush=True)
    for k in ("choice_accuracy", "noul_accuracy", "score_mae", "ece", "brier", "aurc",
              "mean_confidence", "selective_accuracy@50", "selective_accuracy@80",
              "latency_p50_ms", "cost_per_decision_p50_ms"):
        v = ov.get(k)
        if v is None:
            continue
        print("  %-24s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v), flush=True)
    os.unlink(tmp.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
