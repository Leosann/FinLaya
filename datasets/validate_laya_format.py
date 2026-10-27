# -*- coding: utf-8 -*-
"""按 Laya 官方源码的规则校验生成的 JSONL（不依赖模型权重）。

对齐的两处官方实现：
  laya/agent.py  Agent._check_question     -> 题型/instructions/criteria/label 合法性
  laya/train.py  target_from_expected      -> expected 能否映射成合法概率分布
"""
import json
import os
import sys

QTYPES = {"choice", "score", "noul"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "laya")


def check_question(qid, qdef):
    """laya/agent.py::Agent._check_question 的核心约束。"""
    assert isinstance(qdef, dict), "%r 定义不是 dict" % qid
    t = qdef.get("type")
    assert t in QTYPES, "%r 未知题型 %r" % (qid, t)
    ins = qdef.get("instructions")
    assert isinstance(ins, str) and ins.strip(), "%r instructions 为空" % qid
    crit = qdef.get("criteria")
    if t == "choice":
        assert isinstance(crit, (dict, list)) and crit, "%r choice 需要非空 criteria" % qid
        labels = list(crit) if isinstance(crit, list) else list(crit.keys())
        for i, lab in enumerate(labels):
            assert isinstance(lab, (str, int, float, bool)), "%r 标签非标量" % qid
            assert lab is not None, "%r 标签为 null" % qid
        assert len(set(labels)) == len(labels), "%r 标签重复" % qid
        return labels
    if t == "score":
        assert isinstance(crit, list) and crit, "%r score 需要 list" % qid
        return list(range(len(crit)))
    if crit is not None:
        assert isinstance(crit, dict) and {str(k).lower() for k in crit} <= {"true", "false"}
    return ["false", "true"]


def check_expected(qid, qdef, val):
    """laya/train.py::target_from_expected 的核心约束：答案必须落在选项空间内。"""
    t = qdef["type"]
    opts = check_question(qid, qdef)
    if t == "choice":
        assert str(val) in [str(o) for o in opts], "%r 答案 %r 不在 criteria 内" % (qid, val)
    elif t == "noul":
        assert isinstance(val, bool), "%r noul 答案必须是 bool，得到 %r" % (qid, val)
    else:
        assert int(val) in range(len(opts)), "%r score 档位越界" % qid


def main():
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".jsonl"))
    total_rows = total_q = 0
    bad = []
    for fn in files:
        rows = n_q = 0
        schemas = {}
        with open(os.path.join(OUT, fn), encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    r = json.loads(line)
                    assert "state" in r and r["state"] not in (None, ""), "state 为空"
                    qs = r.get("questions")
                    assert isinstance(qs, dict) and qs, "questions 为空"
                    for qid, qdef in qs.items():
                        opts = check_question(qid, qdef)
                        # 同一 qid 的 schema 在各行必须一致
                        sig = (qdef["type"], tuple(map(str, opts)))
                        if qid in schemas:
                            assert schemas[qid] == sig, "%r schema 跨行不一致" % qid
                        else:
                            schemas[qid] = sig
                        n_q += 1
                    for qid, val in (r.get("expected") or {}).items():
                        assert qid in qs, "expected 里有未定义的题 %r" % qid
                        check_expected(qid, qs[qid], val)
                except AssertionError as e:
                    bad.append("%s:%d  %s" % (fn, ln, e))
        total_rows += rows
        total_q += n_q
        status = "OK " if not any(b.startswith(fn + ":") for b in bad) else "FAIL"
        print("%s %-28s rows=%-6d questions=%-6d qids=%s"
              % (status, fn, rows, n_q, ",".join(sorted(schemas))))

    print("\n合计 %d 行样本 / %d 个问题；错误 %d 条" % (total_rows, total_q, len(bad)))
    for b in bad[:20]:
        print("  !", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
