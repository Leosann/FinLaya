# -*- coding: utf-8 -*-
"""
把 CFSC-ABSA / FinFE / FinNSP 转成 Laya 可训练格式（JSONL，每行一条「决策样本」）。

Laya 训练数据 schema（见官方 docs/finetune.md）：
    {"state": <文本或 JSON>, "questions": {qid: {...}}, "expected": {qid: 答案}}

题型（primitive）：
    choice  从 criteria 里单选；expected 填选项字符串（= criteria 的 key）
    noul    是/否判断；expected 填 true / false
    score   有序量表打分；expected 填档位（字符串）

用法：
    python to_laya_format.py
输出到 laya/ 目录。
"""

import json
import os
import random
import re
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "laya")
TASK_PREFIX = "laya_"

# ---------------------------------------------------------------- 公共问题定义
# 所有样本共用同一份 questions 定义（Laya 要求同一 qid 在各行 schema 一致）

ABS_QID = "sentiment_absa"
ABS_QUESTION = {
    "type": "choice",
    "instructions": "在给定方面词的语境下，这句话对该方面词表达的情感倾向是什么？",
    "criteria": {
        "positive": "正面、看好、上涨、利好、增长等积极表述",
        "neutral": "客观陈述、无明显倾向、或正负信息兼有",
        "negative": "负面、看空、下跌、利空、下滑、亏损等消极表述",
    },
}

FE_QID = "sentiment_social"
FE_QUESTION = {
    "type": "choice",
    "instructions": "这段股吧/雪球社交文本对股市或个股的情绪倾向是什么？",
    "criteria": {
        "negative": "看空、恐慌、抱怨、亏损、割肉、骂声等消极情绪",
        "neutral": "中性陈述、疑问、无明确多空倾向",
        "positive": "看多、乐观、期待上涨等积极情绪",
    },
}

NSP_QID = "neg_has"
NSP_QUESTION = {
    "type": "noul",
    "instructions": "文中是否包含针对给定候选实体的负面消息（如诈骗、暴雷、跑路、处罚、亏损、违约等）？",
}

# 主体判定题：用"固定的高频实体词表"做 choice，而不是每题动态生成候选。
# 原因：Laya 要求同一 qid 的 questions 定义在所有行一致（train.py 存 questions.json 时
# 遇到跨行差异会 warn 并保留第一条），每题一套 criteria 会让推理端无法复原该题的选项空间。
NSP_SUBJECT_QID = "neg_subject"
NSP_SUBJECT_INSTRUCTIONS = "负面消息的主体是下列哪个实体？"
NSP_VOCAB_SIZE = 20   # Laya 的 choice 建议 <=20 个选项，超过精度明显下滑
NSP_OTHER = "以上都不是"


def dump_jsonl(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


# ---------------------------------------------------------------- 1. CFSC-ABSA
def conv_cfsc():
    """每个(句子, 方面词)一条样本；state 用带方面词位置的上下文渲染。"""
    stats = {}
    for split, src, dst in [("train", "CFSC-ABSA/train.txt", "cfsc_absa_train.jsonl"),
                            ("test", "CFSC-ABSA/test.txt", "cfsc_absa_test.jsonl")]:
        data = json.load(open(os.path.join(BASE, src), encoding="utf-8"))
        rows = []
        for r in data:
            sent = r["sentence"].strip()
            term = r["term"]
            start, end = r.get("from"), r.get("to")
            # 位置字段偶有偏差，以实际子串定位兜底
            if start is None or sent[start:end] != term:
                idx = sent.find(term)
                start = idx if idx >= 0 else None
            if start is not None:
                end = start + len(term)
                marked = sent[:start] + "【" + term + "】" + sent[end:]
                state = "方面词：%s\n语句：%s" % (term, marked)
            else:
                state = "方面词：%s\n语句：%s" % (term, sent)  # 少数字符被清洗掉
            rows.append({
                "state": state,
                "questions": {ABS_QID: ABS_QUESTION},
                "expected": {ABS_QID: r["polarity"]},
                "meta": {
                    "source": "CFSC-ABSA", "split": split,
                    "category": r.get("Level 1 aspect"),
                    "term": term, "from": r.get("from"), "to": r.get("to"),
                },
            })
        stats[dst] = {"rows": dump_jsonl(rows, os.path.join(OUT, dst)),
                      "labels": Counter(r["expected"][ABS_QID] for r in rows)}
    return stats


# ---------------------------------------------------------------- 2. FinFE
FE_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def conv_finfe():
    stats = {}
    for src, dst in [("FinFE/train_list.json", "finfe_train.jsonl"),
                     ("FinFE/eval_list.json", "finfe_eval.jsonl")]:
        data = json.load(open(os.path.join(BASE, src), encoding="utf-8"))
        rows = []
        for text, label in data:
            rows.append({
                "state": text,
                "questions": {FE_QID: FE_QUESTION},
                "expected": {FE_QID: FE_MAP[int(label)]},
                "meta": {"source": "FinFE"},
            })
        stats[dst] = {"rows": dump_jsonl(rows, os.path.join(OUT, dst)),
                      "labels": Counter(r["expected"][FE_QID] for r in rows)}

    # 无标签测试集：同一 questions 定义，不带 expected，可直接喂给 predict / laya-evals
    prob = json.load(open(os.path.join(BASE, "FinFE/problem_list.json"), encoding="utf-8"))
    rows = [{"state": p["input"], "questions": {FE_QID: FE_QUESTION},
             "meta": {"source": "FinFE", "id": p["id"]}} for p in prob]
    stats["finfe_test_inputs.jsonl"] = {"rows": dump_jsonl(
        rows, os.path.join(OUT, "finfe_test_inputs.jsonl"))}
    return stats


# ---------------------------------------------------------------- 3. FinNSP
def _split_entities(s):
    return [x.strip() for x in str(s or "").replace("，", ";").replace(",", ";").split(";")
            if x.strip()]


def conv_finnsp():
    """两个题型：
       neg_has       noul，文中有没有针对候选实体的负面消息
       neg_subject   choice，负面主体是哪个实体（固定 20 个高频实体 + 「以上都不是」兜底）
    """
    train = json.load(open(os.path.join(BASE, "FinNSP/train_list.json"), encoding="utf-8"))

    # 只用训练集统计实体词表，避免验证集信息泄漏；顺序固定，保证 schema 跨行一致
    def norm_entity(e):
        """去掉括注和空白做归一，用于合并「资易贷」与其全称等近似重复项。"""
        s = re.sub(r"[（(][^）)]*[）)]", "", e)
        return re.sub(r"\s+", "", s).lower()

    counter = Counter()
    for rec in train:
        if str(rec[4]).strip().lower() in ("1", "true", "yes", "是", "negative"):
            for e in _split_entities(rec[5]):
                counter[norm_entity(e) or e] += 1
    # 词表里放最短的写法（最通用、也最不容易被截断）
    display = {}
    for rec in train:
        for e in _split_entities(rec[5]):
            k = norm_entity(e) or e
            if k not in display or len(e) < len(display[k]):
                display[k] = e
    vocab = [display[k] for k, _ in counter.most_common() if k in display][:NSP_VOCAB_SIZE]
    subject_question = {
        "type": "choice",
        "instructions": NSP_SUBJECT_INSTRUCTIONS,
        "criteria": {e: "金融实体（高频出现于负面消息）" for e in vocab + [NSP_OTHER]},
    }

    stats = {"nsp_vocab": vocab}
    max_cand = 0
    for src, dst, split in [("FinNSP/train_list.json", "finnsp_train.jsonl", "train"),
                            ("FinNSP/eval_list.json", "finnsp_eval.jsonl", "eval")]:
        data = json.load(open(os.path.join(BASE, src), encoding="utf-8"))
        rows, unknown, subj_rows, other_hit = [], 0, 0, 0
        for rec in data:
            _id, _meta, text, cands_raw, flag, subject = rec[:6]
            cands = _split_entities(cands_raw)
            max_cand = max(max_cand, len(cands))

            raw_flag = str(flag).strip().lower()
            if raw_flag in ("1", "true", "yes", "是", "negative"):
                has = True
            elif raw_flag in ("0", "false", "no", "否", ""):
                has = False
            else:
                unknown += 1
                continue

            state = "候选实体：%s\n文本：%s" % ("；".join(cands), str(text).strip())
            questions = {NSP_QID: dict(NSP_QUESTION)}
            expected = {NSP_QID: has}

            if has:
                subjects = _split_entities(subject)
                subj_keys = {norm_entity(e) or e for e in subjects}
                hit = [e for e in vocab if (norm_entity(e) or e) in subj_keys]
                if hit:
                    answer = hit[0]
                else:
                    answer = NSP_OTHER
                    other_hit += 1
                questions[NSP_SUBJECT_QID] = subject_question
                expected[NSP_SUBJECT_QID] = answer
                subj_rows += 1

            rows.append({
                "state": state,
                "questions": questions,
                "expected": expected,
                "meta": {"source": "FinNSP", "split": split, "id": _id,
                         "candidates": cands, "subjects": _split_entities(subject) if has else [],
                         "n_cands": len(cands)},
            })
        stats[dst] = {"rows": dump_jsonl(rows, os.path.join(OUT, dst)),
                      "has_neg": dict(Counter(str(r["expected"][NSP_QID]) for r in rows)),
                      "unknown_flag": unknown,
                      "has_subject_question": subj_rows,
                      "subject_out_of_vocab": other_hit}

    prob = json.load(open(os.path.join(BASE, "FinNSP/problem_list.json"), encoding="utf-8"))
    rows = []
    for p in prob:
        inp = p["input"]
        text, cands = inp[0], _split_entities(inp[-1])
        rows.append({
            "state": "候选实体：%s\n文本：%s" % ("；".join(cands), str(text).strip()),
            "questions": {NSP_QID: dict(NSP_QUESTION), NSP_SUBJECT_QID: subject_question},
            "meta": {"source": "FinNSP", "id": p["id"], "candidates": cands},
        })
    stats["finnsp_test_inputs.jsonl"] = {"rows": dump_jsonl(
        rows, os.path.join(OUT, "finnsp_test_inputs.jsonl"))}

    # 三份合一（train 内部整体打散，避免同类样本连续出现）
    merged = []
    for name in ("cfsc_absa_train.jsonl", "finfe_train.jsonl", "finnsp_train.jsonl"):
        with open(os.path.join(OUT, name), encoding="utf-8") as f:
            merged.extend(json.loads(l) for l in f if l.strip())
    random.Random(42).shuffle(merged)
    stats["all_train.jsonl"] = {"rows": dump_jsonl(merged, os.path.join(OUT, "all_train.jsonl")),
                                "max_candidates_in_state": max_cand}
    return stats


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    report.update(conv_cfsc())
    report.update(conv_finfe())
    report.update(conv_finnsp())
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
