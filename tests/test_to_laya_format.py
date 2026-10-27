# -*- coding: utf-8 -*-
"""datasets/to_laya_format.py 的单元测试。

这是「原始金融语料 → 情感决策样本」的转换层，决定模型看到什么文本、什么标签：
  - CFSC-ABSA：把方面词用【】标回句子里（偏移字段不可靠时的兜底）
  - FinFE：0/1/2 → negative/neutral/positive
  - FinNSP：flag 归一化、实体切分、主体词表（<=20 项）与「以上都不是」兜底

所有测试都在临时目录里构造数据，不依赖仓库里未提交的 datasets/CFSC-ABSA 等原始语料。
"""
import json
import os

import pytest


# ---------------------------------------------------------------- 实体切分
@pytest.mark.parametrize("raw,expected", [
    ("甲公司，乙公司", ["甲公司", "乙公司"]),          # 全角逗号
    ("甲公司,乙公司", ["甲公司", "乙公司"]),           # 半角逗号
    ("甲公司;乙公司", ["甲公司", "乙公司"]),           # 半角分号
    (" 甲 , 乙 ", ["甲", "乙"]),                      # 两侧空白
    ("", []),
    (None, []),
    ("   ", []),
    ("单一实体", ["单一实体"]),
])
def test_split_entities(convert_mod, raw, expected):
    assert convert_mod._split_entities(raw) == expected


def test_split_entities_does_not_split_fullwidth_semicolon(convert_mod):
    """已知限制（本测试固定当前行为，改动需是有意为之）。

    转换层自己用「；」拼接候选实体写进 state，但 _split_entities 只把「，」和「,」
    归一成「;」，所以「；」分隔的串切不开。若上游 FinNSP 的候选/主体字段用「；」，
    多实体就会被当成一个实体，导致主体题大量落到「以上都不是」。
    原始语料未随仓库提交，无法在此验证真实分隔符，故不改实现、只锁定行为。
    """
    assert convert_mod._split_entities("甲公司；乙公司") == ["甲公司；乙公司"]


def test_split_entities_tolerates_non_string_input(convert_mod):
    """字段偶尔是 list 时不应崩（str() 之后退化成不可用的切分）。"""
    assert convert_mod._split_entities(["甲", "乙"]) == ["['甲'", "'乙']"]


def test_split_entities_scales_to_many_items(convert_mod):
    raw = "，".join("实体%d" % i for i in range(1000))
    assert len(convert_mod._split_entities(raw)) == 1000


# ---------------------------------------------------------------- 合成数据装置
@pytest.fixture
def converted(isolated_convert):
    """造一份最小的三任务原始数据并跑完整转换，返回 (模块, 统计, 输出目录)。"""
    m = isolated_convert
    base = m.BASE

    os.makedirs(os.path.join(base, "CFSC-ABSA"), exist_ok=True)
    os.makedirs(os.path.join(base, "FinFE"), exist_ok=True)
    os.makedirs(os.path.join(base, "FinNSP"), exist_ok=True)

    cfsc_train = [
        {"sentence": "长谷投资不错", "term": "长谷投资", "from": 0, "to": 4,
         "polarity": "positive", "Level 1 aspect": "company"},
        # 前导空白让 from/to 失效，必须靠 find 兜底
        {"sentence": "  长谷投资不错", "term": "长谷投资", "from": 2, "to": 6,
         "polarity": "neutral"},
        # 缺少偏移字段
        {"sentence": "长谷投资不错", "term": "长谷投资", "polarity": "negative"},
        # 方面词不在句中（上游清洗导致）
        {"sentence": "完全无关的句子", "term": "长谷投资", "polarity": "neutral"},
        # 方面词在句尾（偏移算术边界）
        {"sentence": "末尾词测试", "term": "试", "polarity": "positive"},
    ]
    _dump_json(os.path.join(base, "CFSC-ABSA", "train.txt"), cfsc_train)
    _dump_json(os.path.join(base, "CFSC-ABSA", "test.txt"), cfsc_train[:1])

    _dump_json(os.path.join(base, "FinFE", "train_list.json"),
               [["看多后市", 2], ["看空", 0], ["持仓观望", 1], ["空文本", "2"]])
    _dump_json(os.path.join(base, "FinFE", "eval_list.json"), [["看多", 2], ["看空", 0]])
    _dump_json(os.path.join(base, "FinFE", "problem_list.json"),
               [{"input": "无标签输入", "id": 11}])

    nsp_train = [
        ["1", "m", "网传甲公司被查", "甲公司，乙公司", "1", "甲公司"],
        ["2", "m", "丙公司暴雷", "丙公司", "true", "丁公司"],
        ["3", "m", "正常经营公告", "甲公司", "0", ""],
        ["4", "m", "标志未知的行", "甲公司", "?", ""],
        ["5", "m", "乙公司被处罚", "乙公司", "是", "乙公司"],
    ]
    nsp_eval = [
        ["e1", "m", "甲公司跑路", "甲公司，乙公司", "1", "甲公司"],
        ["e2", "m", "无关新闻", "甲公司", "0", ""],
        # 主体「戊公司」只出现在验证集，不在训练集词表里 → 必须落到「以上都不是」
        ["e3", "m", "戊公司被查", "戊公司", "1", "戊公司"],
    ]
    _dump_json(os.path.join(base, "FinNSP", "train_list.json"), nsp_train)
    _dump_json(os.path.join(base, "FinNSP", "eval_list.json"), nsp_eval)
    _dump_json(os.path.join(base, "FinNSP", "problem_list.json"),
               [{"input": ["丙公司被查", "丙公司"], "id": 21}])

    stats = {}
    stats.update(m.conv_cfsc())
    stats.update(m.conv_finfe())
    stats.update(m.conv_finnsp())
    return m, stats, m.OUT


def _dump_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- CFSC-ABSA
def test_cfsc_marks_aspect_term_in_sentence(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "cfsc_absa_train.jsonl"))
    assert rows[0]["state"] == "方面词：长谷投资\n语句：【长谷投资】不错"
    assert rows[0]["expected"] == {"sentiment_absa": "positive"}
    assert rows[0]["meta"]["category"] == "company"


def test_cfsc_falls_back_to_find_when_offsets_are_stale(converted):
    """sentence 被 strip 后原始 from/to 失效，必须按子串重新定位。"""
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "cfsc_absa_train.jsonl"))
    assert rows[1]["state"] == "方面词：长谷投资\n语句：【长谷投资】不错"
    assert rows[2]["state"] == rows[1]["state"]      # 缺 from/to 也走同一条兜底


def test_cfsc_leaves_text_untouched_when_term_absent(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "cfsc_absa_train.jsonl"))
    assert rows[3]["state"] == "方面词：长谷投资\n语句：完全无关的句子"


def test_cfsc_handles_term_at_end_of_sentence(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "cfsc_absa_train.jsonl"))
    assert rows[4]["state"] == "方面词：试\n语句：末尾词测【试】"


def test_cfsc_label_counts_and_row_count(converted):
    _, stats, _ = converted
    assert stats["cfsc_absa_train.jsonl"]["rows"] == 5
    assert stats["cfsc_absa_train.jsonl"]["labels"] == {"positive": 2, "neutral": 2, "negative": 1}
    assert stats["cfsc_absa_test.jsonl"]["rows"] == 1


def test_cfsc_empty_sentence_and_empty_term_do_not_crash(isolated_convert, tmp_path):
    m = isolated_convert
    os.makedirs(os.path.join(m.BASE, "CFSC-ABSA"), exist_ok=True)
    _dump_json(os.path.join(m.BASE, "CFSC-ABSA", "train.txt"),
               [{"sentence": "", "term": "", "polarity": "neutral"}])
    _dump_json(os.path.join(m.BASE, "CFSC-ABSA", "test.txt"), [])
    m.conv_cfsc()
    rows = _read_jsonl(os.path.join(m.OUT, "cfsc_absa_train.jsonl"))
    assert rows[0]["state"] == "方面词：\n语句：【】"
    assert rows[0]["expected"] == {"sentiment_absa": "neutral"}


# ---------------------------------------------------------------- FinFE
def test_finfe_maps_numeric_labels(converted):
    _, stats, out = converted
    rows = _read_jsonl(os.path.join(out, "finfe_train.jsonl"))
    assert [r["expected"]["sentiment_social"] for r in rows] == \
        ["positive", "negative", "neutral", "positive"]
    assert stats["finfe_train.jsonl"]["labels"] == \
        {"positive": 2, "negative": 1, "neutral": 1}


def test_finfe_uses_shared_question_definition(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "finfe_train.jsonl"))
    assert all(r["questions"]["sentiment_social"]["type"] == "choice" for r in rows)
    assert set(rows[0]["questions"]["sentiment_social"]["criteria"]) == \
        {"negative", "neutral", "positive"}


def test_finfe_unlabeled_inputs_have_no_expected(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "finfe_test_inputs.jsonl"))
    assert len(rows) == 1
    assert "expected" not in rows[0]
    assert rows[0]["meta"]["id"] == 11


def test_finfe_rejects_label_outside_0_1_2(isolated_convert):
    """标签越界目前抛 KeyError（无上下文信息），这里锁定现状以防静默错标。"""
    m = isolated_convert
    os.makedirs(os.path.join(m.BASE, "FinFE"), exist_ok=True)
    _dump_json(os.path.join(m.BASE, "FinFE", "train_list.json"), [["越界", 3]])
    _dump_json(os.path.join(m.BASE, "FinFE", "eval_list.json"), [])
    _dump_json(os.path.join(m.BASE, "FinFE", "problem_list.json"), [])
    with pytest.raises(KeyError):
        m.conv_finfe()


# ---------------------------------------------------------------- FinNSP
def test_finnsp_normalises_flag_variants(converted):
    _, stats, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_train.jsonl"))
    flags = [r["expected"]["neg_has"] for r in rows]
    # "1" / "true" / "0" / "?"(跳过) / "是"
    assert flags == [True, True, False, True]
    assert stats["finnsp_train.jsonl"]["unknown_flag"] == 1
    assert stats["finnsp_train.jsonl"]["rows"] == 4


def test_finnsp_only_asks_subject_question_when_negative(converted):
    _, stats, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_train.jsonl"))
    for r in rows:
        has_subject_q = "neg_subject" in r["questions"]
        assert has_subject_q == r["expected"]["neg_has"]
        assert ("neg_subject" in r["expected"]) == has_subject_q
    assert stats["finnsp_train.jsonl"]["has_subject_question"] == 3


def test_finnsp_subject_answer_uses_vocab_built_from_train(isolated_convert, converted):
    """词表由训练集负例的「主体」字段统计而来，所以训练集内的主体都能命中。"""
    _, stats, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_train.jsonl"))
    answers = {r["meta"]["id"]: r["expected"].get("neg_subject") for r in rows}
    assert answers["1"] == "甲公司"
    assert answers["2"] == "丁公司"        # 主体本身进了词表
    assert answers["5"] == "乙公司"
    assert stats["finnsp_train.jsonl"]["subject_out_of_vocab"] == 0


def test_finnsp_subject_falls_back_to_other_when_out_of_vocab(converted):
    """验证集里出现训练集没见过的实体时，答案必须是「以上都不是」而不是硬选一个。"""
    _, stats, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_eval.jsonl"))
    answers = {r["meta"]["id"]: r["expected"].get("neg_subject") for r in rows}
    assert answers["e1"] == "甲公司"
    assert answers["e2"] is None           # 无负例时不问主体题
    assert answers["e3"] == "以上都不是"
    assert stats["finnsp_eval.jsonl"]["subject_out_of_vocab"] == 1


def test_finnsp_vocab_is_capped_and_includes_fallback_option(converted):
    m, stats, out = converted
    vocab = stats["nsp_vocab"]
    assert len(vocab) <= m.NSP_VOCAB_SIZE
    assert vocab == ["甲公司", "丁公司", "乙公司"]     # 按训练集负例频次排序

    rows = _read_jsonl(os.path.join(out, "finnsp_eval.jsonl"))
    crit = rows[0]["questions"]["neg_subject"]["criteria"]
    assert list(crit) == vocab + [m.NSP_OTHER]
    assert len(crit) == len(vocab) + 1


def test_finnsp_state_carries_candidates_and_text(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_eval.jsonl"))
    assert rows[0]["state"] == "候选实体：甲公司；乙公司\n文本：甲公司跑路"
    assert rows[0]["meta"]["n_cands"] == 2


def test_finnsp_handles_very_long_text(converted, isolated_convert):
    """README 记录 FinNSP 训练集有 6777 token 的样本，转换层必须原样保留。"""
    m = isolated_convert
    long_text = "利空消息" * 5000
    _dump_json(os.path.join(m.BASE, "FinNSP", "eval_list.json"),
               [["L", "m", long_text, "甲公司", "1", "甲公司"]])
    m.conv_finnsp()
    rows = _read_jsonl(os.path.join(m.OUT, "finnsp_eval.jsonl"))
    assert rows[0]["state"].endswith(long_text)
    assert len(rows[0]["state"]) > 20000


def test_finnsp_unlabeled_inputs_always_carry_both_questions(converted):
    _, _, out = converted
    rows = _read_jsonl(os.path.join(out, "finnsp_test_inputs.jsonl"))
    assert set(rows[0]["questions"]) == {"neg_has", "neg_subject"}
    assert "expected" not in rows[0]


# ---------------------------------------------------------------- 合并训练集
def test_all_train_merges_three_sources(converted):
    _, stats, out = converted
    merged = _read_jsonl(os.path.join(out, "all_train.jsonl"))
    assert len(merged) == stats["all_train.jsonl"]["rows"] == 5 + 4 + 4
    sources = {r["meta"]["source"] for r in merged}
    assert sources == {"CFSC-ABSA", "FinFE", "FinNSP"}


def test_all_train_shuffle_is_reproducible(isolated_convert, converted):
    """固定 seed=42 必须可复现，否则同一份数据两次转换得到的训练集顺序不同。"""
    m = isolated_convert
    out = m.OUT
    first = [r["state"] for r in _read_jsonl(os.path.join(out, "all_train.jsonl"))]
    m.conv_cfsc()
    m.conv_finfe()
    m.conv_finnsp()
    second = [r["state"] for r in _read_jsonl(os.path.join(out, "all_train.jsonl"))]
    assert first == second
