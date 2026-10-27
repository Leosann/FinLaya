# -*- coding: utf-8 -*-
"""datasets/validate_laya_format.py 的单元测试。

这个脚本是数据进入训练前的最后一道闸（README 用 exit 0 表示通过），
所以除了规则本身，还要覆盖两类「闸门失灵」：
  - 目录为空时静默通过（假绿灯）；
  - 单行坏数据把整个校验打断，后面的行根本没被检查。
"""
import json
import os

import pytest

CHOICE_Q = {"type": "choice", "instructions": "情感倾向？",
            "criteria": {"positive": "积极", "neutral": "中性", "negative": "消极"}}


def _row(state, qid, qdef, expected=None):
    r = {"state": state, "questions": {qid: qdef}}
    if expected is not None:
        r["expected"] = {qid: expected}
    return r


def _write(dirpath, name, rows):
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, name)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")
    return p


# ---------------------------------------------------------------- 题型定义校验
def test_check_question_choice_returns_criteria_keys_in_order(validate_mod):
    assert validate_mod.check_question("q", CHOICE_Q) == ["positive", "neutral", "negative"]


def test_check_question_choice_accepts_list_criteria(validate_mod):
    q = {"type": "choice", "instructions": "选一个", "criteria": ["a", "b"]}
    assert validate_mod.check_question("q", q) == ["a", "b"]


def test_check_question_rejects_unknown_type(validate_mod):
    with pytest.raises(AssertionError, match="未知题型"):
        validate_mod.check_question("q", {"type": "ranking", "instructions": "x"})


def test_check_question_rejects_blank_instructions(validate_mod):
    with pytest.raises(AssertionError, match="instructions"):
        validate_mod.check_question("q", dict(CHOICE_Q, instructions="   "))


def test_check_question_rejects_empty_criteria(validate_mod):
    with pytest.raises(AssertionError, match="criteria"):
        validate_mod.check_question("q", {"type": "choice", "instructions": "x", "criteria": {}})


def test_check_question_rejects_duplicate_labels(validate_mod):
    q = {"type": "choice", "instructions": "x", "criteria": ["a", "a"]}
    with pytest.raises(AssertionError, match="标签重复"):
        validate_mod.check_question("q", q)


def test_check_question_rejects_non_scalar_label(validate_mod):
    q = {"type": "choice", "instructions": "x", "criteria": [{"label": "a"}]}
    with pytest.raises(AssertionError, match="非标量"):
        validate_mod.check_question("q", q)


def test_check_question_noul_uses_bool_space(validate_mod):
    assert validate_mod.check_question("q", {"type": "noul", "instructions": "x"}) == ["false", "true"]


def test_check_question_noul_rejects_extra_criteria_keys(validate_mod):
    q = {"type": "noul", "instructions": "x", "criteria": {"true": "是", "maybe": "也许"}}
    with pytest.raises(AssertionError):
        validate_mod.check_question("q", q)


def test_check_question_score_requires_list(validate_mod):
    q = {"type": "score", "instructions": "打分", "criteria": ["低", "中", "高"]}
    assert validate_mod.check_question("q", q) == [0, 1, 2]
    with pytest.raises(AssertionError):
        validate_mod.check_question("q", {"type": "score", "instructions": "x", "criteria": {"a": 1}})


# ---------------------------------------------------------------- 答案校验
def test_check_expected_accepts_label_in_space(validate_mod):
    validate_mod.check_expected("q", CHOICE_Q, "neutral")


def test_check_expected_rejects_label_outside_space(validate_mod):
    with pytest.raises(AssertionError, match="不在 criteria 内"):
        validate_mod.check_expected("q", CHOICE_Q, "very_positive")


def test_check_expected_rejects_non_bool_for_noul(validate_mod):
    q = {"type": "noul", "instructions": "x"}
    validate_mod.check_expected("q", q, True)
    with pytest.raises(AssertionError, match="必须是 bool"):
        validate_mod.check_expected("q", q, "yes")


def test_check_expected_score_range(validate_mod):
    q = {"type": "score", "instructions": "x", "criteria": ["低", "中", "高"]}
    validate_mod.check_expected("q", q, 2)
    validate_mod.check_expected("q", q, "1")          # 数字字符串也接受
    with pytest.raises(AssertionError, match="越界"):
        validate_mod.check_expected("q", q, 3)
    with pytest.raises(ValueError):
        validate_mod.check_expected("q", q, "not-a-number")   # 由 main() 兜住并记为坏行


# ---------------------------------------------------------------- main() 闸门行为
@pytest.fixture
def run_main(validate_mod, monkeypatch, capsys):
    def _run(dirpath):
        monkeypatch.setattr(validate_mod, "OUT", str(dirpath))
        rc = validate_mod.main()
        return rc, capsys.readouterr().out
    return _run


def test_main_passes_on_valid_dataset(run_main, tmp_path):
    out = tmp_path / "laya"
    _write(out, "ok.jsonl", [
        _row("看多", "sentiment_social", CHOICE_Q, "positive"),
        _row("看空", "sentiment_social", CHOICE_Q, "negative"),
    ])
    rc, out_text = run_main(out)
    assert rc == 0
    assert "OK" in out_text and "错误 0 条" in out_text


def test_main_fails_when_directory_is_empty(run_main, tmp_path):
    """回归：以前目录为空时会输出「0 行 / 0 错误」并返回 0，等于假通过。"""
    out = tmp_path / "laya"
    out.mkdir()
    rc, text = run_main(out)
    assert rc == 1
    assert "没有 .jsonl 文件" in text


def test_main_fails_when_directory_is_missing(run_main, tmp_path):
    rc, text = run_main(tmp_path / "not-created")
    assert rc == 1
    assert "找不到目录" in text


def test_main_reports_malformed_json_line_instead_of_crashing(run_main, tmp_path):
    """回归：以前坏 JSON 行会让整个校验抛 JSONDecodeError 中断。"""
    out = tmp_path / "laya"
    _write(out, "bad.jsonl", [
        _row("正常", "sentiment_social", CHOICE_Q, "positive"),
        '{"state": "截断的行"',
        _row("正常2", "sentiment_social", CHOICE_Q, "neutral"),
    ])
    rc, text = run_main(out)
    assert rc == 1
    assert "bad.jsonl:2" in text and "JSONDecodeError" in text
    assert "错误 1 条" in text


def test_main_reports_non_numeric_score_expected(run_main, tmp_path):
    """score 题的 expected 是 "abc" 时 int() 抛 ValueError，必须记为坏行。"""
    q = {"type": "score", "instructions": "打分", "criteria": ["低", "中"]}
    out = tmp_path / "laya"
    _write(out, "s.jsonl", [
        _row("a", "score_q", q, 1),
        _row("b", "score_q", q, "abc"),
    ])
    rc, text = run_main(out)
    assert rc == 1
    assert "ValueError" in text and "错误 1 条" in text


def test_main_detects_cross_row_schema_drift(run_main, tmp_path):
    """同一 qid 的 criteria 跨行变化时训练端会静默丢定义，这里必须报错。"""
    other = {"type": "choice", "instructions": CHOICE_Q["instructions"],
             "criteria": {"positive": "积极", "negative": "消极"}}
    out = tmp_path / "laya"
    _write(out, "drift.jsonl", [
        _row("a", "sentiment_social", CHOICE_Q, "positive"),
        _row("b", "sentiment_social", other, "negative"),
    ])
    rc, text = run_main(out)
    assert rc == 1
    assert "schema 跨行不一致" in text


def test_main_rejects_empty_state_and_undefined_question(run_main, tmp_path):
    out = tmp_path / "laya"
    _write(out, "bad.jsonl", [
        _row("", "sentiment_social", CHOICE_Q, "positive"),          # 空文本
        {"state": "有文本", "questions": {"q": CHOICE_Q}, "expected": {"other": "x"}},
    ])
    rc, text = run_main(out)
    assert rc == 1
    assert "state 为空" in text
    assert "未定义的题" in text


def test_main_handles_blank_lines_and_very_long_text(run_main, tmp_path):
    """FinNSP 里有 6777 token 的长文本，转换层不应截断或报错。"""
    long_text = "利空" * 20000
    out = tmp_path / "laya"
    _write(out, "long.jsonl", [
        "",
        _row(long_text, "sentiment_social", CHOICE_Q, "negative"),
        "   ",
    ])
    rc, text = run_main(out)
    assert rc == 0
    assert "rows=1" in text
