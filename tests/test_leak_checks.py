# -*- coding: utf-8 -*-
"""common/laya/check_leak.py 与 strict_acc.py 的单元测试。

这两个脚本决定了「严格口径」的分母：FinFE 有 54.3% 的测试样本与训练集文本重复，
README 的核心结论就建立在这套去重上，所以哈希口径必须被固定住。
"""
import hashlib
import json
import os

import pytest


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write((r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)) + "\n")


def _sha(t):
    return hashlib.sha256((t or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- states()
def test_states_reads_plain_string_state(check_leak, tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"state": "看多"}, {"state": "看空"}])
    assert check_leak.states(str(p)) == [_sha("看多"), _sha("看空")]


def test_states_reads_dict_state(check_leak, tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"state": {"text": "字典文本"}}])
    assert check_leak.states(str(p)) == [_sha("字典文本")]


def test_states_skips_blank_lines(check_leak, tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, ["", '{"state": "a"}', "   ", '{"state": "b"}'])
    assert check_leak.states(str(p)) == [_sha("a"), _sha("b")]


def test_states_on_empty_file(check_leak, tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert check_leak.states(str(p)) == []


def test_states_treats_missing_state_as_empty_string(check_leak, tmp_path):
    """state 缺失时按空串处理（而不是崩），这类行在转换阶段就该被拦下。"""
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"questions": {}}])
    assert check_leak.states(str(p)) == [_sha("")]


# ---------------------------------------------------------------- main() 重叠统计
@pytest.fixture
def leak_env(check_leak, monkeypatch, tmp_path):
    data = tmp_path / "laya"
    _write_jsonl(data / "cfsc_absa_train.jsonl",
                 [{"state": "共同文本"}, {"state": "仅训练集A"}])
    _write_jsonl(data / "finfe_train.jsonl", [{"state": "共同文本"}])
    _write_jsonl(data / "finnsp_train.jsonl", [{"state": {"text": "仅训练集B"}}])
    # 测试集：2 条与训练集重复，1 条全新
    _write_jsonl(data / "finfe_eval.jsonl",
                 [{"state": "共同文本"}, {"state": "仅训练集A"}, {"state": "全新样本"}])
    _write_jsonl(data / "finnsp_eval.jsonl", [{"state": "全新样本"}])
    _write_jsonl(data / "cfsc_absa_test.jsonl", [{"state": "共同文本"}])
    monkeypatch.setattr(check_leak, "DATA", str(data))
    return data


def test_main_reports_overlap_per_test_set(leak_env, check_leak, capsys):
    assert check_leak.main() == 0
    out = capsys.readouterr().out
    assert "train unique states: total=3" in out
    assert "finfe_eval       rows=3      " in out
    assert "与训练集重叠=2      (66.7%)" in out
    assert "finnsp_eval      rows=1      " in out
    assert "与训练集重叠=0      (0.0%)" in out
    assert "cfsc_absa_test   rows=1      " in out


def test_main_counts_overlap_per_source_subset(leak_env, check_leak, capsys):
    check_leak.main()
    out = capsys.readouterr().out
    finfe_line = [l for l in out.splitlines() if l.startswith("finfe_eval")][0]
    # 两条重复样本分别来自 cfsc 训练集（共同文本 + 仅训练集A）与 finfe 训练集（共同文本）
    assert finfe_line.endswith("cfsc=2 finfe=1 finnsp=0")


# ---------------------------------------------------------------- strict_acc
def test_load_states_matches_check_leak_semantics(strict_acc, check_leak, tmp_path):
    """两者对 state 的取文本规则必须一致（dict 取 text、缺失按空串），只是去重与否不同。"""
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"state": "a"}, {"state": {"text": "b"}}, {"state": "a"}])
    assert set(strict_acc.load_states(str(p))) == set(check_leak.states(str(p)))
    assert len(strict_acc.load_states(str(p))) == 2      # load_states 返回集合
    assert len(check_leak.states(str(p))) == 3           # states 保留逐行顺序


def test_strict_acc_sha(strict_acc):
    assert strict_acc.sha("文本") == _sha("文本")
    assert strict_acc.sha(None) == _sha("")
