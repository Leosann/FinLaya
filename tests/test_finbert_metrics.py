# -*- coding: utf-8 -*-
"""FinBert/scripts/finbert_baseline.py 里可离线测试的部分。

只覆盖不碰 torch 的纯逻辑：标签表构建、行读取、softmax，以及那套与
common/laya/laya_metrics.py 重复实现的指标函数。

这些指标函数在两处各写了一遍，是最容易悄悄跑偏的地方，
所以这里除了单点数值，还做「两套实现必须一致」的交叉校验。
"""
import json
import os

import numpy as np
import pytest


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 数据读取
def test_sha_matches_hashlib(finbert):
    import hashlib
    assert finbert.sha("金融文本") == hashlib.sha256("金融文本".encode("utf-8")).hexdigest()
    assert finbert.sha(None) == finbert.sha("")


def test_load_rows_skips_rows_without_the_question(finbert, tmp_path):
    """FinNSP 同一行里只有部分行带主体题，缺 expected[qid] 的行必须跳过。"""
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [
        {"state": "有标签", "expected": {"neg_subject": "甲公司"}},
        {"state": "无该题", "expected": {"neg_has": True}},
        {"state": "没有 expected"},
    ])
    assert finbert.load_rows(str(p), "neg_subject") == [("有标签", "甲公司")]


def test_load_rows_normalises_bool_labels(finbert, tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [
        {"state": "a", "expected": {"neg_has": True}},
        {"state": "b", "expected": {"neg_has": False}},
    ])
    assert finbert.load_rows(str(p), "neg_has") == [("a", "true"), ("b", "false")]


def test_load_rows_reads_text_from_dict_state(finbert, tmp_path):
    p = tmp_path / "d.jsonl"
    _write_jsonl(p, [{"state": {"text": "字典形态"}, "expected": {"q": "x"}}])
    assert finbert.load_rows(str(p), "q") == [("字典形态", "x")]


def test_load_rows_handles_empty_file(finbert, tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    assert finbert.load_rows(str(p), "q") == []


def test_build_label_space_uses_train_split_only(finbert, monkeypatch, tmp_path):
    """标签空间只从训练集统计，验证集里独有的标签不能进来（否则等于泄漏）。"""
    _write_jsonl(tmp_path / "cfsc_absa_train.jsonl",
                 [{"state": "a", "expected": {"sentiment_absa": "positive"}},
                  {"state": "b", "expected": {"sentiment_absa": "negative"}}])
    _write_jsonl(tmp_path / "cfsc_absa_test.jsonl",
                 [{"state": "c", "expected": {"sentiment_absa": "neutral"}}])
    monkeypatch.setattr(finbert, "DATA", str(tmp_path))
    assert finbert.build_label_space("absa") == ["negative", "positive"]


def test_build_label_space_keeps_bool_space_even_if_train_is_single_class(finbert, monkeypatch,
                                                                         tmp_path):
    """neg_has 固定 true/false：训练集只有一类时也不能塌成一维。"""
    _write_jsonl(tmp_path / "finnsp_train.jsonl",
                 [{"state": "a", "expected": {"neg_has": True}}])
    monkeypatch.setattr(finbert, "DATA", str(tmp_path))
    assert finbert.build_label_space("nsp_has") == ["false", "true"]


# ---------------------------------------------------------------- softmax
def test_softmax_rows_sum_to_one(finbert):
    logits = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    probs = finbert.softmax_np(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0].argmax() == 2


def test_softmax_does_not_overflow_on_large_logits(finbert):
    """没有 max 减法时 exp(1e4) 会溢出成 inf/nan。"""
    probs = finbert.softmax_np(np.array([[1e4, 0.0], [-1e4, 0.0]]))
    assert np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------- 指标：与 Laya 侧一致
@pytest.fixture
def rng_probs(finbert):
    """固定种子的 (probs, y)，用于两套指标实现的交叉校验。"""
    rng = np.random.default_rng(2024)
    probs = finbert.softmax_np(rng.normal(size=(300, 4)))
    y = rng.integers(0, 4, 300)
    return probs, y


def test_ece_matches_laya_implementation(finbert, laya_metrics, rng_probs):
    probs, y = rng_probs
    correct = probs.argmax(axis=1) == y
    assert finbert.ece_score(probs, correct) == pytest.approx(
        laya_metrics.ece_score(probs, correct), abs=1e-12)


def test_brier_matches_laya_implementation(finbert, laya_metrics, rng_probs):
    probs, y = rng_probs
    assert finbert.brier_multiclass(probs, y) == pytest.approx(
        laya_metrics.brier_mc(probs, y), abs=1e-12)


def test_nll_matches_laya_implementation(finbert, laya_metrics, rng_probs):
    probs, y = rng_probs
    assert finbert.nll_score(probs, y) == pytest.approx(
        laya_metrics.nll_s(probs, y), abs=1e-12)


def test_selective_at_matches_laya_implementation(finbert, laya_metrics, rng_probs):
    probs, y = rng_probs
    for cov in (0.5, 0.8, 0.9):
        assert finbert.selective_at(probs, y, cov) == pytest.approx(
            laya_metrics.selective_at(probs, y, cov), abs=1e-12)


def test_aurc_matches_laya_implementation_when_no_ties(finbert, laya_metrics):
    """置信度互不相同的理想情况下，两套 AURC 实现必须给出同一个数。"""
    probs = np.array([[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    y = np.array([0, 1, 1, 1])
    assert finbert.aurc(probs, y) == pytest.approx(laya_metrics.aurc_s(probs, y), abs=1e-12)


def test_brier_top1_bounds(finbert):
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert finbert.brier_top1(probs, np.array([0, 1])) == pytest.approx(0.0)
    assert finbert.brier_top1(probs, np.array([1, 0])) == pytest.approx(1.0)


# ---------------------------------------------------------------- 指标：并列时的稳定性
def test_selective_at_is_stable_with_tied_confidences(finbert, tied_case):
    """FinBERT 侧这份实现与 Laya 侧必须同样按原始下标定序（并列时 quicksort 会给 16/30）。"""
    probs, y = tied_case
    acc, th = finbert.selective_at(probs, y, 0.5)
    assert acc == pytest.approx(14 / 30)
    assert th == pytest.approx(0.8)


def test_selective_at_ties_match_laya_implementation(finbert, laya_metrics, tied_case):
    probs, y = tied_case
    for cov in (0.5, 0.8, 0.9):
        assert finbert.selective_at(probs, y, cov) == pytest.approx(
            laya_metrics.selective_at(probs, y, cov), abs=1e-12)


def test_aurc_is_deterministic_with_tied_confidences(finbert):
    probs = np.round(np.random.default_rng(5).random((120, 2)), 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.random.default_rng(6).integers(0, 2, 120)
    assert len({finbert.aurc(probs, y) for _ in range(5)}) == 1


# ---------------------------------------------------------------- 任务表
def test_task_table_covers_four_decision_tasks(finbert):
    """TASKS 是训练/评测的唯一入口表，四个决策题的 qid 必须对得上数据集定义。"""
    assert set(finbert.TASKS) == {"absa", "finfe", "nsp_has", "nsp_subject"}
    assert finbert.TASKS["absa"][2] == "sentiment_absa"
    assert finbert.TASKS["finfe"][2] == "sentiment_social"
    assert finbert.TASKS["nsp_has"][2] == "neg_has"
    assert finbert.TASKS["nsp_subject"][2] == "neg_subject"
