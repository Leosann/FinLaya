# -*- coding: utf-8 -*-
"""common/laya/laya_metrics.py 的单元测试。

分三层：
  1. 指标函数本身的数值正确性（边界、退化输入、numpy 版本兼容）；
  2. 排序稳定性回归（置信度并列时结果必须可复现）；
  3. 黄金回归：用仓库里已提交的 laya-evals 报告重算指标，必须复现
     reports/eval/ft_*_metrics.json 里已发布的数值（README 的结论依赖这些数字）。
"""
import json
import os
import shutil

import numpy as np
import pytest

REPORT_DIR = os.path.join("FinLaya-mmBERT", "reports", "eval")
GOLDEN = [
    ("ft_finfe_eval.json", "ft_finfe_eval_metrics.json"),
    ("ft_finnsp_eval.json", "ft_finnsp_eval_metrics.json"),
    ("ft_cfsc_absa_test.json", "ft_cfsc_absa_test_metrics.json"),
]


# ---------------------------------------------------------------- 混淆矩阵 / P-R-F1
def test_confusion_and_prf_perfect_prediction(laya_metrics):
    y = np.array([0, 1, 2, 2, 1])
    cm, per, mf1, wf1, mf1_present, n_present = laya_metrics.confusion_and_prf(
        y, y.copy(), ["neg", "neu", "pos"])
    assert np.array_equal(np.array(cm), np.diag([1, 2, 2]))
    assert all(v["precision"] == 1.0 and v["recall"] == 1.0 and v["f1"] == 1.0
               for v in per.values())
    assert mf1 == 1.0 and wf1 == 1.0 and mf1_present == 1.0 and n_present == 3


def test_confusion_and_prf_zero_support_class_is_excluded_from_present_macro(laya_metrics):
    """测试集里没出现过的类，F1 计 0 会系统性压低 macro-F1 —— 两种口径都要能算出来。"""
    y = np.array([0, 0, 2])
    pred = np.array([0, 2, 2])
    _, per, mf1, _, mf1_present, n_present = laya_metrics.confusion_and_prf(
        y, pred, ["a", "b", "c"])
    assert per["b"]["support"] == 0 and per["b"]["f1"] == 0.0
    assert n_present == 2
    assert mf1 < mf1_present
    assert mf1_present == pytest.approx((per["a"]["f1"] + per["c"]["f1"]) / 2)


def test_confusion_and_prf_no_division_by_zero_when_class_never_predicted(laya_metrics):
    y = np.array([0, 0, 0])
    pred = np.array([0, 0, 0])
    _, per, *_ = laya_metrics.confusion_and_prf(y, pred, ["a", "b"])
    assert per["b"]["precision"] == 0.0 and per["b"]["recall"] == 0.0
    assert per["b"]["f1"] == 0.0


# ---------------------------------------------------------------- ECE
def test_ece_is_zero_for_perfect_calibration(laya_metrics):
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])
    correct = np.array([True, True])
    assert laya_metrics.ece_score(probs, correct) == pytest.approx(0.0, abs=1e-12)


def test_ece_equals_confidence_when_everything_is_wrong(laya_metrics):
    probs = np.array([[0.9, 0.1]] * 4)
    correct = np.array([False] * 4)          # top-1 置信度 0.9，全错
    assert laya_metrics.ece_score(probs, correct) == pytest.approx(0.9, abs=1e-12)


def test_ece_handles_confidence_exactly_one(laya_metrics):
    """bins 右边界 = 1.0，最后一个 bin 必须把 conf == 1.0 收进去，否则会漏样本。"""
    probs = np.array([[1.0, 0.0]])
    assert laya_metrics.ece_score(probs, np.array([False])) == pytest.approx(1.0)
    assert laya_metrics.ece_score(probs, np.array([True])) == pytest.approx(0.0)


# ---------------------------------------------------------------- Brier / NLL
def test_brier_multiclass_one_hot_bounds(laya_metrics):
    probs = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert laya_metrics.brier_mc(probs, np.array([0, 1])) == pytest.approx(2.0)


def test_nll_is_clipped_instead_of_infinite(laya_metrics):
    """真实类别概率为 0 时不能返回 inf（会把整组均值污染成 inf/NaN）。"""
    probs = np.array([[0.0, 1.0]])
    v = laya_metrics.nll_s(probs, np.array([0]))
    assert np.isfinite(v)
    assert v == pytest.approx(-np.log(1e-12))


# ---------------------------------------------------------------- AURC / 选择性准确率
def test_aurc_is_zero_when_all_correct(laya_metrics):
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    y = np.array([0, 0, 0])
    assert laya_metrics.aurc_s(probs, y) == pytest.approx(0.0)


def test_aurc_penalises_confidently_wrong_more_than_confidently_right(laya_metrics):
    y = np.array([0, 0, 1, 1])
    good = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])
    bad = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    assert laya_metrics.aurc_s(good, y) < laya_metrics.aurc_s(bad, y)


def test_selective_at_keeps_at_least_one_sample(laya_metrics):
    """n=1 且 cov=0.5 时 round(0.5)==0，必须靠 max(1, ...) 兜住，不能返回空集。"""
    probs = np.array([[0.6, 0.4]])
    acc, th = laya_metrics.selective_at(probs, np.array([0]), 0.5)
    assert acc == 1.0 and th == pytest.approx(0.6)


def test_selective_at_coverage_rounding(laya_metrics):
    probs = np.tile(np.array([[0.9, 0.1]]), (10, 1))
    y = np.array([0] * 6 + [1] * 4)
    acc, th = laya_metrics.selective_at(probs, y, 0.5)   # k=5
    assert acc == pytest.approx(1.0)
    assert th == pytest.approx(0.9)


# ---------------------------------------------------------------- 排序稳定性（回归）
def _stable_reference(probs, y, cov):
    """不依赖 numpy 排序实现的参照：按 (置信度降序, 原始下标升序) 取前 k 条。

    这正是「稳定排序」的语义，而 np.argsort 的默认 quicksort 不保证这一点。
    """
    conf = probs.max(axis=1)
    ok = (probs.argmax(axis=1) == y).astype(float)
    k = max(1, int(round(cov * len(conf))))
    order = sorted(range(len(conf)), key=lambda i: (-conf[i], i))[:k]
    return float(np.mean([ok[i] for i in order])), float(min(conf[i] for i in order))


def test_selective_at_breaks_confidence_ties_by_original_order(laya_metrics, tied_case):
    """置信度并列时，取哪 k 条完全由排序稳定性决定。

    laya 报告里的概率是 4 位小数（FinFE 2020 条里有 1051 条置信度重复），
    默认 quicksort 不稳定，会让 selective_acc 随 numpy 版本/平台漂移。
    这组构造数据上 quicksort 给 0.5333、稳定排序给 0.4667，能区分两种实现。
    """
    probs, y = tied_case
    acc, th = laya_metrics.selective_at(probs, y, 0.5)
    assert (acc, th) == _stable_reference(probs, y, 0.5)
    assert acc == pytest.approx(14 / 30)     # quicksort 会给 16/30
    assert th == pytest.approx(0.8)          # 前 30 条里最低置信度


def test_selective_at_is_deterministic_across_calls(laya_metrics):
    rng = np.random.default_rng(7)
    probs = np.round(rng.random((200, 3)), 2)      # 取整制造大量并列
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = rng.integers(0, 3, 200)
    runs = {laya_metrics.selective_at(probs, y, 0.5) for _ in range(5)}
    assert len(runs) == 1
    assert runs.pop() == _stable_reference(probs, y, 0.5)


def test_aurc_is_deterministic_across_calls(laya_metrics):
    rng = np.random.default_rng(11)
    probs = np.round(rng.random((150, 2)), 2)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = rng.integers(0, 2, 150)
    values = {laya_metrics.aurc_s(probs, y) for _ in range(5)}
    assert len(values) == 1


# ---------------------------------------------------------------- 命令行输入处理
def test_main_without_arguments_prints_usage(laya_metrics, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["laya_metrics.py"])
    assert laya_metrics.main() == 2
    assert "用法" in capsys.readouterr().out


def test_main_rejects_report_without_cases(laya_metrics, monkeypatch, tmp_path, capsys):
    """最常见的误用：把 *_metrics.json 当成原始报告传进来。"""
    report = tmp_path / "ft_finfe_eval_metrics.json"
    report.write_text(json.dumps({"sentiment_social": {"accuracy": 0.78}}), encoding="utf-8")
    ds = tmp_path / "d.jsonl"
    ds.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["laya_metrics.py", str(report), str(ds)])
    assert laya_metrics.main() == 1
    assert "cases" in capsys.readouterr().out


def test_main_rejects_missing_dataset(laya_metrics, monkeypatch, tmp_path, capsys):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"cases": []}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["laya_metrics.py", str(report), str(tmp_path / "nope.jsonl")])
    assert laya_metrics.main() == 1
    assert "找不到数据集" in capsys.readouterr().out


def test_main_reports_case_hash_misalignment(laya_metrics, monkeypatch, tmp_path, capsys):
    report = tmp_path / "r.json"
    report.write_text(json.dumps({"cases": [{"qid": "q"}]}), encoding="utf-8")
    ds = tmp_path / "d.jsonl"
    ds.write_text(json.dumps({"state": "a", "expected": {"q": "x"}}, ensure_ascii=False) + "\n"
                  + json.dumps({"state": "b", "expected": {"q": "x"}}, ensure_ascii=False) + "\n",
                  encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["laya_metrics.py", str(report), str(ds)])
    assert laya_metrics.main() == 1
    assert "对齐失败" in capsys.readouterr().out


# ---------------------------------------------------------------- 黄金回归
def _write_dataset_from_report(report, path):
    """按 cases 顺序还原一个最小数据集：每行一题，state 用唯一串占位。

    全量口径下 state 哈希只用来分组，不参与指标计算，所以占位串不影响结果；
    这样就不需要仓库里未提交的 datasets/laya/*.jsonl。
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(report["cases"]):
            f.write(json.dumps({
                "state": "%s#%d" % (c["qid"], i),
                "questions": {c["qid"]: {"type": c["answer"]["type"], "instructions": "x"}},
                "expected": {c["qid"]: c["expected"]},
            }, ensure_ascii=False) + "\n")


@pytest.mark.parametrize("report_name,metrics_name", GOLDEN)
def test_reproduces_published_metrics(laya_metrics, monkeypatch, tmp_path,
                                      report_name, metrics_name, capsys):
    """用已提交的原始报告重算，必须复现已发布的指标文件。"""
    src_report = os.path.join(REPORT_DIR, report_name)
    src_metrics = os.path.join(REPORT_DIR, metrics_name)
    if not (os.path.exists(src_report) and os.path.exists(src_metrics)):
        pytest.skip("仓库里缺少 %s / %s" % (report_name, metrics_name))

    report_path = tmp_path / report_name
    shutil.copy(src_report, report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ds_path = tmp_path / (report_name.replace(".json", ".jsonl"))
    _write_dataset_from_report(report, ds_path)

    monkeypatch.setattr("sys.argv", ["laya_metrics.py", str(report_path), str(ds_path)])
    assert laya_metrics.main() == 0
    capsys.readouterr()

    got = json.loads((tmp_path / metrics_name).read_text(encoding="utf-8"))
    want = json.loads(open(src_metrics, encoding="utf-8").read())

    assert sorted(got) == sorted(want), "qid 集合不一致"
    for qid in want:
        for key in ("accuracy", "macro_f1", "weighted_f1", "macro_f1_present",
                    "n_classes_present", "n_classes", "n", "ece", "brier", "nll",
                    "mean_confidence", "selective_acc@50", "selective_acc@80",
                    "selective_acc@90", "selective_threshold@50",
                    "selective_threshold@80", "selective_threshold@90"):
            assert got[qid][key] == pytest.approx(want[qid][key], abs=1e-12), \
                "%s / %s 与已发布数值不一致" % (qid, key)
        # AURC 依赖并列样本的排序，历史数值无法逐位复现（见 README 的说明与测试注释），
        # 只要求落在 1e-3 内。
        assert got[qid]["aurc"] == pytest.approx(want[qid]["aurc"], abs=1e-3)


def test_report_hashes_expands_one_entry_per_question(laya_metrics, tmp_path):
    """同一行两题（FinNSP）要展开成两个哈希，否则和 cases 无法逐位对齐。"""
    ds = tmp_path / "d.jsonl"
    ds.write_text(
        json.dumps({"state": "同一行", "expected": {"neg_has": True, "neg_subject": "A"}},
                   ensure_ascii=False) + "\n"
        + json.dumps({"state": {"text": "第二行"}, "expected": {"neg_has": False}},
                     ensure_ascii=False) + "\n",
        encoding="utf-8")
    hs = laya_metrics.report_hashes(str(ds))
    assert len(hs) == 3
    assert hs[0] == hs[1] != hs[2]
    assert hs[0] == laya_metrics.sha("同一行")
    assert hs[2] == laya_metrics.sha("第二行")   # state 是 {"text": ...} 时取 text


def test_sha_is_stable_and_handles_none(laya_metrics):
    assert laya_metrics.sha("文本") == laya_metrics.sha("文本")
    assert laya_metrics.sha(None) == laya_metrics.sha("")
    assert len(laya_metrics.sha("x")) == 64
