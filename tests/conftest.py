# -*- coding: utf-8 -*-
"""测试公共装置。

仓库里的脚本不是包（common/laya、datasets、FinBert/scripts 都没有 __init__.py），
所以这里按文件路径加载模块，避免为了测试去改目录结构。

只加载不依赖 torch / transformers 的模块：metrics 与数据转换层都是纯 Python + numpy，
可以在没有 GPU、没有模型权重的环境里完整跑通。
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module(name, rel_path):
    """按相对路径加载模块，并登记到 sys.modules（便于 dataclass/pickle 等按名查找）。"""
    path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def laya_metrics():
    return load_module("laya_metrics", os.path.join("common", "laya", "laya_metrics.py"))


@pytest.fixture(scope="session")
def check_leak():
    return load_module("check_leak", os.path.join("common", "laya", "check_leak.py"))


@pytest.fixture(scope="session")
def strict_acc():
    return load_module("strict_acc", os.path.join("common", "laya", "strict_acc.py"))


@pytest.fixture(scope="session")
def validate_mod():
    return load_module("validate_laya_format", os.path.join("datasets", "validate_laya_format.py"))


@pytest.fixture(scope="session")
def convert_mod():
    return load_module("to_laya_format", os.path.join("datasets", "to_laya_format.py"))


@pytest.fixture(scope="session")
def finbert():
    return load_module("finbert_baseline", os.path.join("FinBert", "scripts", "finbert_baseline.py"))


@pytest.fixture
def isolated_convert(convert_mod, tmp_path, monkeypatch):
    """把转换脚本的 BASE / OUT 指向临时目录，避免读写仓库内的 datasets/。"""
    monkeypatch.setattr(convert_mod, "BASE", str(tmp_path))
    monkeypatch.setattr(convert_mod, "OUT", str(tmp_path / "laya"))
    return convert_mod


# 置信度大量并列的构造用例（59 条，只有 1.0 / 0.8 / 0.6 三个取值）。
# 在这组数据上，np.argsort 默认的 quicksort 与稳定排序会选出不同的前 30 条：
#   quicksort → selective_acc@50 = 0.5333…，稳定排序 → 0.4666…
# 用来锁定「并列时按原始下标定序」这个语义，也就是 sort 的稳定性。
_TIED_CONF = "66661161666818686861811661661888816616618816611161866118681"
_TIED_OK = "10001101100001000010111111010000011111100101011010011011101"


@pytest.fixture(scope="session")
def tied_case():
    """返回 (probs, y)：两分类、argmax 恒为 0，因此 y 直接决定每条的对错。"""
    import numpy as np
    conf = np.array([{"6": 0.6, "8": 0.8, "1": 1.0}[c] for c in _TIED_CONF])
    ok = np.array([int(c) for c in _TIED_OK])
    probs = np.stack([conf, 1.0 - conf], axis=1)
    y = np.where(ok == 1, 0, 1)
    assert (probs.argmax(axis=1) == 0).all()
    return probs, y
