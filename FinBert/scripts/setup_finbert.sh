#!/bin/bash
# FinBERT 基线环境：与 laya 环境隔离（laya 的 torch 2.6.0 已够用，但避免依赖互相干扰）
set -x
source /data/miniconda/etc/profile.d/conda.sh
conda create -y -p /data/miniconda/envs/finbert python=3.11 2>&1 | tail -5
PY=/data/miniconda/envs/finbert/bin/python
$PY -m pip install -q --upgrade pip -i https://mirrors.bfsu.edu.cn/pypi/web/simple
$PY -m pip install --timeout 60 --retries 3 -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
    torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -3
$PY -m pip install --timeout 60 --retries 3 -i https://mirrors.bfsu.edu.cn/pypi/web/simple \
    "transformers>=4.40,<5" scikit-learn numpy tqdm 2>&1 | tail -3
$PY -c "import torch,transformers,sklearn;print('torch',torch.__version__,torch.cuda.is_available());print('tf',transformers.__version__);print('sk',sklearn.__version__)"
echo "FINBERT_ENV_DONE."
