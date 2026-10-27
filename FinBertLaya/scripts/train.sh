#!/bin/bash
# ============================================================================
# FinBERT-Laya 训练：与 mmBERT 版 Laya 完全相同的 RLCD 配方，只换 encoder
#
#   数据 : ${LAYA_DATA}/all_train.jsonl   53,341 行 / 55,562 题
#   基座 : ${FB_BASE_OUT}  (FinBERT encoder 102.27M + 决策头 14.97M = 117.24M)
#   对照 : mmBERT 版 4 epoch / 3068 步 / 32 分钟 / 显存 9.2GB
#          本实验   4 epoch / 3068 步 / 23 分钟 / 显存 ~3.8GB
#
# 前置: 先跑 make_fb_base.py 生成 base checkpoint
# 用法: bash train.sh
# ============================================================================
set -uo pipefail

ROOT=${LAYA_ROOT:-/data/laya-fin}
DATA=${LAYA_DATA:-${ROOT}/laya}
BASE=${FB_BASE_OUT:-${ROOT}/fb_base}
OUT=${FB_CKPT:-${ROOT}/fb_ckpt}
REPORTS=${LAYA_REPORTS:-${ROOT}/reports}
ENV=${LAYA_ENV:-/data/miniconda/envs/laya}

export PATH=${ENV}/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-${ROOT}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${OUT}" "${REPORTS}"

echo "=== 训练开始 $(date) ==="
# 超参与 mmBERT 版逐项一致，唯一差别是 --base 指向 FinBERT-Laya base
laya-train --data "${DATA}/all_train.jsonl" --eval "${DATA}/finfe_eval.jsonl" \
  --out "${OUT}" --base "${BASE}" \
  --epochs 4 --micro-batch 24 --grad-accum 3 --device cuda \
  --shuffle-options --label-smoothing 0.05 \
  2>&1 | tee "${REPORTS}/fb_train.log"
echo "=== 训练结束 $(date) ==="
