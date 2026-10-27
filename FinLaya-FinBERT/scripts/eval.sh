#!/bin/bash
# ============================================================================
# FinBERT-Laya 三测试集评测 + 全套指标复算（全量 & 严格口径）
#
# 用法: bash eval.sh              # 三个测试集
#       bash eval.sh finfe_eval   # 只跑一个
#
# 注意（踩过的坑）: laya_metrics.py **自己会写** <report>_metrics.json 与
# <report>_metrics_strict.json，所以**不要**把它的 stdout 重定向到这两个文件——
# 那会把脚本刚写好的 JSON 覆盖成人类可读文本。stdout 丢 /dev/null 即可。
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${LAYA_ROOT:-/data/laya-fin}
DATA=${LAYA_DATA:-${ROOT}/laya}
CKPT=${FB_CKPT:-${ROOT}/fb_ckpt}
REPORTS=${LAYA_REPORTS:-${ROOT}/reports}
ENV=${LAYA_ENV:-/data/miniconda/envs/laya}
LAYA_TOOLS="$(cd "${SCRIPT_DIR}/../../common/laya" && pwd)"   # 与 FinLaya-mmBERT 共用的评测工具

export PATH=${ENV}/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-${ROOT}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export EVAL_BATCH=${EVAL_BATCH:-16}

mkdir -p "${REPORTS}"
TARGETS=${*:-"finfe_eval finnsp_eval cfsc_absa_test"}

for f in ${TARGETS}; do
  echo ""
  echo "======== EVAL fb_${f} ========"
  python "${LAYA_TOOLS}/run_eval.py" "${DATA}/${f}.jsonl" "${CKPT}" "${REPORTS}/fb_${f}.json" 2>&1 | tail -16
done

echo ""
echo "======== 复算指标（脚本自写 JSON，勿重定向）========"
for f in ${TARGETS}; do
  python "${LAYA_TOOLS}/laya_metrics.py" "${REPORTS}/fb_${f}.json" "${DATA}/${f}.jsonl"          > /dev/null 2>&1
  python "${LAYA_TOOLS}/laya_metrics.py" "${REPORTS}/fb_${f}.json" "${DATA}/${f}.jsonl" --strict > /dev/null 2>&1
  echo "  ${f}: full $(wc -c < "${REPORTS}/fb_${f}_metrics.json") B / strict $(wc -c < "${REPORTS}/fb_${f}_metrics_strict.json") B"
done

echo ""
echo "=== EVAL_DONE $(date) ==="
