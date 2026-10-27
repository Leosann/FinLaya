#!/bin/bash
# ============================================================================
# 微调后 checkpoint 在三个测试集上的评测（走 Python API + Router.register 本地目录）
#
# 为什么不用 `laya-evals run --model <ckpt>`：
#   该 CLI 的 --model 只接受注册表别名（english / multilingual / ml / laya / ...），
#   传本地目录会被 router.resolve() 拒绝。官方支持的本地目录入口是
#   Router.register(name, source)，所以统一走 run_eval.py。
#
# 用法:
#   bash eval_ft.sh                      # 三个测试集全跑
#   bash eval_ft.sh finfe_eval           # 只跑一个
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAYA_TOOLS="$(cd "${SCRIPT_DIR}/../../common/laya" && pwd)"   # 与 FinBertLaya 共用的评测工具
ROOT=${LAYA_ROOT:-/data/laya-fin}
DATA=${LAYA_DATA:-${ROOT}/laya}
CKPT=${LAYA_CKPT:-${ROOT}/ckpt}
REPORTS=${LAYA_REPORTS:-${ROOT}/reports}
ENV=${LAYA_ENV:-/data/miniconda/envs/laya}

export PATH=${ENV}/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # 国内直连 huggingface 不通
export HF_HOME=${HF_HOME:-${ROOT}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export EVAL_BATCH=${EVAL_BATCH:-16}
mkdir -p "${REPORTS}"

TARGETS=${*:-"finfe_eval finnsp_eval cfsc_absa_test"}

for f in ${TARGETS}; do
  echo ""
  echo "======== FT ${f} ========"
  python "${LAYA_TOOLS}/run_eval.py" \
      "${DATA}/${f}.jsonl" "${CKPT}" "${REPORTS}/ft_${f}.json" 2>&1 | tail -20
done
echo "FT_EVAL_DONE."
