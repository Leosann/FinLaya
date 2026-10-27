#!/bin/bash
# ============================================================================
# 未微调基座的三测试集 zero-shot 基线
#
# 关于 --model 的取值：
#   本次实验实际跑通的是 HUB ID（convaiinnovations/laya）。但不同 laya 版本对
#   --model 的解析行为不一致，部分版本会报 unknown model。若这里报错，改用已注册
#   的别名 "ml"（等价于 multilingual），它一定可用：
#       BASE_MODEL=ml bash baseline.sh
# ============================================================================
set -uo pipefail

ROOT=${LAYA_ROOT:-/data/laya-fin}
DATA=${LAYA_DATA:-${ROOT}/laya}
REPORTS=${LAYA_REPORTS:-${ROOT}/reports}
ENV=${LAYA_ENV:-/data/miniconda/envs/laya}
MODEL=${BASE_MODEL:-convaiinnovations/laya}

export PATH=${ENV}/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HOME=${HF_HOME:-${ROOT}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${REPORTS}/baseline"

for f in finfe_eval finnsp_eval cfsc_absa_test; do
  echo ""
  echo "======== BASE ${f} ========"
  laya-evals run "${DATA}/${f}.jsonl" \
      --model "${MODEL}" \
      --device cuda \
      --on-error skip \
      --json "${REPORTS}/baseline/base_${f}.json" \
      2>&1 | tail -22
done
echo "BASELINE_DONE."
