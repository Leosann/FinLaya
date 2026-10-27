#!/bin/bash
# ============================================================================
# FinLaya：Laya 决策模型在中文金融三任务上的微调与评测
#
#   训练数据 : ${LAYA_DATA}/all_train.jsonl    53,341 行 / 55,562 题
#   验证数据 : ${LAYA_DATA}/finfe_eval.jsonl   2,020 条，三分类
#   基座     : convaiinnovations/laya  (--base multilingual = mmBERT-base 322M)
#   参考硬件 : RTX 4090 24GB / torch 2.6.0+cu126 / transformers 4.57.6 / laya 0.4.15
#
# 用法:
#   bash train.sh dryrun   只构建训练项、不加载权重（先跑，验证数据能被正确解析）
#   bash train.sh train    正式训练（4 epoch，实测 32 分钟 / 显存 9.2GB）
#   bash train.sh eval     三个测试集评测（内部转调 eval_ft.sh）
#   bash train.sh all      dryrun -> train -> eval
#
# 可用环境变量（都有默认值，默认值即本次实验的服务器布局）:
#   LAYA_ROOT     /data/laya-fin              实验根目录
#   LAYA_DATA     ${LAYA_ROOT}/laya           转换好的 JSONL
#   LAYA_CKPT     ${LAYA_ROOT}/ckpt           训练产物
#   LAYA_REPORTS  ${LAYA_ROOT}/reports        评测报告
#   LAYA_ENV      /data/miniconda/envs/laya   含 laya-train / laya-evals 的 conda 环境
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${LAYA_ROOT:-/data/laya-fin}
DATA=${LAYA_DATA:-${ROOT}/laya}
OUT=${LAYA_CKPT:-${ROOT}/ckpt}
REPORTS=${LAYA_REPORTS:-${ROOT}/reports}
ENV=${LAYA_ENV:-/data/miniconda/envs/laya}

export PATH=${ENV}/bin:$PATH
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}   # 国内直连 huggingface 不通
export HF_HOME=${HF_HOME:-${ROOT}/hf_cache}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

MODE="${1:-dryrun}"
mkdir -p "${OUT}" "${HF_HOME}" "${REPORTS}"

# 三任务合训：同一个 --data 文件里混着 sentiment_absa / sentiment_social /
# neg_has / neg_subject 四种题，laya-train 按 qid 自动分组学习各自的决策头。
COMMON="--data ${DATA}/all_train.jsonl --eval ${DATA}/finfe_eval.jsonl --out ${OUT} --base multilingual"

case "${MODE}" in
  dryrun)
    echo "=== DRY RUN：构建训练项，不加载权重、不训练 ==="
    laya-train ${COMMON} --dry-run 2>&1 | tee "${REPORTS}/dryrun.log"
    ;;

  train)
    echo "=== 训练开始 $(date) ==="
    # 4090 24GB，322M 编码器 + 1024 维决策头：
    #   micro-batch 24 x grad-accum 3 = 有效 batch 72，实测显存 9.2GB
    #   显存吃紧可退到 --micro-batch 8 --grad-accum 8（有效 batch 64）
    laya-train ${COMMON} \
        --epochs 4 \
        --micro-batch 24 \
        --grad-accum 3 \
        --device cuda \
        --shuffle-options \
        --label-smoothing 0.05 \
        2>&1 | tee "${REPORTS}/train.log"
    echo "=== 训练结束 $(date) ==="
    ;;

  eval)
    # 注意：微调后的本地 ckpt 不能用 `laya-evals run --model <dir>`，该参数只认别名
    bash "${SCRIPT_DIR}/eval_ft.sh"
    ;;

  all)
    bash "$0" dryrun && bash "$0" train && bash "$0" eval
    ;;

  *)
    echo "用法: bash train.sh [dryrun|train|eval|all]"; exit 2 ;;
esac
