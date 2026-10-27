#!/bin/bash
# ============================================================================
# 把分片合并回 laya 能直接加载的 checkpoint 目录
#
#   bash restore.sh            # 合并 + 校验
#   bash restore.sh --check    # 只校验，不合并
#
# 合并后目录就是 laya.load() / laya.Router.register() 认的布局：
#   finbert-laya/
#   ├─ model.safetensors      (223.6 MB, fp16)
#   ├─ rl_agent_config.json   (encoder / 温度 / 弃权阈值)
#   ├─ encoder/config.json    (BertConfig)
#   └─ tokenizer/             (BertTokenizerFast)
# ============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EXPECT="745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3"

check() {
  if [ ! -f model.safetensors ]; then
    echo "[!] model.safetensors 不存在，先跑一次不带 --check 的合并"
    return 1
  fi
  echo -n "[*] 校验 SHA256 ... "
  if command -v sha256sum >/dev/null 2>&1; then
    got=$(sha256sum model.safetensors | awk '{print $1}')
  elif command -v shasum >/dev/null 2>&1; then
    got=$(shasum -a 256 model.safetensors | awk '{print $1}')
  else
    echo "跳过（找不到 sha256sum / shasum）"
    return 0
  fi
  if [ "$got" = "$EXPECT" ]; then
    echo "OK"
    return 0
  fi
  echo "不匹配"
  echo "    期望 $EXPECT"
  echo "    实际 $got"
  return 1
}

if [ "${1:-}" = "--check" ]; then
  check
  exit $?
fi

echo "[*] 合并分片 ..."
cat model.safetensors.part-* > model.safetensors
echo "[*] 合并完成：$(du -h model.safetensors | cut -f1)"
check
echo ""
echo "现在可以直接用："
echo "  python -c \"import laya; m = laya.load('$(pwd)')\""
