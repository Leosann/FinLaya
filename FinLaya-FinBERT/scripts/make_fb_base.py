# -*- coding: utf-8 -*-
"""构造 FinBERT-Laya base checkpoint。

思路：Laya 的 DecisionModel 是 encoder-agnostic 的——决策头按 `encoder.config.hidden_size`
动态构建（common.py: `d = encoder.config.hidden_size`），`build_model` 用的是通用
`AutoModel`，`cfg["encoder"]` 接受任意 HF ID。所以只要 hidden_size 一致，就能把 mmBERT
换成 FinBERT，**不需要改 laya 一行源码**。

mmBERT-base 与 FinBERT 的 hidden_size 都是 768，决策头结构完全一致（14.97M 参数），
差别全在 encoder 与 embedding：
    mmBERT : vocab 256,000 -> embedding 196.61M, 22 层
    FinBERT: vocab  21,128 -> embedding  16.23M, 12 层

本脚本做的事：
  1. 复制原 ckpt 的 rl_agent_config.json，改 encoder / max_len / temperature
  2. build_model(cfg, pretrained=True)  -> encoder 载 FinBERT 预训练权重，决策头随机初始化
  3. save_checkpoint(...)  -> 写出 laya.load 能读的四件套
  4. load_checkpoint(...)  -> strict=True 回读验证，并跑一次前向确认输出形状
"""
import json
import os
import sys

# 全部可用环境变量覆盖，默认值即本次实验的服务器布局
ROOT = os.environ.get("LAYA_ROOT", "/data/laya-fin")
SRC = os.environ.get("LAYA_CKPT", os.path.join(ROOT, "ckpt"))          # mmBERT 版 ckpt，只借它的 cfg 骨架
OUT = os.environ.get("FB_BASE_OUT", os.path.join(ROOT, "fb_base"))      # 产出的 FinBERT-Laya base
ENCODER = os.environ.get("FB_ENCODER", "yiyanghkust/finbert-tone-chinese")
MAX_LEN = int(os.environ.get("FB_MAX_LEN", "512"))   # BERT 的 max_position_embeddings，不能沿用 mmBERT 的 1024
HEAD_MAX_LEN = int(os.environ.get("FB_HEAD_MAX_LEN", "256"))


def sec(t):
    print("\n=== %s ===" % t, flush=True)


def main():
    import torch
    from transformers import AutoTokenizer

    from laya.common import build_model
    from laya.train import load_checkpoint, save_checkpoint

    os.makedirs(OUT, exist_ok=True)

    sec("1. 读取原 cfg 并改造")
    with open(os.path.join(SRC, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    print("  before: encoder=%s max_len=%s head_layers=%s act_costs=%s"
          % (cfg.get("encoder"), cfg.get("max_len"), cfg.get("head_layers"), cfg.get("act_costs")),
          flush=True)

    cfg["encoder"] = ENCODER
    cfg["max_len"] = MAX_LEN
    cfg["head_max_len"] = HEAD_MAX_LEN
    # 温度与弃权阈值是在 mmBERT 表征上拟合的，换 encoder 后失效，重置为中性值等重训重拟
    cfg["temperature"] = [1.0, 1.0, 1.0]
    cfg.pop("abstention_thresholds", None)
    # 别继承 mmBERT 那轮的训练元数据
    cfg.pop("training", None)

    print("  after : encoder=%s max_len=%s head_layers=%s act_costs=%s"
          % (cfg.get("encoder"), cfg.get("max_len"), cfg.get("head_layers"), cfg.get("act_costs")),
          flush=True)

    sec("2. 构建模型（encoder 载 FinBERT 预训练权重，决策头随机初始化）")
    tok = AutoTokenizer.from_pretrained(ENCODER)
    print("  tokenizer: %s | vocab %d | model_max_length %s"
          % (type(tok).__name__, tok.vocab_size, tok.model_max_length), flush=True)

    model = build_model(cfg, pretrained=True)
    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_head = n_total - n_enc
    print("  encoder   : %-40s %8.2f M" % (type(model.encoder).__name__, n_enc / 1e6), flush=True)
    print("  decision  : %-40s %8.2f M" % ("head+type_emb+scorer+act_head", n_head / 1e6), flush=True)
    print("  total     : %-40s %8.2f M" % ("", n_total / 1e6), flush=True)
    print("  encoder.config.hidden_size = %d -> head d = %d, nhead = %d"
          % (model.encoder.config.hidden_size, model.encoder.config.hidden_size,
             max(1, model.encoder.config.hidden_size // 64)), flush=True)

    sec("3. 写出 checkpoint")
    save_checkpoint(model, tok, cfg, OUT)
    for name in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, name)
        sz = os.path.getsize(p) / 1048576 if os.path.isfile(p) else 0
        print("  %-24s %8.2f MB" % (name, sz), flush=True)
    print("  encoder/ : %s" % sorted(os.listdir(os.path.join(OUT, "encoder"))), flush=True)
    print("  tokenizer/: %s" % sorted(os.listdir(os.path.join(OUT, "tokenizer"))), flush=True)

    sec("4. strict 回读验证")
    m2, tok2, cfg2 = load_checkpoint(OUT)
    n2 = sum(p.numel() for p in m2.parameters())
    print("  load_checkpoint OK: %s | params %.2fM | encoder=%s | max_len=%s"
          % (type(m2).__name__, n2 / 1e6, cfg2.get("encoder"), cfg2.get("max_len")), flush=True)

    sec("5. 前向 smoke test")
    m2.eval()
    texts = ["试仓500股，看看门道！", "方面词：长谷投资\n语句：请问长谷投资是否是你旗下的公司？【长谷投资】"]
    enc = tok2(texts, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    ids, am = enc["input_ids"], enc["attention_mask"]
    B, T = ids.shape
    # marker_pos/marker_mask 是 head 读选项标记位置的张量，这里给合法占位值
    marker_pos = torch.zeros(B, 4, dtype=torch.long)
    marker_mask = torch.ones(B, 4, dtype=torch.bool)
    qtype = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        logits, act_logits = m2(ids, am, marker_pos, marker_mask, qtype)
    print("  input ids %s -> logits %s, act_logits %s"
          % (tuple(ids.shape), tuple(logits.shape), tuple(act_logits.shape)), flush=True)
    probs = torch.softmax(logits[0], -1)
    print("  sample row0 probs = %s" % [round(float(x), 4) for x in probs], flush=True)

    print("\nMAKE_BASE_DONE out=%s" % OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
