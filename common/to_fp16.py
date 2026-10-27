# -*- coding: utf-8 -*-
"""把已训练好的分类头 checkpoint 转成 float16，并报告转换前后的体积。

这个脚本是事后补写的：原实验里 FinBERT 的 F16 体积（195.1 MB）是一次性测出来的，
转换代码没留下来，所以这里按同样的做法重写一份，保证结论可复现。

用法:
    python to_fp16.py <model_dir>               # 原地转换
    python to_fp16.py <model_dir> --out <dir>   # 另存，不改原目录
    python to_fp16.py <model_dir> --dry-run     # 只算体积差，不写文件
    python to_fp16.py <model_dir> --check       # 只检查现有权重是什么 dtype

注意事项:
  - 只转权重，不改 tokenizer / config；分类头的 num_labels 等结构信息原样保留。
  - fp16 在 CPU 上前向会报错，转完请在 GPU 上跑（或再转回 float32 用于 CPU 推理）。
  - 原实验实测：fp32 390.2 MB -> fp16 195.1 MB（约 0.500x），
    若你得到的比例明显偏离 0.5，说明权重里混有非 fp32 张量，先跑 --check。
"""
import argparse
import json
import os
import shutil
import sys


def dir_size_mb(d):
    tot = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            tot += os.path.getsize(os.path.join(r, f))
    return tot / 1048576.0


def weights_size_mb(d):
    tot = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            if f.endswith((".safetensors", ".bin", ".pt")):
                tot += os.path.getsize(os.path.join(r, f))
    return tot / 1048576.0


def describe_dtypes(model_dir):
    """列出权重文件里的 dtype 分布（safetensors 直接读 header，不加载权重）。"""
    import struct

    found = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, name)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n).decode("utf-8"))
        for k, v in header.items():
            if k == "__metadata__":
                continue
            dt = v.get("dtype", "?")
            found[dt] = found.get(dt, 0) + 1
    return found


def convert(model_dir, out_dir, dry_run=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    before_w = weights_size_mb(model_dir)
    before_d = dir_size_mb(model_dir)
    print("[in ] %s" % model_dir)
    print("      weights %.1f MB / dir %.1f MB" % (before_w, before_d))
    print("      dtypes %s" % (describe_dtypes(model_dir) or "(no safetensors)"))

    if dry_run:
        print("[dry] 预计 fp16 权重约 %.1f MB（按 0.5x 估算）" % (before_w * 0.5))
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.half()
    if out_dir and os.path.abspath(out_dir) != os.path.abspath(model_dir):
        os.makedirs(out_dir, exist_ok=True)
    target = out_dir or model_dir
    model.save_pretrained(target)
    # tokenizer 若存在一并带上，体积统计才有可比性
    try:
        AutoTokenizer.from_pretrained(model_dir).save_pretrained(target)
    except Exception as e:  # tokenizer 缺失不影响权重转换
        print("[warn] tokenizer 未复制: %s" % e)

    after_w = weights_size_mb(target)
    after_d = dir_size_mb(target)
    print("[out] %s" % target)
    print("      weights %.1f MB / dir %.1f MB" % (after_w, after_d))
    print("      ratio   %.3fx (weights)" % (after_w / before_w if before_w else 0))
    print("      dtypes %s" % (describe_dtypes(target) or "(no safetensors)"))

    # 和原实验的实测值对一下，方便判断是否复现成功
    if abs(after_w - 195.1) < 5.0:
        print("[ok  ] 与原实验的 195.1 MB 一致")
    else:
        print("[note] 与原实验 195.1 MB 有偏差，请确认是同一个 checkpoint")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--out", default=None, help="输出目录；缺省则原地覆盖")
    ap.add_argument("--dry-run", action="store_true", help="只报体积，不写文件")
    ap.add_argument("--check", action="store_true", help="只打印现有 dtype 分布")
    args = ap.parse_args()

    if not os.path.isdir(args.model_dir):
        print("不是目录: %s" % args.model_dir)
        return 2

    if args.check:
        print("weights %.1f MB" % weights_size_mb(args.model_dir))
        print("dtypes  %s" % (describe_dtypes(args.model_dir) or "(no safetensors)"))
        return 0

    return convert(args.model_dir, args.out, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
