#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 FinBERT-Laya 权重上传到 HuggingFace Hub。

为什么需要你在本地跑：当前开发环境（本机与 HPC 服务器）都无法访问
huggingface.co / api.huggingface.co / cdn-lfs.huggingface.co（全部超时），
只有只读镜像 hf-mirror.com 可用，镜像不支持写入。所以上传这一步必须在
一个能访问 HuggingFace 的网络里执行。

用法
----
    # 1) 安装（如果还没有）
    pip install -U huggingface_hub

    # 2) 登录（二选一）
    export HF_TOKEN=hf_xxxxxxxxxxxx        # 或 hf auth login

    # 3) 上传
    python weights/upload_to_hf.py --repo <你的用户名>/FinLaya

    # 只看要传什么、不真传
    python weights/upload_to_hf.py --repo <用户名>/FinLaya --dry-run

    # 想用国内镜像端点（注意：hf-mirror 不支持写，会失败）
    python weights/upload_to_hf.py --repo ... --endpoint https://hf-mirror.com

Token 权限
----------
在 https://huggingface.co/settings/tokens 创建，类型选 **Write**。

脚本会做什么
------------
1. 校验 weights/finlaya-finbert/ 下的 3 个分片，合并成 model.safetensors
2. 校验 SHA256（745cb939...）
3. 组一个符合 laya 加载布局的上传目录（该目录的内容会落到**仓库根**）：
       model.safetensors / rl_agent_config.json / questions.json
       encoder/config.json / tokenizer/* / README.md（模型卡）
4. 创建仓库（已存在则复用）并上传
5. 打印仓库地址和加载示例
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "finbert-laya")
MODEL_CARD = os.path.join(HERE, "hf-model-card.md")
QUESTIONS = os.path.join(HERE, "..", "FinLaya", "configs", "questions.json")

EXPECT_SHA256 = "745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3"
PARTS = ["model.safetensors.part-00", "model.safetensors.part-01", "model.safetensors.part-02"]


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def merge_parts(dst_dir):
    """把分片合并到 dst_dir/model.safetensors，返回路径。"""
    out = os.path.join(dst_dir, "model.safetensors")
    with open(out, "wb") as w:
        for p in PARTS:
            src = os.path.join(SRC, p)
            if not os.path.exists(src):
                raise SystemExit("[x] 缺少分片 %s" % src)
            with open(src, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 20)
    return out


def build_upload_dir(tmp):
    """组出要上传的目录结构。"""
    # 权重
    merged = merge_parts(tmp)
    size_mb = os.path.getsize(merged) / 1048576
    print("[*] 合并完成 %.1f MB，校验 SHA256 ..." % size_mb, end=" ", flush=True)
    got = sha256_of(merged)
    if got != EXPECT_SHA256:
        print("不匹配")
        raise SystemExit("    期望 %s\n    实际 %s" % (EXPECT_SHA256, got))
    print("OK")

    # 配置与 tokenizer
    for rel in ("rl_agent_config.json", "encoder/config.json",
                "tokenizer/special_tokens_map.json", "tokenizer/tokenizer.json",
                "tokenizer/tokenizer_config.json", "tokenizer/vocab.txt"):
        src = os.path.join(SRC, rel)
        if not os.path.exists(src):
            raise SystemExit("[x] 缺少 %s" % src)
        dst = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)

    # 题面定义（推理端要按它复原选项空间）
    if os.path.exists(QUESTIONS):
        shutil.copy2(QUESTIONS, os.path.join(tmp, "questions.json"))
    else:
        print("[!] 没找到 questions.json，跳过（推理时需自行提供题面）")

    # 模型卡
    if os.path.exists(MODEL_CARD):
        shutil.copy2(MODEL_CARD, os.path.join(tmp, "README.md"))
    else:
        print("[!] 没找到 hf-model-card.md，将不带模型卡")
    return tmp


def list_files(d):
    out = []
    for r, _, fs in os.walk(d):
        for f in sorted(fs):
            p = os.path.join(r, f)
            out.append((os.path.relpath(p, d).replace("\\", "/"), os.path.getsize(p)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="例如 zhanghanshuo/FinLaya")
    ap.add_argument("--token", default=None, help="缺省读 HF_TOKEN 环境变量或本地登录态")
    ap.add_argument("--endpoint", default=None, help="自定义 Hub 端点（默认官方）")
    ap.add_argument("--private", action="store_true", help="创建私有仓库")
    ap.add_argument("--dry-run", action="store_true", help="只准备并列出文件，不上传")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    tmp = tempfile.mkdtemp(prefix="hf_upload_")
    try:
        build_upload_dir(tmp)
        files = list_files(tmp)
        total = sum(s for _, s in files) / 1048576
        print("\n[*] 待上传 %d 个文件，合计 %.1f MB：" % (len(files), total))
        for name, size in files:
            print("      %-40s %8.2f MB" % (name, size / 1048576))

        if args.dry_run:
            print("\n[dry-run] 不执行上传。去掉 --dry-run 即真正上传。")
            print("          临时目录保留在 %s" % tmp)
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise SystemExit("\n[x] 未安装 huggingface_hub：pip install -U huggingface_hub")

        kw = {}
        if token:
            kw["token"] = token
        if args.endpoint:
            kw["endpoint"] = args.endpoint
            os.environ["HF_ENDPOINT"] = args.endpoint

        api = HfApi(**kw)

        print("\n[*] 检查身份 ...", end=" ", flush=True)
        who = api.whoami()
        print(who.get("name", "?"))

        print("[*] 创建/复用仓库 %s ..." % args.repo, end=" ", flush=True)
        url = api.create_repo(repo_id=args.repo, repo_type="model",
                              private=args.private, exist_ok=True)
        print("OK")

        print("[*] 上传中（223 MB，视带宽约 1-10 分钟）...")
        # path_in_repo="" 是关键：省缺时 huggingface_hub 会把 folder_path 的
        # 目录名（如 hf_upload_xxxx）当成仓库内子目录，导致文件落到
        # <repo>/hf-ready/... 而不是仓库根，模型卡和权重都不会被正确加载。
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.repo,
            repo_type="model",
            path_in_repo="",
            commit_message="Upload FinBERT-Laya 117M (CFSC-ABSA + FinFE + FinNSP)",
        )

        print("\n" + "=" * 62)
        print("上传完成：https://huggingface.co/%s" % args.repo)
        print("=" * 62)
        print("""
加载方式：

    import laya
    router = laya.Router(device="cuda", preload=False)
    name = router.register("finbert-laya", "%s")
    ans = router.predict("试仓500股，看看门道！", questions, model=name)

题面定义见仓库里的 questions.json（必须与训练时完全一致）。
""" % args.repo)
        return 0
    finally:
        if not args.dry_run:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
