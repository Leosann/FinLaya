# 手动上传到 HuggingFace

本机无法访问 `huggingface.co`（代理对 `CONNECT` 返回 502），所以上传要你在浏览器或能联网的环境里手动做。**文件已经准备好，就在本地。**

## 文件在哪

```
D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready\
```

9 个文件，合计 224.2 MB：

```
hf-ready/
├─ README.md                          模型卡（带 YAML frontmatter）
├─ model.safetensors                 223.64 MB  235 个张量，全 F16，117.24M 参数
├─ rl_agent_config.json                encoder / max_len / 温度 / 弃权阈值
├─ questions.json                      4 个 qid 的题面定义
├─ encoder/config.json                 BertConfig（hidden 768 / 12 层 / vocab 21128）
└─ tokenizer/
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ vocab.txt
```

校验值：`model.safetensors` 的 SHA256 应为

```
745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3
```

> 目录里是**硬链接**，不额外占磁盘。改动其中一份会影响另一份，但只读上传没有影响。

---

## 方法 A：网页拖拽（最省事，不用命令行）

1. 打开 https://huggingface.co/new
2. **Owner** 选 `zhanghanshuo`，**Model name** 填 `FinLaya-117M`，可见性选 Public，点 **Create model**
3. 进入仓库页 → **Files** 标签 → **Add file** → **Upload files**
4. 把 `hf-ready` 目录里的**内容**（不是目录本身）全选拖进上传区
   - 拖整个文件夹也行，浏览器会保留 `encoder/` 和 `tokenizer/` 的子目录结构
   - 223 MB 单文件会走 LFS，上传过程中不要关页面
5. 下方 Commit message 随便填，点 **Commit changes**

上传完刷新页面，应该能看到 9 个文件，且 `model.safetensors` 旁有 `LFS` 标记。

---

## 方法 B：命令行（更快，支持断点）

```powershell
pip install -U huggingface_hub

# 登录（会提示粘贴 token）
hf auth login
# 或者直接设环境变量
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"

# 上传整个目录
hf upload zhanghanshuo/FinLaya-117M "D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready" .
```

Token 在 https://huggingface.co/settings/tokens 创建，类型选 **Write**。

最后那个 `.` 表示传到仓库根目录，别漏掉，否则会多一层 `hf-ready/` 前缀。

---

## 方法 C：git push（需要 git-lfs）

```bash
git lfs install
git clone https://huggingface.co/zhanghanshuo/FinLaya-117M
cd FinLaya-117M
cp -r "D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready\"* .
git lfs track "*.safetensors"
git add .
git commit -m "Upload FinBERT-Laya 117M"
git push
```

只在方法 A/B 都不顺时用。

---

## 上传后自查

打开 `https://huggingface.co/zhanghanshuo/FinLaya-117M`，确认：

- [ ] **模型卡正常渲染** —— 顶部显示「FinBERT-Laya (117M)」，不是一堆 YAML 原文
- [ ] **文件数 9 个**，`model.safetensors` 显示 `223.6 MB` 且带 LFS 标记
- [ ] **子目录结构正确** —— 有 `encoder/config.json` 和 `tokenizer/`，不是被拍平成根目录
- [ ] **右侧信息栏** 显示参数量约 117M（HF 从 safetensors header 自动读）

### 关于「模型页不显示 transformers 代码片段」

这是**预期行为**，不是出错。HF 会在仓库根目录找 `config.json` + `pytorch_model.bin`/`model.safetensors` 来生成 `AutoModel.from_pretrained(...)` 示例。本项目是 Laya 的 checkpoint 布局（`encoder/config.json` 在子目录、权重键带 `encoder.` 前缀），所以不适用 transformers 的自动加载。

**不要**为了让它显示而往根目录加 `config.json` —— 那会让 HF 尝试用 transformers 加载，然后因为权重键前缀对不上而报错，反而更糟。

正确的加载方式是 Laya 的 API：

```python
import laya
router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "zhanghanshuo/FinLaya-117M")
```

`Router.register` 支持 Hub ID：`resolve_checkpoint_dir` 会 `snapshot_download` 那 4 类文件（`rl_agent_config.json` / `model.safetensors` / `tokenizer/*` / `encoder/*`），正好就是本目录的内容。

---

## 传完之后

告诉我，我用只读通道帮你核对模型卡渲染和文件列表。也可以顺便把 HF 链接补进 GitHub 的 README（现在那里写的是「待上传」）。

---

## 如果要在另一台电脑上传

这台电脑上的文件没法直接给另一台机器。两个办法：

1. **从 GitHub 拉**（推荐，不用拷 223 MB）
   ```bash
   git clone https://github.com/Leosann/FinLaya.git
   cd FinLaya/weights/finbert-laya
   bash restore.sh          # 合并分片 + SHA256 校验
   ```
   然后把 `weights/finbert-laya/` 里的 `model.safetensors`、`rl_agent_config.json`、`encoder/`、`tokenizer/` 和 `weights/hf-model-card.md`（改名为 `README.md`）一起传。`questions.json` 在 `FinLaya/configs/` 下。

2. **U 盘 / 网盘拷 `hf-ready` 整个目录**（224 MB）。

无论哪种方式，上传后都应该跑一次 `restore.sh --check` 或比对 SHA256，确认文件没有在传输中损坏。
