# Hugging Face upload guide

The model repo is live at **https://huggingface.co/zhanghanshuo/FinLaya**

> **Status (2026-10-27)**: all 9 files (235 MB) are uploaded, but they landed **inside an `hf-ready/` subfolder**. The repo root only holds a 28-byte placeholder README, so the model card does not render and `huggingface-cli download` does not produce a valid Laya layout. See "Fixing the current repo" below.

This machine cannot reach `huggingface.co` (the proxy answers `CONNECT` with 502), so the upload has to happen from a browser or a machine with working network access. **The files are already prepared locally.**

## Where the files are

```
D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready\
```

9 files, 224.2 MB in total:

```
hf-ready/
├─ README.md                          model card (with YAML frontmatter)
├─ model.safetensors                 223.64 MB  235 tensors, all F16, 117.24M parameters
├─ rl_agent_config.json               encoder / max_len / temperature / abstain threshold
├─ questions.json                     task definitions for the 4 qids
├─ encoder/config.json                BertConfig (hidden 768 / 12 layers / vocab 21128)
└─ tokenizer/
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   └─ vocab.txt
```

Checksum — the SHA256 of `model.safetensors` should be

```
745cb939f9311d73a3417b41db8b893dd5f9e9a2afeb25473efc547e00da08d3
```

> The directory contents are **hard links** and take no extra disk space. Editing one copy affects the other, which is harmless for a read-only upload.

---

## Fixing the current repo (keeping what you already uploaded)

Three options. The first is recommended.

### Option 1: move the files in the web UI (cheapest — no 235 MB re-upload)

1. Open https://huggingface.co/zhanghanshuo/FinLaya/tree/main/hf-ready
2. Select all five entries (`README.md`, `model.safetensors`, `rl_agent_config.json`, `questions.json`, `encoder/`, `tokenizer/`)
3. Click **Move** (if there is no Move button, use **Edit → Move file** on each entry) and set the target to `/` (repo root)
4. Go back to the root, then delete the now-empty `hf-ready/` folder
5. Open the **Model card** tab and confirm it shows the FinBERT-Laya content instead of a blank page

### Option 2: re-upload to the root with the `hf` CLI

```powershell
pip install -U huggingface_hub
hf auth login          # or: $env:HF_TOKEN = "hf_xxxxxxxxxxxx"

# Note the trailing "." — it means "upload to the repo root"
hf upload zhanghanshuo/FinLaya "D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready" .
```

**This is the argument that was missing last time**: in `hf upload <repo> <local> <path_in_repo>`, an omitted third argument makes `hf` use the local folder name as the remote subfolder, so everything ends up under `hf-ready/`. Only an explicit `.` flattens the files into the root.

Afterwards delete the `hf-ready/` folder at the root (via the web UI) so you are not paying for 235 MB twice.

### Option 3: use the script (idempotent, best for repeat use)

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxx"
python weights/upload_to_hf.py --repo zhanghanshuo/FinLaya
```

The script calls `upload_folder(..., path_in_repo="")`, so it never nests an extra level. Add `--dry-run` to preview what would be uploaded. Because the files already exist under the same names, re-running only overwrites — it does not create duplicates.

---

## Method A: drag and drop in the web UI (easiest when creating a repo from scratch)

1. Open https://huggingface.co/new
2. Set **Owner** to `zhanghanshuo`, **Model name** to `FinLaya`, pick Public visibility, then click **Create model**
3. Open the repo → **Files** tab → **Add file** → **Upload files**
4. Select the **contents** of the `hf-ready` folder (not the folder itself) and drag them into the upload area
   - Dragging the whole folder also works — the browser preserves the `encoder/` and `tokenizer/` subfolders
   - The 223 MB file goes through LFS, so do not close the page while it uploads
5. Type any commit message and click **Commit changes**

> The important part is **not to drag the `hf-ready` folder level itself** — that is exactly what happened last time.

After refreshing, the root should show 9 files, with an `LFS` badge next to `model.safetensors`.

---

## Method B: command line

```powershell
pip install -U huggingface_hub
hf auth login
# or: $env:HF_TOKEN = "hf_xxxxxxxxxxxx"

hf upload zhanghanshuo/FinLaya "D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready" .
```

Create the token at https://huggingface.co/settings/tokens and choose the **Write** type.

The trailing `.` means "upload to the repo root" — **do not drop it**.

---

## Method C: git push (requires git-lfs)

```bash
git lfs install
git clone https://huggingface.co/zhanghanshuo/FinLaya
cd FinLaya
cp -r "D:\WorkBuddyAIWorkhome\2026-10-24-19-45-48\weights\hf-ready\"* .
git lfs track "*.safetensors"
git add .
git commit -m "Upload FinBERT-Laya 117M"
git push
```

Use this only if methods A and B both fail.

---

## Post-upload checklist

Open `https://huggingface.co/zhanghanshuo/FinLaya` and confirm:

- [ ] **9 files at the root**, with `model.safetensors` showing `223.6 MB` and an LFS badge
- [ ] **No `hf-ready/` subfolder**
- [ ] **The model card renders** — the header reads "FinBERT-Laya (117M)", not a blank page and not raw YAML
- [ ] **Subfolder structure is intact** — `encoder/config.json` and `tokenizer/` exist and were not flattened
- [ ] **The right-hand info panel** reports roughly 117M parameters (HF reads this from the safetensors header)
- [ ] **License shows MIT** (the model card frontmatter has `license: mit`; if it still says apache-2.0, the root README has not been replaced yet)

### Why the model page shows no transformers snippet

This is **expected**, not an error. HF looks for `config.json` plus `pytorch_model.bin`/`model.safetensors` at the repo root to generate the `AutoModel.from_pretrained(...)` example. This project uses the Laya checkpoint layout (`encoder/config.json` lives in a subfolder and the weight keys carry an `encoder.` prefix), so transformers auto-loading does not apply.

**Do not** add a `config.json` at the root just to make the snippet appear — that would make HF try to load the model with transformers, which then fails because the weight key prefixes do not match. That is strictly worse.

The correct way to load it is the Laya API:

```python
import laya
router = laya.Router(device="cuda", preload=False)
name = router.register("finbert-laya", "zhanghanshuo/FinLaya")
```

`Router.register` accepts a Hub ID: `resolve_checkpoint_dir` runs `snapshot_download` for exactly those four kinds of files (`rl_agent_config.json` / `model.safetensors` / `tokenizer/*` / `encoder/*`), which is precisely what this directory contains.

---

## Uploading from a different machine

Files on this machine cannot be handed to another one directly. Two options:

1. **Pull from GitHub** (recommended — no need to copy 223 MB)
   ```bash
   git clone https://github.com/Leosann/FinLaya.git
   cd FinLaya/weights/finbert-laya
   bash restore.sh          # merge the shards + verify SHA256
   ```
   Then upload `model.safetensors`, `rl_agent_config.json`, `encoder/` and `tokenizer/` from `weights/finbert-laya/`, plus `weights/hf-model-card.md` renamed to `README.md`. `questions.json` lives in `FinLaya/configs/`.

2. **Copy the whole `hf-ready` directory** (224 MB) via a USB stick or cloud drive.

Either way, run `restore.sh --check` or compare the SHA256 afterwards to confirm nothing was corrupted in transit.
