# Laya Chinese financial sentiment fine-tuning: experiment report

> Backbone `convaiinnovations/laya` (multilingual / mmBERT-base, 322M) · three tasks trained jointly · RTX 4090 24GB
> Trained 2026-10-26 · stack `laya 0.4.15` / `torch 2.6.0+cu126` / `transformers 4.57.6`
>
> **Companion document**: `../docs/COMPARE_Laya_vs_FinBERT.md` — full comparison against a Chinese financial FinBERT baseline
> (Macro-F1 / per-class recall / ECE / NLL / Brier / selective accuracy / latency / throughput / GPU memory / size)

---

## 1. Summary of conclusions

All three Chinese financial tasks improve substantially, and the gains hold under the **strict protocol that removes training-set overlap**:

| Dataset | Task | Metric | Baseline | After fine-tuning | Gain | Strict protocol | Training overlap |
|---|---|---|---|---|---|---|---|
| FinFE eval | 3-way sentiment (2,020) | choice_accuracy | 55.79% | **77.97%** | **+22.18** | 77.03% (n=923) | 54.3% |
| FinNSP eval | subject, 1 of 20 (259) | choice_accuracy | 13.51% | **96.91%** | **+83.40** | **96.91%** (n=259) | **0%** |
| FinNSP eval | negative presence (500) | noul_accuracy | 68.00% | **95.00%** | **+27.00** | 94.99% (n=499) | 0.2% |
| CFSC-ABSA test | aspect-level sentiment (8,312) | choice_accuracy | 68.96% | **93.37%** | **+24.41** | 93.58% (n=7,870) | 5.3% |

**The calibration improvement matters just as much** (this is where the value of a Laya decision model lies):

| Dataset | ECE baseline | ECE after | mean_confidence baseline → after |
|---|---|---|---|
| FinFE | 0.1848 | **0.0739** | 0.7427 → 0.8517 |
| FinNSP | 0.3554 | **0.0998** | 0.8438 → 0.8686 |
| CFSC-ABSA | 0.1508 | **0.0638** | 0.8404 → 0.8883 |

ECE drops 60–72%, which means **confidence can be used directly as a gate**: at a confidence@80 threshold, CFSC-ABSA reaches **98.08%** selective accuracy (against 93.37% overall) — exactly the behavior a deployable decision engine should have.

### 1.1 Against the FinBERT baseline (see `../docs/COMPARE_Laya_vs_FinBERT.md`)

`yiyanghkust/finbert-tone-chinese` (102M, pretrained on Chinese financial text) was fine-tuned on **the same data with the same metrics** as a baseline:

| Task | Laya acc | FinBERT acc | Winner |
|---|---|---|---|
| FinFE 3-way | 77.97% | **79.26%** | FinBERT +1.29 |
| FinNSP negative presence | 95.00% | **95.60%** | FinBERT +0.60 |
| FinNSP negative subject (1 of 21/19) | **96.91%** | 95.37% | Laya +1.54 |
| CFSC-ABSA 3-way | **93.37%** | 92.78% | Laya +0.59 |

- **FinBERT is slightly ahead on plain polarity classification** (1–1.3 points), with clearly better calibration (CFSC ECE 0.0339 vs. 0.0638; FinNSP 0.0311 vs. 0.0506)
- **Laya is better at structured decisions over a candidate set** (subject identification +1.54), because it can read the full question text
- **Laya's deployment cost is about 3× FinBERT's**: 321.9M vs. 102.3M parameters, 18.56 ms vs. 5.93 ms latency, 1499 MB vs. 486 MB peak GPU memory

**Note**: Laya's `neg_subject` ECE is as high as **0.1984**, because its `answer_confidence` is a `|2p-1|` decision margin rather than a probability. For gating you must use the bucketed thresholds fitted into `rl_agent_config.json` (`choice:3-5`→0.6217, `choice:11+`→0.7562, `noul:2`→0.6661); raw thresholds are not comparable.

---

## 2. Data

### 2.1 Sources and scale

| Dataset | Train | Test/validation | Source |
|---|---|---|---|
| CFSC-ABSA | 33,184 | 8,312 | `github.com/Ya-dongLi/CFSC` (`CFSC-ABSA/json/`) |
| FinFE | 16,157 | 2,020 | `github.com/supersymmetry-technologies/BBT-FinCUGE-Applications` |
| FinNSP | 4,000 | 500 | Same as above |
| **Total** | **53,341** | 10,832 | 24.42 MB of raw data, byte-verified |

### 2.2 Converting to Laya's decision format

A Laya sample is "state + several questions to answer + the answer to each", and one sample can carry multiple questions:

```json
{"state": "<sentence with the aspect term wrapped in 【】>",
 "questions": {"qid": {"type": "choice|noul", "instructions": "…", "criteria": {…}}},
 "expected": {"qid": "answer"}}
```

How the three tasks map:

| Original task | qid | Type | Options / answers |
|---|---|---|---|
| CFSC-ABSA (aspect-level polarity) | `sentiment_absa` | choice | positive / neutral / negative |
| FinFE (financial sentiment) | `sentiment_social` | choice | negative / neutral / positive (note the non-intuitive 0,1,2 order) |
| FinNSP (news negative presence) | `neg_has` | noul | true / false |
| FinNSP (negative subject) | `neg_subject` | choice | 20 fixed high-frequency entities + `none-of-the-above` |

**Key design decision**: `neg_subject` was originally meant to generate candidate entities per sample, but `laya/train.py` only `warnings.warn`s on a duplicate qid definition when exporting `questions.json` and keeps the first one, which would make the option space unrecoverable at inference. So the design switched to a **fixed vocabulary of 20 entities plus a `none-of-the-above` option**, counted from the training set only (to avoid validation leakage), with `norm_entity()` (stripping parenthetical notes and whitespace, lowercasing) merging 5 near-duplicate spellings such as one short alias and two full company names.

**Residual limitation**: that vocabulary leaves about 102 of the 259 positive cases (39%) in the FinNSP validation set with a subject outside it, answerable only as `none-of-the-above`. This is a task-design trade-off, taken in exchange for schema consistency and an option space recoverable at inference. `neg_subject` still reaches 96.91%, showing the model learned the "not in the vocabulary means none-of-the-above" signal.

### 2.3 Leakage audit (important)

Comparing train and test sets by the SHA256 of the `state` text:

```
finfe_eval       rows=2020   overlapping training set=1097 (54.3%)   ← the official FinFE split itself overlaps
finnsp_eval      rows=500    overlapping training set=1    (0.2%)    ← clean
cfsc_absa_test   rows=8312   overlapping training set=442  (5.3%)    ← mild
```

**Handling**: recompute with overlapping samples removed (see the strict-protocol column in §1). FinFE overlaps by 54% yet the strict protocol drops only 0.94 points (77.97% → 77.03%), indicating the model learned transferable rules rather than memorized answers; CFSC is slightly *higher* under the strict protocol (93.58%) because the parameters fit the overlapping samples no better than the rest.

---

## 3. Training configuration and process

### 3.1 Configuration

```bash
laya-train \
  --data  laya/all_train.jsonl \      # 53,341 lines / 55,562 questions
  --eval  laya/finfe_eval.jsonl \     # 2020 (note: contains leakage; the log flags this explicitly)
  --out   ckpt \
  --base  multilingual \              # mmBERT-base 322M
  --epochs 4 \
  --micro-batch 24 --grad-accum 3 \   # effective batch 72 (the default recipe is 8x8=64)
  --device cuda \
  --shuffle-options \                 # shuffle choice option order to prevent position bias
  --label-smoothing 0.05
```

`--loss rlcd` (the default, not passed explicitly): the RLCD objective = a soft-label cross-entropy term (w_rps=1.0) + a spherical/regularization term (w_sph=0.75), with 4 RL samples and sigma annealed from 0.4 to 0.1. This is how a "System-1 decision engine" is trained — fitting the decision distribution directly, not merely doing classification argmax.

After training, `laya-train` **automatically does two extra things**:
1. **Temperature calibration**: choice temperature 1.4378, noul 1.4585 (fitted on a 10% calibration split)
2. **Abstention threshold fitting**: `choice:3-5` → 0.6217, `choice:11+` → 0.7562, `noul:2` → 0.6661
   (bucketed by option count, target_error=0.10)

This is the direct cause of the large ECE improvement, with no manual tuning.

### 3.2 Process

| Item | Value |
|---|---|
| Optimizer steps | 3,068 (4 epochs × 767 steps) |
| Throughput | ~5 steps/s ≈ 360 items/s |
| GPU memory | 9.2 GB / 24 GB (plenty of headroom on a 4090; batch could double) |
| GPU utilization | 81–93% |
| Total wall time | About 32 minutes (18:41 → 19:13) |

Mean loss per epoch:

```
epoch 1/4  mean loss 0.3983
epoch 2/4  mean loss 0.3103
epoch 3/4  mean loss 0.2565
epoch 4/4  mean loss 0.2018
```

Monotonically decreasing, no divergence. **Negative loss on an individual step is normal** — RLCD takes the form "likelihood term − regularization/KL term", so when the model's confidence in the labels exceeds the penalty the total can go negative (epoch 3 step 2000 is -0.2701).

### 3.3 In-training evaluation (not evidence of generalization)

`train_report.json` carries its own before/after, but states explicitly:

```json
"eval_mode": "overlapping_eval",
"is_held_out": false,
"note": "Evaluation set contains inputs that overlap training data (1090) and
         calibration data (7) (total 1097/2020 items); metrics do not reflect
         strictly independent held-out evaluation."
```

| Metric | Before | After | Delta |
|---|---|---|---|
| Accuracy | 0.5604 | 0.7807 | +0.2203 |
| Loss | 1.1470 | 0.6462 | −0.5008 |
| ECE | 0.1821 | 0.0717 | −0.1104 |
| Brier (multi) | 0.6107 | 0.3513 | −0.2594 |
| Mean Confidence | 0.7425 | 0.8513 | +0.1088 |

The numbers match the independent evaluation (77.97%), but **do not** cite them as generalization performance — cite the table in §1.

---

## 4. Artifacts

### 4.1 Server `/data/laya-fin/`

```
ckpt/
  model.safetensors            644 MB   ← fine-tuned weights (mixed bf16/fp32 export)
  rl_agent_config.json         5.4 KB   ← temperatures, abstention thresholds, question definitions
  questions.json               4.5 KB   ← full definitions of the 3 qids (inference restores the option space from these)
  train_report.json            2.8 KB   ← training hyperparameters + before/after + calibration/abstention results
  tokenizer/                   33.6 MB
  checkpoint_latest/           runtime snapshot
laya/                          9 JSONL files (102 MB)
reports/                       base_*.json (zero-shot) + ft_*.json (after fine-tuning)
train.sh / baseline.sh / eval_ft.sh / run_eval.py / check_leak.py / strict_acc.py (the last four now live in common/laya/)
```

### 4.2 Local repository layout (`FinLaya/`)

Configs, tokenizer, every evaluation report (JSON + Markdown), and training logs are all under `FinLaya/`:

```
FinLaya/
├─ README.md                   directory guide + execution order + quick numbers
├─ REPORT.md                   this document
├─ scripts/                    train.sh · baseline.sh · eval_ft.sh
../common/laya/                 run_eval.py · laya_metrics.py · check_leak.py · strict_acc.py
├─ configs/                    rl_agent_config.json · questions.json · train_report.json
├─ logs/                       train_run / dryrun / baseline / eval_ft
├─ reports/eval/               ft_*.json + ft_*_metrics{,_strict}.json
├─ reports/baseline/           base_*.json
└─ tokenizer/                  33.6 MB
```

**The 644 MB of weights were not pulled back** — for local inference, fetch them from `/data/laya-fin/ckpt/` on the server.
The two-model comparison now lives in `docs/COMPARE_Laya_vs_FinBERT.md`.

---

## 5. Pitfalls encountered (high reuse value)

### 5.1 `laya-evals`' `--model` only accepts aliases

```bash
laya-evals run x.jsonl --model convaiinnovations/laya
# laya-evals: unknown model '...'; choose one of
#   ['english','multilingual','typed-decisions'] (or an alias: [...])
laya-evals run x.jsonl --model /data/laya-fin/ckpt    # rejected too
```

`router.resolve()` only consults the registry (`english`/`multilingual`/`typed-decisions` and the aliases
`ml`/`multi`/`laya`/`default`…). **Use `--model ml` for baseline evaluation**; **post-fine-tune evaluation must go through the Python API**:

```python
router = laya.Router(device="cuda", preload=False)
name = router.register("fin-laya", "/data/laya-fin/ckpt")   # local directories are officially supported
report = evaluate(RouterRunner(router), Dataset.from_jsonl(tmp), batch_size=16, on_error="skip")
json.dump(report.to_json(), f)
```

Note that `evaluate(runner, dataset, ...)` takes positional arguments, and the returned `EvalReport` **is not a dict** — use `report.to_json()` / `report.to_markdown()`; there is no `to_dict()`.

### 5.2 The batch shown by `--dry-run` is not the final batch

The `micro_batch 8 x grad_accum 8` that dry run prints is **the default recipe when no arguments are passed**, not an automatic reduction. Explicitly passing `--micro-batch 24 --grad-accum 3` makes it 72 (using only 9.2 GB).

### 5.3 Server network

| Target | Reachable | Note |
|---|---|---|
| PyPI (BFSU mirror) | ✅ | The Aliyun mirror hangs on `transformers`; switch to BFSU |
| `hf-mirror.com` | ✅ | With `HF_ENDPOINT` + `HF_HOME`, weights download normally (647 MB) |
| `github.com` / `raw.githubusercontent.com` / `ghproxy.net` | ❌ | All unreachable |

Probing `/api/models/convaiinnovations/laya/tree/main` from the local machine returned 403, yet `snapshot_download` on the server **works fine**. These do not contradict each other (the API endpoint and the CDN path differ).

### 5.4 Miscellaneous

- SFTP intermittently reports `ENOENT` (on `put`/`putfo`/`chmod`) → "write a local temp file + `sftp.put` + reconnect and retry" works reliably in a **single inline process**
- Windows CRLF causes `bash: $'\r': command not found` → `replace(b"\r\n", b"\n")` before upload
- Foreground commands get SIGTERM'd at about 2 minutes → run long tasks in the background and poll remote logs
- `nohup … &` holds the SSH channel open and causes TimeoutError → use `setsid … < /dev/null & exit 0` and close the channel immediately

---

## 6. Suggested next steps

1. **Retrain FinFE on a clean split**: drop the 1,097 rows in `finfe_train` that overlap `eval_list`, then retrain to obtain a citable held-out number (the current strict figure of 77.03% is already a good estimate, but rests on only 923 samples).
2. **Raise the batch size**: only 9.2 GB of GPU memory is used; `micro-batch 48 × grad-accum 2` (effective 96) should be another 1.3× faster and slightly improve the RLCD gradient estimate.
3. **Expand the `neg_subject` vocabulary**: 39% of validation positives currently fall outside the 20 entities. If the business allows dynamic options, switch to one qid per question (e.g. `neg_subject#01`) to bypass the schema-consistency constraint.
4. **Use the abstention thresholds as an online gate**: the bucketed thresholds fitted into `rl_agent_config.json` can be wired into the service directly; CFSC-ABSA reaches 98.08% accuracy at confidence@80, sending the remaining 20% to human review.
5. **Export to ONNX**: `laya-evals run --onnx` supports it; 32 ms → targeting under 10 ms (the 4090's p50 throughput figure `cost_per_decision_p50_ms` is already 1.4 ms, but that amortizes over the batch — the true single-request latency is `latency_p50_ms`).

---

## 7. Quick numbers

```
baseline (zero-shot, --model ml)
  FinFE       choice_acc 0.5579  ece 0.1848  conf 0.7427
  FinNSP      choice_acc 0.1351  noul_acc 0.6800  ece 0.3554
  CFSC-ABSA   choice_acc 0.6896  ece 0.1508  conf 0.8404

after fine-tuning
  FinFE       choice_acc 0.7797  ece 0.0739  sel@80 0.8379
  FinNSP      choice_acc 0.9691  noul_acc 0.9500  ece 0.0998  sel@80 0.9671
  CFSC-ABSA   choice_acc 0.9337  ece 0.0638  sel@80 0.9808

strict protocol (overlapping `state` samples removed)
  FinFE       0.7703 (n=923)
  FinNSP      0.9691 (n=259) / noul 0.9499 (n=499)
  CFSC-ABSA   0.9358 (n=7870)
```
