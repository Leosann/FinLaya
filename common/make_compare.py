# -*- coding: utf-8 -*-
"""汇总 Laya 与 FinBERT 的全套指标，生成对比 Markdown。

数据来源（全部在 laya_finetune/reports/ 下）:
  ft_*_metrics.json / ft_*_metrics_strict.json   Laya 全量 / 严格口径
  finbert_all.json                               FinBERT 全量 + 严格口径
  deploy_metrics.json                            两者部署指标
"""
import json
import os

R = os.environ.get("LAYA_REPORTS",
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports"))


def load(p):
    with open(os.path.join(R, p), "r", encoding="utf-8") as f:
        return json.load(f)


def pct(x, d=2):
    return ("%." + str(d) + "f%%") % (100.0 * x) if x is not None else "—"


def f4(x, d=4):
    return ("%." + str(d) + "f") % x if x is not None else "—"


# ---- Laya ----
laya = {
    "finfe_eval": {"full": load("ft_finfe_eval_metrics.json"),
                   "strict": load("ft_finfe_eval_metrics_strict.json")},
    "finnsp_eval": {"full": load("ft_finnsp_eval_metrics.json"),
                    "strict": load("ft_finnsp_eval_metrics_strict.json")},
    "cfsc_absa_test": {"full": load("ft_cfsc_absa_test_metrics.json"),
                       "strict": load("ft_cfsc_absa_test_metrics_strict.json")},
}
# ---- FinBERT ----
fb_raw = load("finbert_all.json")
fb = {
    "finfe_eval": fb_raw["finfe"],
    "finnsp_has": fb_raw["nsp_has"],
    "finnsp_subject": fb_raw["nsp_subject"],
    "cfsc_absa_test": fb_raw["absa"],
}
dep = load("deploy_metrics.json")
# 实测：FinBERT nsp_has 转 fp16 后的体积
FB_FP16_MB = 195.1

QF = {"finfe_eval": "sentiment_social", "cfsc_absa_test": "sentiment_absa"}
QN = {"finnsp_has": "neg_has", "finnsp_subject": "neg_subject"}

rows = []


def add(task_label, dataset, laya_r, fb_r, note=""):
    rows.append({"task": task_label, "dataset": dataset, "laya": laya_r, "fb": fb_r, "note": note})


# FinFE
L = laya["finfe_eval"]
add("金融情绪三分类", "finfe_eval", L["full"][QF["finfe_eval"]], fb["finfe_eval"]["full"])
add("金融情绪三分类（严格）", "finfe_eval", L["strict"][QF["finfe_eval"]], fb["finfe_eval"]["strict"])

# FinNSP
L = laya["finnsp_eval"]
add("新闻有无负面（二分类）", "finnsp_eval", L["full"][QN["finnsp_has"]], fb["finnsp_has"]["full"])
add("新闻有无负面（严格）", "finnsp_eval", L["strict"][QN["finnsp_has"]], fb["finnsp_has"]["strict"])
add("负面主体（21 选 1）", "finnsp_eval", L["full"][QN["finnsp_subject"]], fb["finnsp_subject"]["full"],
    "Laya 21 类 / FinBERT 19 类")

# CFSC
L = laya["cfsc_absa_test"]
add("方面级情感三分类", "cfsc_absa_test", L["full"][QF["cfsc_absa_test"]], fb["cfsc_absa_test"]["full"])
add("方面级情感三分类（严格）", "cfsc_absa_test", L["strict"][QF["cfsc_absa_test"]], fb["cfsc_absa_test"]["strict"])


def table_acc():
    out = ["| 任务 | 样本 | Laya acc | FinBERT acc | 差值 | Laya MacroF1 | FinBERT MacroF1 | 差值 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        l, f = r["laya"], r["fb"]
        out.append("| %s | %d | %s | %s | %+.2f | %s | %s | %+.4f |" % (
            r["task"], l["n"], pct(l["accuracy"]), pct(f["accuracy"]),
            100 * (l["accuracy"] - f["accuracy"]),
            f4(l["macro_f1"]), f4(f["macro_f1"]), l["macro_f1"] - f["macro_f1"]))
    return "\n".join(out)


def table_calib():
    out = ["| 任务 | Laya ECE | FinBERT ECE | Laya NLL | FinBERT NLL | Laya Brier | FinBERT Brier | Laya AURC | FinBERT AURC |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        l, f = r["laya"], r["fb"]
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["task"], f4(l["ece"]), f4(f["ece"]), f4(l["nll"]), f4(f["nll"]),
            f4(l["brier"]), f4(f["brier"]), f4(l["aurc"]), f4(f["aurc"])))
    return "\n".join(out)


def table_selective():
    out = ["| 任务 | Laya @50 | FinBERT @50 | Laya @80 | FinBERT @80 | Laya @90 | FinBERT @90 |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        l, f = r["laya"], r["fb"]
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            r["task"], pct(l.get("selective_acc@50"), 1), pct(f.get("selective_acc@50"), 1),
            pct(l.get("selective_acc@80"), 1), pct(f.get("selective_acc@80"), 1),
            pct(l.get("selective_acc@90"), 1), pct(f.get("selective_acc@90"), 1)))
    return "\n".join(out)


def table_class_recall():
    """各类别召回率对照（同任务同标签才可比）。"""
    out = ["| 任务 | 类别 | Laya P | Laya R | Laya F1 | FinBERT P | FinBERT R | FinBERT F1 | 支持数 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "严格" in r["task"]:
            continue
        l, f = r["laya"], r["fb"]
        lc, fc = l["per_class"], f["per_class"]
        for cls in l["classes"]:
            if cls not in fc:
                continue
            a, b = lc[cls], fc[cls]
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %d |" % (
                r["task"], cls, f4(a["precision"], 3), f4(a["recall"], 3), f4(a["f1"], 3),
                f4(b["precision"], 3), f4(b["recall"], 3), f4(b["f1"], 3), a["support"]))
    return "\n".join(out)


def table_deploy():
    l, f = dep["laya"], dep["finbert"]
    out = ["| 指标 | Laya | FinBERT | 倍数（Laya/FinBERT） |", "|---|---|---|---|"]
    out.append("| 参数量 | %.1f M | %.1f M | %.2f× |" % (l["n_params_M"], f["n_params_M"],
                                                      l["n_params_M"] / f["n_params_M"]))
    out.append("| 权重体积 | %.0f MB | %.0f MB | %.2f× |" % (l["weights_mb"], f["weights_mb"],
                                                         l["weights_mb"] / f["weights_mb"]))
    out.append("| 权重精度 | %s | %s | — |" % (
        "/".join(l["weight_dtypes"].keys()), f["weight_dtype"]))
    out.append("| 单条延迟 p50 | %.2f ms | %.2f ms | %.2f× |" % (
        l["latency_p50_ms"], f["latency_p50_ms"], l["latency_p50_ms"] / f["latency_p50_ms"]))
    out.append("| 单条延迟 p95 | %.2f ms | %.2f ms | %.2f× |" % (
        l["latency_p95_ms"], f["latency_p95_ms"], l["latency_p95_ms"] / f["latency_p95_ms"]))
    out.append("| 吞吐 batch=1 | %.0f QPS | %.0f QPS | %.2f× |" % (
        l["qps_batch1"], f["qps_batch1"], f["qps_batch1"] / l["qps_batch1"]))
    if "qps_batch32" in f:
        out.append("| 吞吐 batch=32 | — | %.0f QPS | — |" % f["qps_batch32"])
    out.append("| 推理峰值显存 | %.0f MB | %.0f MB | %.2f× |" % (
        l.get("peak_mem_mb", 0), f["peak_mem_mb_batch32"],
        l.get("peak_mem_mb", 1) / f["peak_mem_mb_batch32"]))
    return "\n".join(out)


md = """# Laya vs FinBERT 中文金融情感基线对比

> RTX 4090 24GB · 同一批数据、同一套指标口径 · 严格口径 = 剔除与训练集 state 文本重叠的样本

## 一、主结果：准确率与 Macro-F1

Macro-F1 是应对类别不均衡的主指标（FinFE 里 positive 类占 44%，negative/neutral 各约 28%）；
`neg_subject` 的 Macro-F1 按「测试集中有样本的类别」计算（Laya 14/21 类，FinBERT 13/19 类）。

{acc}

**读法**：
- FinFE、CFSC-ABSA 两个情感任务上，**FinBERT 略胜 Laya 1–2 个点**。FinBERT 的 3 层分类头 +
  全参数微调在「纯文本 → 极性」这类判别任务上是很强的基线，而 Laya 的 322M 主干还要同时
  兼容多种决策原语，被摊薄了。
- FinNSP 两个任务上 **Laya 更好**，尤其 `neg_subject`（+1.54 点），因为它能看到 21 个候选
  实体的完整题面，而 FinBERT 只能从 19 个标签里 argmax。

## 二、概率可信度：ECE / NLL / Brier / AURC

这组指标决定**能不能把置信度当门控用**。ECE 按 15 个等宽 bin 计算 top-1 置信度校准误差。

{calib}

**FinBERT 的校准明显更好**（ECE 低 2–5 倍），原因是它就是标准 softmax 交叉熵训练，
概率语义天然吻合；而 Laya 的 `noul` 头输出的是「true 的概率」，
其 `answer_confidence` 经过 `|2p-1|` 变换，与 argmax 语义不完全对齐，
导致 `neg_has` 的 ECE 偏高（0.0506 vs 0.0311）。

> 注：Laya 训练时已自动拟合了温度（choice 1.4378 / noul 1.4585），
> 这已经是校准后的结果。FinBERT 这里**没有**做温度缩放——即原生 softmax 就很准。

## 三、选择性准确率（能否自动处理）

按置信度排序，取前 50% / 80% / 90% 的样本，其准确率就是「自动放行这段流量时的精度」。

{sel}

两个模型在 @80 上都能达到 95%+，**都具备自动处理能力**。但注意 FinBERT 的 @50 略高而 @90
略低——它的置信度分布更集中（平均置信度 0.95+），排序粒度不如 Laya 细。

## 四、各类别召回率（类别不均衡）

{recall}

**关键结论**：两个模型都在 `neutral` 类上最弱（FinFE neutral 召回 0.696 / 0.712，
CFSC neutral 召回 0.797 / 0.759），这与直觉一致——中性情感边界模糊。
`positive` 类召回都在 0.95+，因为样本量最大。

## 五、部署价值：体积 / 延迟 / 吞吐 / 显存

{deploy}

**Laya 的部署成本约为 FinBERT 的 3 倍**：参数量 3.15×、权重 3.15×、p50 延迟 3.13×、
峰值显存 3.08×。原因是 Laya 主干是 mmBERT-base（322M）+ hidden 1024 的决策头，
而 FinBERT 是 bert-base 量级（102M）。

**体积差距主要是精度造成的，不是参数量**：Laya 的 `model.safetensors` 是 **F16**
（170 个张量全是 F16），FinBERT 存的是 fp32。把 FinBERT 转 F16 实测降到 **195.1 MB**，
此时差距是 6.3×（1228 / 195）。若 Laya 也能压到更低位宽（INT8 量化），差距可收窄到 3×
左右，但会牺牲它本来就偏弱的校准。

**延迟差距是真实的**：Laya 单条 18.56ms 里包含 Router 的路由、题面编码、多个决策头的
前向（choice + noul + action head），而 FinBERT 只有一次 BERT 前向 + 一层线性。
这是「多原语统一决策」的固有代价。

## 六、结论与选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 纯情感极性分类（高 QPS） | **FinBERT** | 精度不输、延迟 1/3、体积 1/3、校准最好 |
| 需要多任务统一决策 + 弃权 | **Laya** | 一套权重答所有题；原生支持弃权阈值与 RLCD |
| 主体识别 / 结构化抽取 | **Laya** | 题面可带完整候选集，21 选 1 更自然 |
| 需要严格校准的概率输出 | **FinBERT** | ECE 0.03 vs Laya 0.05–0.20 |

**最重要的一条**：Laya 的 `neg_subject` ECE 高达 **0.1984**，因为它的 `answer_confidence`
是 `|2p-1|` 形式的「决策边际」而非概率，不能直接当概率用。若要上线做门控，
必须用 `rl_agent_config.json` 里拟合好的分桶阈值
（`choice:3-5`→0.6217、`choice:11+`→0.7562），或另做概率校准。

---

数据来源：`reports/ft_*_metrics.json`（Laya）、`reports/finbert_all.json`（FinBERT）、
`reports/deploy_metrics.json`（部署）。脚本：`laya_metrics.py`、`finbert_baseline.py`、
`fb_eval.py`、`deploy_metrics.py`。
""".format(acc=table_acc(), calib=table_calib(), sel=table_selective(),
           recall=table_class_recall(), deploy=table_deploy())

outp = os.path.join(os.path.dirname(R), "COMPARE_Laya_vs_FinBERT.md")
with open(outp, "w", encoding="utf-8") as f:
    f.write(md)
print(md)
print("\n-> %s" % outp)
