# Chinese financial sentiment datasets (CFSC-ABSA / FinFE / FinNSP)

Downloaded 2026-10-26. 8 files in total, 24,420,089 bytes (about 24.42 MB).

## 1. Layout

```
datasets/
├── CFSC-ABSA/
│   ├── train.txt        14,900,480 B (14.90 MB)  33,184 rows
│   └── test.txt          3,724,954 B ( 3.72 MB)   8,312 rows
├── FinFE/
│   ├── train_list.json   1,085,511 B ( 1.09 MB)  16,157 rows
│   ├── eval_list.json      135,818 B ( 0.14 MB)   2,020 rows
│   └── problem_list.json   173,232 B ( 0.17 MB)   2,020 rows (unlabeled test input)
└── FinNSP/
    ├── train_list.json   3,481,746 B ( 3.48 MB)   4,000 rows
    ├── eval_list.json      458,944 B ( 0.46 MB)     500 rows
    └── problem_list.json   459,404 B ( 0.46 MB)     500 rows (unlabeled test input)
```

## 2. Sources

| Dataset | Repository | Original path |
| --------- | ------------------------------------------------------------------------ | ------------------------- |
| CFSC-ABSA | <https://github.com/Ya-dongLi/CFSC> | `CFSC-ABSA/json/` |
| FinFE | <https://github.com/supersymmetry-technologies/BBT-FinCUGE-Applications> | `FinCUGE_Publish/finfe/` |
| FinNSP | Same as above | `FinCUGE_Publish/finnsp/` |

Note: direct access to `raw.githubusercontent.com` is blocked from this machine, so the files were actually fetched through the mirror `https://ghproxy.net/https://raw.githubusercontent.com/...`. Byte counts match the sizes returned by the repository API exactly.

## 3. Data format

### CFSC-ABSA (ABSA, aspect-based sentiment analysis)

`train.txt` / `test.txt` have a `.txt` extension but the contents are a **standard JSON array** (UTF-8), so read them with `json.load`.

```json
[{"sentence": "...", "Level 1 aspect": "Economy", "term": "<aspect term>",
  "polarity": "negative", "from": 5, "to": 9}]
```

- `sentence` is the text; `Level 1 aspect` ∈ {Corporate, Stock, Market, Economy}
- `term` is the aspect term; `from`/`to` are its start and end offsets in the sentence; `polarity` ∈ {positive, neutral, negative}
- Label distribution: training positive 20,385 / neutral 4,228 / negative 8,571;
  test positive 5,134 / neutral 1,032 / negative 2,146
- Train + test = 41,496 rows, matching the paper's protocol (one aspect term counts as one sample)

### FinFE (financial social-media sentiment, 3-way classification)

- `train_list.json`, `eval_list.json`: `[["text", label], ...]`, where label `0`=negative, `1`=neutral, `2`=positive
- Label distribution: train 0:4,690 / 1:4,143 / 2:7,324; eval 0:574 / 1:560 / 2:886
- `problem_list.json`: `[{"id": 0, "input": "text"}, ...]`, input only, for prediction/submission

### FinNSP (identifying the subject of negative news)

- `train_list.json`, `eval_list.json`: each row is a 6-element list, typically
  `[id, unused/empty field, body, entity1, entity2, label]` (label looks like "yes, &lt;entity name&gt;" / "no")
- `problem_list.json`: `[{"id": 0, "input": [body, entity...]}, ...]`, input only
- Counts: train 4,000 / eval 500 / test 500

> Note: the BBT-FinCUGE README states FinFE 8000/1000/1000 and FinNSP 4800/600/600,
> but the files actually published in the repository have the counts above (FinFE 16,157/2,020, FinNSP 4,000/500/500).
> The actual files are authoritative.

## 4. SHA256

```
1e86047e28fe0c542da149cebbcc63f087eed0e3d6acf4435ca56ea8e238ebb0  CFSC-ABSA/train.txt
bb4e9424b552fb723f40a916eae63753b24730228b3f16d721b04d829d6b240b  CFSC-ABSA/test.txt
213b308d7e648bbc57ab57ee7d868d981939ce881f8027b215659e5ff34515bd  FinFE/train_list.json
80cdffaefba9696ecdf1fcb1cccf5052779c33e976195c19502c2130d7b454b6  FinFE/eval_list.json
f5b8bc10b49286a6b28d16fe24ccc30e6e46bfa90b0e5575033cfed0a0944edd  FinFE/problem_list.json
8e8e84ae7ced4ba5b636a86f9332747d3c93df61c18c1c7f71ddb1d28547460d  FinNSP/train_list.json
722b4b942fcd5ff2b24a1ab8e0b1e8c1c8905e2dc13a07179293757f4cce92b7  FinNSP/eval_list.json
40aa03e60e61363ff0f8314fb871ebb67a594c82f406e8317ee17b0721575e9b  FinNSP/problem_list.json
```

## 5. Citation

- CFSC: Li Y. et al. *CFSC* (Chinese Financial Sentiment Corpus), <https://github.com/Ya-dongLi/CFSC>
- FinFE / FinNSP: Lu D. et al. *BBT-Fin: Comprehensive Construction of Chinese Financial Domain
  Pre-trained Language Model, Corpus and Benchmark*, arXiv:2302.09432 (the CFLEB benchmark)
