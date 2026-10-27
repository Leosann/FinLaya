# 中文金融情感分析数据集（CFSC-ABSA / FinFE / FinNSP）

下载时间：2026-10-26　总计 8 个文件，24,420,089 字节（约 24.42 MB）

## 一、目录结构

```
datasets/
├── CFSC-ABSA/
│   ├── train.txt        14,900,480 B (14.90 MB)  33,184 条
│   └── test.txt          3,724,954 B ( 3.72 MB)   8,312 条
├── FinFE/
│   ├── train_list.json   1,085,511 B ( 1.09 MB)  16,157 条
│   ├── eval_list.json      135,818 B ( 0.14 MB)   2,020 条
│   └── problem_list.json   173,232 B ( 0.17 MB)   2,020 条（无标签，测试输入）
└── FinNSP/
    ├── train_list.json   3,481,746 B ( 3.48 MB)   4,000 条
    ├── eval_list.json      458,944 B ( 0.46 MB)     500 条
    └── problem_list.json   459,404 B ( 0.46 MB)     500 条（无标签，测试输入）
```

## 二、来源

| 数据集       | 仓库                                                                       | 原始路径                      |
| --------- | ------------------------------------------------------------------------ | ------------------------- |
| CFSC-ABSA | <https://github.com/Ya-dongLi/CFSC>                                      | `CFSC-ABSA/json格式/`       |
| FinFE     | <https://github.com/supersymmetry-technologies/BBT-FinCUGE-Applications> | `FinCUGE_Publish/finfe/`  |
| FinNSP    | 同上                                                                       | `FinCUGE_Publish/finnsp/` |

说明：本机到 `raw.githubusercontent.com` 直连被阻断，实际经镜像 `https://ghproxy.net/https://raw.githubusercontent.com/...` 下载，文件字节数与仓库 API 返回的 size 完全一致。

## 三、数据格式

### CFSC-ABSA（ABSA，方面级情感分析）

`train.txt` / `test.txt` 虽为 .txt 后缀，内容是**标准 JSON 数组**（UTF-8），需 `json.load` 读取。

```json
[{"sentence": "...", "Level 1 aspect": "Economy", "term": "印度卢比",
  "polarity": "negative", "from": 5, "to": 9}]
```

- `sentence` 语句文本；`Level 1 aspect` ∈ {Corporate, Stock, Market, Economy}
- `term` 方面词；`from`/`to` 方面词在句中的起止位置；`polarity` ∈ {positive, neutral, negative}
- 标签分布：训练集 positive 20,385 / neutral 4,228 / negative 8,571；  
  测试集 positive 5,134 / neutral 1,032 / negative 2,146
- 训练+测试 = 41,496 条，与论文口径一致（一个方面词算一条样本）

### FinFE（金融社交媒体情绪分类，三分类）

- `train_list.json`、`eval_list.json`：`[["文本", 标签], ...]`，标签 `0`=消极、`1`=中性、`2`=积极
- 标签分布：train 0:4,690 / 1:4,143 / 2:7,324；eval 0:574 / 1:560 / 2:886
- `problem_list.json`：`[{"id": 0, "input": "文本"}, ...]`，仅输入，供预测/提交用

### FinNSP（负面消息主体判定）

- `train_list.json`、`eval_list.json`：每条为 6 元素列表，典型结构  
  `[id, 无关/空字段, 正文, 实体1, 实体2, 标签]`（标签形如「是，实体名」/「否」）
- `problem_list.json`：`[{"id": 0, "input": [正文, 实体...]}, ...]`，仅输入
- 数量：train 4,000 / eval 500 / test 500

> 注意：BBT-FinCUGE 的 README 写的是 FinFE 8000/1000/1000、FinNSP 4800/600/600，  
> 但仓库里实际发布的文件是上表数量（FinFE 16,157/2,020、FinNSP 4,000/500/500）。  
> 以实际文件为准。

## 四、SHA256

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

## 五、引用

- CFSC：Li Y. et al. *CFSC* (Chinese Financial Sentiment Corpus)，<https://github.com/Ya-dongLi/CFSC>
- FinFE / FinNSP：Lu D. et al. *BBT-Fin: Comprehensive Construction of Chinese Financial Domain  
  Pre-trained Language Model, Corpus and Benchmark*, arXiv:2302.09432（CFLEB 基准）
