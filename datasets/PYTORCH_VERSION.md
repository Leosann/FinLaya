# PyTorch 版本选择（训练/开发环境）

针对本机：Windows + **GTX 1650（Turing，算力 sm_75，4GB 显存）**，驱动 466.92 → 支持最高 CUDA 11.3。

## 结论

```
torch 2.7.1+cu118   +   transformers 4.57.x   +   Python 3.10 / 3.11
```

```bash
# 新建隔离环境（3.11 最稳；本机已有 3.13 也行，见下文）
py -3.11 -m venv .venv
.venv\Scripts\activate

# 关键：必须用 cu118 轮子，cu121/cu124/cu126 在驱动 466.92 上装得上但无法使用 GPU
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# Laya 依赖（走国内源）
pip install transformers>=4.48 safetensors huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证
python -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability())"
# 期望：2.7.1+cu118 11.8 True (7, 5)
```

## 为什么是 2.7.1

| 约束 | 说明 |
| --- | --- |
| Laya 声明 `torch>=2.0.0` | 只要 2.0 以上，**下限没有约束力**，坑在别处 |
| 驱动 466.92 = CUDA 11.3 | 只能用 **cu118** 及更早的轮子；cu121/cu124/cu126 编译目标高于驱动上限，会报 CUDA driver version is insufficient |
| cu118 已停更 | 官方 cu118 索引**最后一个版本就是 2.7.1**（2.8 起只发 cu126/cu128/cu129），所以 2.7.1 是这条路上的天花板 |
| Python 支持 | 2.7.1+cu118 有 cp39/310/311/312/313 全套 Win 轮子，本机 3.13 也能装 |

实测官方索引确认：

- `https://download.pytorch.org/whl/cu118/torch/` → 最高 `2.7.1`
- `https://download.pytorch.org/whl/cu121/torch/` → 最高 `2.5.1`（且本机驱动用不了）

## transformers 版本

**必须 ≥ 4.48.0**：低于此版本不支持 ModernBERT 架构，模型加载会直接失败。Laya 的
`pyproject.toml` 也是写 `transformers>=4.48.0`。

另外 **transformers ≥ 5.0 会要求 `torch>=2.6`**，与本方案兼容；但 `option_layout="parallel"`
训练（需要 transformers>=5）只在 eager 后端可用，本机没必要开。**建议锁 4.5x 系列**：

```bash
pip install "transformers>=4.48,<5"
```

## 一个必须知道的性能陷阱（GTX 1650 专属）

ModernBERT 隐藏维度是 1024，注意力 softmax 的 `dim_size = 1024`。查 PyTorch 源码
`aten/src/ATen/native/cuda/SoftMax.cu` 的 forward 分支：

```cuda
if (dim_size <= 2048 && dim_size*sizeof(scalar_t) <= 8192) {
    dispatch_softmax_forward<...>();          // sub-warp / 128 线程 快路径（C10_WARP_SIZE=16 时每 lane 处理 2 个，
} else {                                      // dim_size=1024 恰好落在 <=2048 内）
    ... ILP / 512 线程通用路径 ...
}
```

而 `at::cuda::warp_size()` 在 sm_5x–7x 上取 **32**，此时 `dim_size(1024) <= warp_size^2(1024)`
成立的通用路径要求**每 lane 处理 ≥2 元素**，1024 元素只能给 32 lane 各 1 个 →
**回落为 32 线程的慢 kernel**。Turing（sm_75）正是最典型的受害架构。

结果：GTX 1650 上每条样本的 attention 开销比理论值差一个量级，**bf16 也救不了**
（瓶颈是线程数不是带宽）。所以：

- 别指望在这张 4GB 卡上跑完 53k 样本的训练，显存和速度都不够
- 本机只用来做 **`--dry-run` 校验数据、跑 `predict` 冒烟测试、小批量 debug**
- 真正训练放到云端 GPU（T4/A10/4090 都是 sm_80+，warp=32 但 fp16 走
  `dispatch_softmax_forward` 的 half 路径或 tensor core 分支，不受此限）

如果只有这张卡且必须本地训练，实用做法是**把 `max_len` 压到 256 以内**、micro-batch 1 +
grad-accum 拉大，并接受要比云端慢 10 倍以上。

## 备选方案

| 场景 | 选择 |
| --- | --- |
| 本机 4GB 卡，只想 debug | `torch 2.7.1+cu118` + `max_len≤256` |
| 本机没有可用 GPU，纯 CPU 校验数据 | `pip install torch --index-url https://download.pytorch.org/whl/cpu`（版本随意，2.7.1 即可） |
| 云端训练（推荐） | `torch 2.7.1+cu126` 或 `2.8.x+cu128`，配 A10/4090；T4 用 cu118 也一样 |
| 升级驱动后 | 驱动 ≥ 525 可上 cu118/cu121 全线；≥ 550 可上 cu126。**即便如此也别在本机训**，4GB 显存是硬伤 |

## 不要做的事

- ❌ `pip install torch`（默认源默认拉 cu12x 版本，且会装到全局）
- ❌ 装 `torch 2.8+` 后再找 cu118 轮子 —— 2.8 起没有 cu118 构建
- ❌ 装 `tilelang`（`laya[fast]`）—— 需要 sm_80+，GTX 1650 不支持
- ❌ 用 `laya[onnx]` 配 torch<2.3（NumPy 2 会与 torch 1.x ABI 冲突，官方已在 pyproject 注释里写明）
