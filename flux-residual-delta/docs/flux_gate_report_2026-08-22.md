# FLUX.1-dev arm: G0 / G1 gate results

日期:2026-08-22 · FLUX.1-dev(12B,19 dual + 38 single = 57 blocks)· 512×512,
30 step,guidance 3.5,bf16 · 单 A40 · exact = 10.58 s/图
prompt 用的是 PixArt arm 的 **同一份 test24**(24 条,主体不相交),所以两个骨干可直接比较。

## 为什么不用原来的脚手架

`src/flux_residual_delta/runtime.py` 那套设计(缓存 38 个 single block 里的一段、
用 residual MSE 做 gate)是在 PixArt arm 出结论之前写的,两个前提都已被推翻:

- **粒度**:PixArt 的 K 扫描证明只有 K=1(整栈一段、每步注入一次)部署可用,
  逐块注入会把误差从"相加"变成沿深度"相乘",图直接崩。
- **判据**:residual MSE 会把排序**搞反**(同数据同架构只换 loss,它 9/9 全错)。

所以这一轮直接从 K=1 起步,判据只用图像。新代码在
[`whole_stack.py`](../src/flux_residual_delta/whole_stack.py)。

## 实现:FLUX 的整栈缓存可以零开销

把两个 block list 都置空之后,forward 里 concat/slice 的算术恰好是恒等的:

```
h = x_embedder(latent)
(无 dual block)          -> hidden = h, encoder = e
hidden = cat([e, h])
(无 single block)
hidden = hidden[:, e_len:]  -> 正好是 h
norm_out(hidden, temb) -> proj_out
```

所以复用步 = 「置空两个 list + 在 norm_out 前加上缓存残差」,**完全不付 per-block
的 Python 派发开销**（这一项在 PixArt 上值 10.5%）。

## G0:通过,而且比 PixArt 更强

| | block stack 占端到端 | 全缓存上限 |
|---|---:|---:|
| PixArt-Σ(0.6B) | 91.6% | 11.96× |
| **FLUX.1-dev(12B)** | **95.7%** | **22.83×** |

## G1:纯缓存前沿(24 prompt)

| 变体 | 加速 | SSIM | LPIPS | PSNR |
|---|---:|---:|---:|---:|
| fixed i2 | 1.984× | 0.679 | 0.308 | 17.84 |
| fixed i3 | 2.951× | 0.595 | 0.403 | 15.87 |
| fixed i4 | 3.664× | 0.559 | 0.464 | 14.95 |
| fixed i5 | 4.832× | 0.516 | 0.521 | 14.37 |
| all_cache | 22.828× | 0.239 | 0.760 | 12.23 |

等速与 PixArt 的纯缓存前沿比:1.98× 处打平(0.679 vs 0.679),2.95× 处 **+0.072**,
3.66× 处 **+0.093**;FLUX 还能在 4.83× 处守住 0.516,而 PixArt 的前沿根本到不了这个速度。

**注意一个混杂因素**:本表 FLUX 跑 30 step,PixArt 是 20 step。step 越多、每步位移越小、
缓存残差越可复用,这对 FLUX 有利,可能解释掉相当一部分差距。20-step 对照见下节。

## per-prompt 方差很大

SSIM 的 prompt 间标准差是 0.11~0.14,标准误约 0.026。所以小于 ~0.05 的差距不要当结论;
上面 +0.072 / +0.093 约为 2.8~3.6 个标准误。单条 prompt 完全不可信:第 0 条 prompt 在
i5 上是 0.251,第 12 条是 0.632 —— 我一开始只看了第 0 条,得出了"FLUX 更差"的错误判断。

## 20-step 对照:step 数不是解释

同样 24 条 prompt,把 FLUX 也跑成 20 step(和 PixArt 一致):

| 变体 | 加速 | SSIM(20 step) | SSIM(30 step) | PixArt 等速 | 差 |
|---|---:|---:|---:|---:|---:|
| fixed i2 | 1.981× | 0.672 | 0.679 | 0.679 | −0.007 |
| fixed i3 | 2.800× | 0.599 | 0.595 | 0.543 | **+0.055** |
| fixed i4 | 3.868× | 0.545 | 0.559 | 超出前沿范围 | — |
| fixed i5 | 4.778× | 0.496 | 0.516 | 超出前沿范围 | — |

20-step 与 30-step 的前沿在所有点上相差 ≤0.02,都在 0.028 的标准误之内。
**所以 step 数混杂不成立,FLUX 的优势是真的。** 全缓存上限在 20 step 下是 15.46×
(30 step 是 22.83×,step 越少那唯一一次真算占比越大,符合预期)。

结论的形状:**~2× 处两个骨干等价;FLUX 的前沿衰减更慢,所以越激进越占优**,
并且能到 PixArt 根本到不了的速度(3.87× 处 0.545,比 PixArt 在更慢的 3.65× 处的
0.466 还好)。

**仍然要保留的口径**:这是**整条 pipeline** 的比较,不只是架构 —— FLUX 是
flow matching + 蒸馏 guidance(无 CFG),PixArt 是 DPM-solver + CFG 4.5。
而且 SSIM 都是相对各自的 exact,若 FLUX 的输出本身更平滑,SSIM 会更宽容,
这一项无法从现有数据排除。

## 下一步

1. 两个 gate 都过了,而且比 PixArt 更强,所以**在 FLUX 上训 C 有充分依据**。
   直接照搬 PixArt 已验证的配置开局:K=1、per-slot 尺度无关 loss、
   注入强度在 val 上选、判据只看图像、多 seed 复现。
2. **采集成本是主要障碍**:FLUX exact 是 10.58 s/图(30 step)或 7.08 s/图(20 step),
   而采集要在每个复用 slot 上额外跑一次真实整栈。PixArt 上采 5376 slot 要 6 分钟,
   FLUX 上是几小时量级。建议先用小数据集(如 train32 × 2 seed)验证 C 学不学得出来,
   再决定要不要投全量。
3. 训 C 之前先把 `evaluate_variants` 级别的 val/test 划分补齐(现在只用了 test24);
   注入强度必须在独立的 val prompt 上选,不能在 test24 上选。
