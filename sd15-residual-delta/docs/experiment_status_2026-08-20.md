# SD1.5 Residual-Delta Reuse：实验方案与阶段结果

更新时间：2026-08-21

## 1. 研究问题

目标是在 Stable Diffusion 1.5 上验证以下假设：与直接复用旧 attention
residual 相比，一个以当前状态变化为条件的 surrogate 能否在保持实际加速的同时，
更准确地重建被跳过模块的当前 residual，并最终改善生成图像。

对第 `m` 个 `Transformer2DModel`，定义：

```text
R_t = G_m(h_t) - h_t
delta_h = h_t - h_anchor
delta_R = R_t - R_anchor
delta_R_hat = C(delta_h, R_anchor, timestep, horizon, module_id)
h_out_hat = h_t + R_anchor + delta_R_hat
```

这里保留 SD1.5 U-Net 的卷积路径，仅替代 16 个输入输出同形状的
`Transformer2DModel`。因此 surrogate 能使用当前去噪步真实产生的 `h_t`，而不是
只根据旧 cache 外推。

## 2. 为什么从 FLUX 改为 SD1.5

FLUX 对显存、下载、数据采集和多轮消融的成本过高，不适合先验证机制。SD1.5
可以在单张 A40 上快速完成 512×512、30 步的端到端实验，同时仍然包含多个跨尺度
attention 模块，足以验证 residual-delta 方法是否成立。

本阶段只回答机制问题，不声称 SD1.5 上的绝对速度可直接外推到 FLUX。

## 3. 实验对象与运行设置

- 基础模型：Stable Diffusion 1.5 FP16。
- 目标模块：U-Net 内全部 16 个 `Transformer2DModel`。
- 通道宽度：320、640、1280；每种宽度共享一个 surrogate，并使用 module ID 条件。
- 推理：512×512，30 denoising steps。
- 复用调度：10 个 full-compute anchor steps，20 个 reuse steps。
- 基线：
  - Exact：所有模块每步精确计算。
  - Fixed cache：reuse step 直接使用 anchor residual。
  - Learned：使用 residual-delta surrogate 修正 cache。
- 图像指标：相对 exact 同 prompt、同 seed 输出计算 MSE、PSNR、SSIM。
- 性能指标：端到端 wall-clock speedup、峰值显存、精确/复用/surrogate 调用次数。

## 4. 数据方案

### 4.1 Exact-trajectory 数据

在完整精确轨迹上采集 `(h_anchor, R_anchor, h_t, R_t)`。已完成：

- `pilot`：8 prompts，65 个文件，约 4.44 GB。
- `pilot20`：20 prompts，161 个文件，约 11.09 GB。
- prompt-disjoint split，避免同一 prompt 同时进入训练和测试。

该数据适合衡量单步 residual 预测，但训练输入来自精确轨迹，和实际复用轨迹存在
distribution shift。

### 4.2 On-policy 数据

沿 fixed-cache 实际复用轨迹运行；在选定 reuse steps 额外计算 oracle residual 作为
监督，但 pipeline 仍返回 fixed-cache 结果继续去噪。每个 prompt 采集：

- 4 个 reuse steps：1、5、13、23。
- 每步 16 个模块，共 64 items/prompt。
- 20 prompts，总计约 1280 个监督样本。
- horizon 覆盖 1 和 2。

该数据直接解决 exact-trajectory 与实际复用状态不一致的问题，但监督仍是局部的
单模块 residual MSE，而不是最终图像或后续轨迹损失。

## 5. 已完成实验

### 5.1 Exact-trajectory surrogate 消融

| 实验 | 测试 residual MSE 相对 fixed cache 的改善 | 结论 |
|---|---:|---|
| 8-prompt，含 `delta_h` | 8.62% | 可学习，但幅度有限 |
| 8-prompt，不含 `delta_h` | 5.45% | `delta_h` 有正贡献 |
| 全局 1280-capacity | 8.37% | 单纯增大容量无明显收益 |
| 20-prompt，含 `delta_h` | 9.75% | 更多 prompts 有小幅收益 |
| 共同未见 prompt，旧模型 | 6.31% | 公平对照 |
| 共同未见 prompt，新模型 | 7.48% | 数据扩充后略好 |

结论：当前状态条件是有效信号，但 exact-trajectory 的离线收益尚不足以保证 rollout
收益。

### 5.2 Exact-trajectory 模型的端到端 rollout

未见 prompt：`a bowl of colorful tropical fruit on a marble kitchen counter`，seed 1244。

| 方法 | Speedup | MSE vs exact | PSNR | SSIM | MSE 相对 fixed |
|---|---:|---:|---:|---:|---:|
| Fixed cache | 1.38× | 0.017307 | 17.6177 | 0.6597 | 0.00% |
| Learned，全模块 | 1.03× | 0.017965 | 17.4558 | 0.6450 | -3.80% |
| Learned，仅 320/640 | 1.22× | 约 0.01722 | 17.6393 | 0.6564 | +0.50% |

选择性修正只有 0.50% 的 MSE 改善，且 SSIM 略降；不足以证明端到端有效。

### 5.3 On-policy surrogate 离线结果

训练 2000 steps 后，在 prompt-disjoint 测试集上：

- learned relative MSE：0.7884。
- 相对 fixed cache 改善：21.16%，通过预设 20% 离线门槛。
- cosine similarity：0.4405。
- horizon 1 / 2 改善：23.08% / 19.23%。
- 320 / 640 / 1280 通道改善：28.46% / 22.03% / 14.35%。
- 64² / 32² / 16² / 8² 分辨率改善：28.46% / 22.03% / 16.24% / 4.94%。

收益随低分辨率、深层和 1280 通道模块明显衰减，因此同时测试了全模块修正和仅
320/640 修正。

### 5.4 On-policy 模型的端到端 rollout

使用与 5.2 完全相同的 prompt、seed 和调度：

| 方法 | Speedup | MSE vs exact | PSNR | SSIM | MSE 相对 fixed |
|---|---:|---:|---:|---:|---:|
| Fixed cache | 1.379× | 0.017307 | 17.6177 | 0.6597 | 0.00% |
| Learned，全模块 | 1.067× | 0.017859 | 17.4815 | 0.6454 | -3.19% |
| Learned，仅 320/640 | 1.233× | 0.017786 | 17.4992 | 0.6450 | -2.77% |

峰值显存约 2.71 GiB。全模块 corrector 调用 320 次；选择性版本调用 200 次。

### 5.5 训练集外 prompt × 多 seed 稳定性测试

为避免单个 prompt/seed 的偶然性，额外使用 4 个完全未参与特征采集和训练的 prompts，
每个 prompt 运行 seeds 2101、2102、2103，共 12 个案例。每个案例都使用相同 latent
分别运行 exact、fixed、learned-all 和 learned-320/640。

| 方法 | Speedup | 平均 MSE | 平均 SSIM | 逐案例 MSE 改善均值 | MSE 胜率 | SSIM 胜率 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed cache | 1.409× | 0.016519 | 0.68084 | 0.00% | — | — |
| Learned，全模块 | 1.182× | 0.015755 | 0.68631 | +3.53% ± 14.41% | 5/12 | 6/12 |
| Learned，仅 320/640 | 1.257× | 0.016344 | 0.68116 | +1.42% ± 11.72% | 5/12 | 5/12 |

另一种聚合方式是先对 MSE 求均值，再计算比值。按该口径，全模块相对 fixed 改善
4.62%，选择性版本改善 1.06%。两种口径都没有通过预设 5% 图像 MSE 门槛。

单案例范围也很大：全模块从恶化 16.17% 到改善 27.18%，选择性版本从恶化
13.26% 到改善 27.67%。因此平均值中存在真实的正信号，但当前 predictor 缺少稳定性
和可识别的适用条件，不能把最好案例当作方法结论。

### 5.6 静态与动态 confidence 分析

先用 on-policy validation split 选择 `(module_id, horizon)` 静态 gate，再在 test split
评估。validation 上 32/32 个组合都优于 fixed cache，因此最优策略是 100% 调用：

| validation 改善门槛 | 选择组合 | test residual 改善 |
|---:|---:|---:|
| 0% / 5% | 32/32 | 21.16% |
| 10% | 30/32 | 20.85% |
| 20% | 24/32 | 18.74% |

静态分组无法解释图像级的正负案例。随后重放 12 条 learned trajectory，记录无需
oracle 的输入漂移与预测 norm。最强 Spearman 相关是 `input_drift / anchor_hidden`
中位数与图像 MSE 改善的 -0.434；`prediction / anchor_residual` 中位数的 Pearson
为 -0.469，但 Spearman 仅 -0.238。样本只有 12 个且检查了多个特征，这些相关不足以
支持可靠的 norm threshold gate。

### 5.7 两步 rollout loss

实现了可微的两步短展开训练：从 exact anchor latent 开始，anchor U-Net 精确运行，
接下来两个 denoising steps 使用 surrogate；每一步 scheduler 输出都直接匹配 exact
teacher latent。SD1.5 全冻结，只训练 surrogate。

梯度 smoke test：

- rollout loss：0.000830。
- gradient norm：0.001305，非零且有限。
- 一次优化：0.525 秒。
- 峰值显存：3.108 GiB。

随后只使用原始 train split 的 16 个 prompts，从 on-policy `best.pt` 初始化。200-step
运行的 teacher trajectory 构建耗时 21.27 秒，训练耗时 44.02 秒，峰值显存 3.114
GiB。用同一随机序列重建 step-50 与 step-100 checkpoint，形成严格早停曲线：

| Rollout steps | 全模块：MSE 改善均值 ± std / 胜率 | 选择性 320/640：MSE 改善均值 ± std / 胜率 |
|---:|---:|---:|
| 0 | +3.53% ± 14.41% / 5/12 | +1.42% ± 11.72% / 5/12 |
| 50 | +2.55% ± 11.10% / 6/12 | **+1.90% ± 8.64% / 7/12** |
| 100 | +1.95% ± 10.18% / 6/12 | +0.66% ± 6.93% / 7/12 |
| 200 | +1.20% ± 9.60% / 6/12 | +0.26% ± 6.50% / 5/12 |

rollout loss 会单调压低方差，但训练过久也会收缩有益修正。当前最合理的 operating
point 是 step-50 selective：1.241× speedup、MSE 胜率 7/12、SSIM 胜率 7/12；但
平均 MSE 改善只有 1.90%，仍未通过 5% 目标。

由于 step-50 是通过这 4 个 prompts 的结果选择的，原 `rollout_holdout4` 从现在起只
能视为 early-stopping validation，不能再作为最终无偏测试集。

### 5.8 独立 final holdout

为检验上述早停选择是否泛化，新增 4 个从未参与特征采集、训练或 early stopping 的
prompts，并使用 seeds 4101、4102、4103，共 12 个最终案例。固定缓存的平均 MSE 为
0.014657，平均 SSIM 为 0.71620。下表同时报告逐案例改善的均值和“先聚合 MSE、再算
比值”的总体改善；正值表示优于 fixed cache。

| Checkpoint / 方法 | 平均 MSE | 平均 SSIM | 逐案例 MSE 改善均值 ± std | 最差案例 | MSE / SSIM 胜率 | 总体 MSE 改善 |
|---|---:|---:|---:|---:|---:|---:|
| On-policy step0，全模块 | 0.016451 | 0.69745 | -5.69% ± 26.73% | -65.00% | 6/12 / 6/12 | -12.24% |
| On-policy step0，仅 320/640 | 0.016942 | 0.69670 | -10.25% ± 26.85% | -85.55% | 4/12 / 2/12 | -15.59% |
| Rollout step50，全模块 | 0.017190 | 0.69887 | -8.40% ± 29.18% | -73.30% | 6/12 / 4/12 | -17.28% |
| Rollout step50，仅 320/640 | 0.016696 | 0.69880 | -12.24% ± 29.13% | -96.13% | 4/12 / 3/12 | -13.91% |

step50 的干净计时为：fixed 1.395×、全模块 1.203×、选择性版本 1.251×。step0 的质量
指标是有效的，但该轮计时与后台 `/workspace` 写入重叠，因此不用于速度比较。

独立 final holdout 推翻了 early-stopping validation 上的小幅正均值。step50 selective
虽然按总体 MSE 口径比 step0 selective 好 1.68 个百分点，但逐案例均值更差、胜率不变，
最差案例从 -85.55% 扩大到 -96.13%，而且两者都显著劣于 fixed cache。两步 rollout
loss 因此没有解决跨 prompt/seed 的泛化和尾部风险。

### 5.9 多 seed + 尾部 rollout loss

针对上一轮只使用单个 trajectory seed 的问题，实现了更宽的 rollout 训练分布：16 个
train prompts × 4 个 latent seeds × 10 个 anchor 位置。每个 optimizer step 同时采样
4 个片段，目标为片段平均损失加最差一半片段的平均损失（权重 0.5）。一次连续
100-step 训练同时保存 step25、step50、step100，避免分开重跑引入随机差异。

- 64 条 exact teacher trajectories：76.27 秒。
- 100-step 训练：83.35 秒。
- 峰值显存：7.37 GiB。
- gradient norm 范围：0.000366–0.006714，全程有限非零。
- 前 25 / 后 25 步平均 loss：0.000709 / 0.000744；训练 loss 不单调，因此不用于
  checkpoint 选择。

在原 early-stopping validation（4 prompts × 3 seeds）上得到：

| Checkpoint / 方法 | Speedup | 逐案例 MSE 改善均值 ± std | 最差案例 | MSE / SSIM 胜率 | 总体 MSE 改善 |
|---|---:|---:|---:|---:|---:|
| Step25，全模块 | 1.119× | +3.63% ± 12.35% | -11.49% | 5/12 / 6/12 | +4.34% |
| Step25，仅 320/640 | 1.224× | +1.98% ± 10.09% | -10.74% | 5/12 / 6/12 | +1.20% |
| Step50，全模块 | 1.121× | +3.14% ± 10.72% | -11.94% | 6/12 / 7/12 | +4.15% |
| Step50，仅 320/640 | 1.238× | +1.82% ± 8.45% | -11.72% | 6/12 / 7/12 | +1.37% |
| Step100，全模块 | 1.095× | +3.46% ± 10.39% | -13.26% | 7/12 / 8/12 | +4.81% |
| Step100，仅 320/640 | 1.217× | +1.36% ± 7.26% | **-9.55%** | 5/12 / 7/12 | +1.49% |

尾部训练将 selective 的标准差和最差案例逐步压低，但同时继续收缩平均正修正。按预先
声明的三项门槛（speedup ≥ 1.20×、总体 MSE 改善 ≥ 5%、平均 SSIM 不下降），没有
任何 checkpoint/variant 合格，机器可读 gate 报告的选择结果为 `null`。因此本轮不进入
final holdout。5.8 使用的 final holdout 已经影响本轮设计，也不能再次作为无偏测试集。

### 5.10 下游敏感性与 horizon-aware gate

为避免继续等权优化 residual error，新增单点 oracle refresh：在 fixed-cache rollout 的某
个 reuse step、某个 attention module 上只执行一次精确计算，其他位置保持 fixed cache，
直接测量最终图像 MSE 的变化。两个设计集 prompt/seed 分别测试 step 1/2、13/14、
28/29 × 16 modules，每个 case 96 次干预。

| Case | 固定缓存 MSE | 干预改善均值 ± std | 范围 | 正收益率 |
|---|---:|---:|---:|---:|
| Case 1 | 0.022670 | -0.178% ± 0.525% | -2.699% 至 +0.954% | 58.3% |
| Case 2 | 0.025591 | +0.860% ± 3.777% | -11.824% 至 +21.375% | 76.0% |

同一个 `(step,module)` 在两 case 间的 Pearson 为 -0.136、Spearman 为 -0.186，符号
一致率 55.2%。聚合成 32 个 `(module,horizon)` 后，Pearson 0.105、Spearman 0.120，
符号一致率仅 31.3%。因此下游敏感性高度依赖 prompt/seed 和当前 trajectory，不能用一张
全局静态权重表表示。更重要的是，单点 oracle 精确刷新也可能显著恶化最终图像，说明
fixed-cache rollout 中不同模块的近似误差存在相互补偿，边际干预不是单调的。

两个 case 仍有 5 个分组同时为正：module 7 的 horizon 1/2、module 8 的 horizon 2、
module 9 的 horizon 1/2。用 multi-seed step100 checkpoint 只修正这 5/32 个组合，并在
独立 validation 上测试：

| 方法 | Surrogate calls | Speedup | 逐案例 MSE 改善均值 ± std | 最差案例 | MSE / SSIM 胜率 | 总体 MSE 改善 |
|---|---:|---:|---:|---:|---:|---:|
| Step100，全模块 | 320 | 1.095× | +3.46% ± 10.39% | -13.26% | 7/12 / 8/12 | +4.81% |
| Step100，仅 320/640 | 200 | 1.217× | +1.36% ± 7.26% | -9.55% | 5/12 / 7/12 | +1.49% |
| Step100，sensitivity gate | **50** | **1.367×** | +1.54% ± 6.97% | -13.36% | 7/12 / 6/12 | **+3.14%** |

sensitivity gate 明显改善成本/质量折中，并通过速度与平均 SSIM 门槛，但总体 MSE 改善
3.14%，仍未通过 5% 门槛。其机器可读 gate 结果仍为 `eligible=false`。

## 6. 当前结论

现阶段不能声称方法成功改善 SD1.5 生成质量。

已经确认的正结果：

1. `delta_h` 提供了超出旧 cache 的可学习信息。
2. On-policy 数据把单步 residual MSE 改善从约 10% 提升到 21.16%。
3. 320/640 通道模块比 1280 通道和低分辨率模块更容易预测。
4. 在 early-stopping validation 上曾观察到全模块平均 MSE 改善 4.62%，说明局部存在
   可利用信号；但该结果没有在独立 final holdout 上复现。
5. 两步 rollout loss 可在 early-stopping validation 上降低方差，并维持大于 1.20× 的
   速度；这只能视为训练行为证据，不能视为泛化质量收益。
6. 多 seed + 尾部 loss 能把 selective validation 的标准差降至 7.26%、最差案例收窄至
   -9.55%，证明尾部风险可以被训练目标直接影响。
7. 下游 sensitivity gate 只需 50 次 surrogate 调用即可达到 1.367× 和 3.14% 总体 MSE
   改善，优于原 selective 的成本/质量折中。

已经确认的失败点：

1. 单步 residual MSE 的改善没有转化为最终图像改善。
2. surrogate 在每个 reuse step 连续介入后，微小的方向性误差会改变后续状态，形成
   rollout distribution shift 和累计误差。
3. 当前 loss 对模块、timestep、horizon 和下游敏感性近似等权；但不同 residual
   误差对最终图像的影响并不等价。
4. predictor 的运行成本消耗了大部分理论加速，尤其是全模块修正。
5. 多 prompt/seed 测试的 MSE 胜率只有 5/12，改善的标准差远大于均值；全模块版本
   也只有 1.182×，没有同时通过质量、胜率和速度门槛。
6. rollout early stopping 在 validation 上的最好平均改善仍仅 1.90%，而且该集合已被
   用于模型选择。
7. 全新 final holdout 上，step0 和 step50 的所有版本平均都劣于 fixed cache；step50
   selective 的最差案例达到 -96.13%，说明当前方法有不可接受的尾部风险。
8. step50 没有改善 final holdout 的逐案例均值或胜率，因此不能靠继续延长同一种
   两步 rollout 训练解决问题。
9. 多 seed + 尾部训练仍没有产生合格 operating point：全模块最好的总体 MSE 改善
   4.81% 但速度只有 1.095×；selective 速度达标但总体改善最高只有 1.49%。
10. 下游敏感性在两个 case 间几乎不相关，单点 oracle refresh 也可能恶化最终图像；
    静态 module/timestep 权重不能解决 trajectory-conditional 风险。
11. sensitivity gate 虽然通过速度和 SSIM 门槛，但总体 MSE 改善只有 3.14%，因此仍不
    进入新的 final holdout。

因此真正瓶颈已经从“corrector 容量不够”收敛为“局部监督目标与多步 rollout 目标
错配”。

## 7. 下一轮实验方案

### Phase A：确认失败是否稳定（已完成）

已完成 4 个未见 prompts × 3 seeds。结果为小幅平均正收益，但方差高、胜率低，且
全模块版本速度未达标。结论是当前方法存在信号但不稳定，尚不是图像级正结果。

### Phase B：下游敏感性加权

用少量 oracle 扰动估计各 `(module, timestep, horizon)` 的下游图像/latent 敏感性，
训练时对高影响误差加权。先保留 320/640 模块，暂不修正离线收益最低的 1280/8²
模块，以控制成本和误差注入。

### Phase C：短展开 rollout loss

从 on-policy state 开始展开 2–4 个 denoising steps，让 surrogate 的输出进入下一步，
优化后续 latent 与 exact teacher latent 的差异。建议 loss：

```text
L = lambda_local * normalized_residual_MSE
  + lambda_latent * normalized_next_latent_MSE
  + lambda_direction * (1 - cosine(delta_R_hat, delta_R))
```

先冻结 SD1.5，仅训练 surrogate；使用 gradient checkpointing 和短展开控制显存。

两步版本、独立 rollout validation 和 final holdout 均已完成。final holdout 为负，所以下
一轮不继续增加同一种 loss 的训练步数。应改为在多个 prompt × 多个 latent seed × 多个
anchor 位置上直接采样 rollout 片段，并对高分位误差或最差案例加权；checkpoint 仍只能
由独立 validation 选择，final holdout 在设计冻结前不得查看。

该多 seed/尾部版本已实现。它改善了方差和最差案例，但平均收益没有提高，因此下一步
应转向 Phase B 的下游敏感性加权，而不是继续增加 seeds、batch 或训练步数。

Phase B 的单点 oracle pilot 与静态 horizon-aware gate 也已完成。静态权重只改善成本/
质量折中，没有通过质量门槛。若继续研究，应学习以当前 trajectory feature 为条件的有符号
下游风险，并显式建模多个模块干预的非加性；在获得足够的跨 prompt/seed sensitivity 标签
前，不应再次训练或查看新的 final holdout。

### Phase D：门控与成本控制

训练一个低成本 confidence gate，只有预测收益高于误差风险时才调用 surrogate；否则
使用 fixed cache 或强制刷新。目标 operating point：

- 端到端 speedup ≥ 1.20×。
- 多 prompt/seed 平均图像 MSE 相对 fixed cache 改善 ≥ 5%。
- SSIM 不低于 fixed cache。
- predictor 额外耗时不超过 exact 与 fixed 延迟差的 50%。

只有同时通过这些门槛，才进入量化、更多 prompts 和 FLUX 迁移。

## 8. 结果目录

所有正式结果均位于 `/workspace/sd15-residual-delta`：

```text
data/features/pilot/
data/features/pilot20/
data/features/on_policy20/
runs/overfit/
runs/overfit_no_delta_h/
runs/global_1280/
runs/pilot20/
runs/comparisons/
runs/runtime_prompt10/
runs/runtime_prompt10_selective/
runs/on_policy20/
runs/runtime_on_policy_prompt10/
runs/runtime_grid_on_policy_holdout4/
runs/rollout_smoke/
runs/rollout_train50/
runs/rollout_train100/
runs/rollout_train200/
runs/runtime_grid_rollout50_holdout4/
runs/runtime_grid_rollout100_holdout4/
runs/runtime_grid_rollout200_holdout4/
runs/runtime_grid_on_policy_final_holdout4/
runs/runtime_grid_rollout50_final_holdout4/
runs/final_holdout_checkpoint_comparison.json
runs/rollout_multiseed_smoke/
runs/rollout_multiseed100/
runs/runtime_grid_rollout_multiseed25_holdout4/
runs/runtime_grid_rollout_multiseed50_holdout4/
runs/runtime_grid_rollout_multiseed100_holdout4/
runs/rollout_multiseed_validation_curve.json
runs/refresh_sensitivity_smoke/
runs/refresh_sensitivity_pilot96/
runs/refresh_sensitivity_pilot96_case2/
runs/refresh_sensitivity_case_comparison.json
runs/runtime_grid_sensitivity_gate_holdout4/
runs/sensitivity_gate_validation.json
```

`/root` 下的模型、数据和运行目录仅为快速 scratch 副本，不能作为最终结果位置。
