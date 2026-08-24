# SD1.5 Residual-Delta Correction 最终实验报告

日期：2026-08-21  
模型：Stable Diffusion 1.5，512×512，30-step DDIM，CFG 7.5  
结论：在冻结的 extra10 缓存计划上，learned Correction 在未触碰的 12 个最终样本中同时改善 MSE、PSNR、Gaussian SSIM 和 AlexNet LPIPS，并保持平均速度大于 1.25×。

## 1. 最终方法

固定缓存 anchor steps：

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28]
```

Correction 使用当前 step 的真实 prefix feature 变化与缓存 residual：

```text
delta_h = h_t - h_anchor
delta_R_hat = C(delta_h, R_anchor, timestep, horizon)
R_hat_t = R_anchor + scale * delta_R_hat
```

最终注入位置为 UNet 第 14 个 attention 模块（最后执行的 up-block attention）、全局 denoising step 14、cache horizon 2。Surrogate 的 320-channel 分支宽度为 128、深度为 3、conditioning dim 为 128。

训练分两阶段：

1. 用 on-policy 固定缓存轨迹监督 residual delta，得到 `extra10_module14_s14/best.pt`；其独立 residual test MSE 相对改善约 68.74%。
2. 从该 checkpoint 出发，只训练 `models.320.output`，直接最小化 correction step 后 latent 相对固定缓存轨迹的误差。500 步训练中验证集最佳点在 step 50；后续明显过拟合，因此最终固定使用 step-50 `best.pt`。

设计集冻结的注入 scale 为 3.0。最终集上不再调 checkpoint、step、模块、scale 或缓存计划。

## 2. 评估协议

- Exact：不缓存，作为同 prompt、同 seed 的参考图。
- Fixed cache：extra10 anchor schedule，不启用 Correction。
- Correction：与 Fixed cache 使用完全相同的 anchor schedule，只额外启用 learned surrogate。
- 设计集：4 prompts × 2 seeds = 8 cases，用于选择 scale。
- 最终集：另 4 prompts × 3 个新 seeds（11201、11202、11203）= 12 cases。
- 指标：wall-clock speedup、RGB image MSE、PSNR、Gaussian-window skimage SSIM、AlexNet LPIPS。
- 通过门槛：平均 speedup ≥ 1.25×、Gaussian SSIM > 0.850、LPIPS < 0.125。

这里的 0.850/0.125 来自前期 OnlineCache FLUX 结果所设的数值参考线。由于模型、采样器和硬件不同，这不是对 OnlineCache 官方实现的严格同模型复现；受控结论来自 Correction 与同一 SD1.5 fixed-cache baseline 的配对比较。

## 3. 未触碰最终集结果

| Variant | Speedup ↑ | MSE ↓ | PSNR ↑ | Gaussian SSIM ↑ | AlexNet LPIPS ↓ |
|---|---:|---:|---:|---:|---:|
| Fixed cache | 1.2589× | 0.00169074 | 29.2573 | 0.905889 | 0.049502 |
| Learned Correction | **1.2729×** | **0.00163607** | **29.4290** | **0.910703** | **0.047408** |

相对 Fixed cache：

- aggregate MSE 改善 **3.233%**；
- LPIPS 改善 **4.231%**；
- Gaussian SSIM 相对提升 **0.531%**，绝对提升 0.004814；
- PSNR 提升 0.172 dB；
- 平均 speedup 仍为 1.273×。

因此最终候选同时通过三个绝对门槛，并在相同缓存计划下取得超过 1% 的 MSE 与 LPIPS 改善。

## 4. 配对稳健性

对 12 个 case 做 paired case-wise 改善，并用固定 seed、20,000 次 bootstrap 对 case mean 求 95% 区间：

| Paired quantity | Positive cases | Case-wise mean | Median | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| MSE reduction | 10/12 | 3.786% | 3.123% | [1.600%, 6.202%] |
| LPIPS reduction | 11/12 | 4.210% | 3.854% | [1.703%, 7.081%] |
| Gaussian SSIM absolute delta | 11/12 | 0.004814 | 0.004797 | [0.002217, 0.007391] |

三个区间均完全位于正向一侧。样本量仍只有 12，因此这些区间用于检查当前实验的内部稳健性，不应被解释为大规模数据集统计结论。

## 5. 设计集与失败路线

设计集最终选择 scale 3.0：speedup 1.268×，LPIPS 相对改善 1.343%，Gaussian SSIM 相对提升 0.297%，MSE 恶化 0.397%。最终集上四项质量指标都转为改善，说明选择没有依赖最终集，但也说明小设计集估计存在方差。

本轮保留了所有失败实验，主要结论如下：

- module 9 的 on-policy surrogate 虽能把 residual test MSE 改善到约 9%，端到端图像 MSE只出现约 0.04% 的微小改善。
- 更大 module-9 step-14 surrogate 的 residual test 改善约 19.25%，端到端仍只有约 0.03% 改善。
- module 14 的局部 residual 预测明显更容易（test 改善约 68.74%），但直接正向注入最初反而损害图像；这证明局部 residual accuracy 不能替代 rollout-level 验证。
- 未做 rollout adaptation 的 module-14 checkpoint 在负 scale -2.0 时仅获得约 0.43% MSE 改善，SSIM/LPIPS没有同步改善。
- 全模型 targeted rollout 微调没有超过初始化；只训练输出 adapter 后，最佳验证点提前到 step 50，继续训练会过拟合。

这些失败结果支持最终策略：使用执行顺序靠后的模块，保留高质量 residual 初始化，只对输出适配器做早停 rollout adaptation，再在独立设计集校准注入幅度。

## 6. 可复现入口

- 基础 surrogate 配置：`configs/extra10_module14_s14.toml`
- targeted adapter 配置：`configs/extra10_module14_targeted_adapter.toml`
- 冻结最终 policy：`configs/extra10_module14_targeted_adapter_final_policy.json`
- 运行时实现：`src/sd15_residual_delta/runtime.py`
- targeted rollout 训练：`scripts/train_targeted_rollout.py`
- policy 评估：`scripts/search_correction_policies.py`
- 配对统计：`scripts/analyze_paired_final.py`
- 最终原始结果：`runs/extra10_module14_targeted_adapter_final12/correction_policy_search.json`
- 最终配对统计：`runs/extra10_module14_targeted_adapter_final12/paired_statistics.json`

## 7. 限制与下一步

当前结论只覆盖 SD1.5、单一 30-step DDIM 设置和 12 个最终样本。下一步应在更大的 prompt 集、更多 seed、不同 scheduler/step 数上复验，并报告端到端延迟分位数。若要声称严格“超过 OnlineCache”，还需把 OnlineCache 官方方法移植到同一 SD1.5 pipeline、同一硬件、同一 prompt/seed 集上做受控比较。
