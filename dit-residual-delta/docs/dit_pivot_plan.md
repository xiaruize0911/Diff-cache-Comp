# DiT 侧实验计划(2026-08-21 修订)

承接 SD1.5 arm 的负结果。骨干选 **PixArt-Σ-XL-2-512-MS**(0.6B DiT,28 blocks,
20-step,ungated)。FLUX.1-dev / FLUX.1-schnell 在 HF 上都是 gated,需要 token;
本计划的所有测量对 DiT 类骨干是通用的,拿到 token 后同一套 harness 换 backbone 即可。

## 0. 为什么要改变原 FLUX 计划的门顺序

`/workspace/flux-residual-delta/docs/experiment_plan.md` 写于 SD1.5 结论之前,
它的 Gate 1 是"残差相对 MSE 相对 fixed reuse 改善 ≥20%,否则停"。SD1.5 的证据
说明**这个门是无效的**:

- 全量 240 位置的 surrogate 把残差相对 MSE 改善了 **15.4%**(每个 module 都 <1.0),
  部署后图像质量在所有注入幅度下**都更差**(scale 1.0 时 Gaussian SSIM −0.0053)。
  残差精度与图像质量在这个设置下是脱钩的。
- 真正有判别力的是 **oracle 上界**:把缓存残差换成真值(完美 corrector),在 25%
  的缓存位置上只拿回 **13%** 的质量差距。这一个实验就排除了容量、数据量、
  分布偏移等全部解释——它给出的是与预测器无关的结构上界。
- 速度侧同样有一个与预测器无关的结构事实:修正是缓存路径上的**净增**计算,
  `speedup = T_exact / (T_fixed + C)`,`C ≥ 0`,所以它的渐近线永远在 fixed cache
  **之下**。SD1.5 上 attention 只占端到端 41%,全缓存天花板 1.70×,修正把
  1.311× 压到最好 1.287×。

因此本计划把两个"与预测器无关的上界"提到最前面,任何 surrogate 训练都在它们
之后。**这两个门加起来只要几个 GPU 小时,而原计划的 Gate 1 要 12–24 小时。**

## 1. 预注册的门(按顺序,任一失败即停)

### G0 速度天花板(`scripts/profile_model.py`)

测 DiT block 栈占端到端 wall clock 的比例,以及 self-attn / cross-attn / FF 的拆分。

- 通过条件:**blocks 占端到端 ≥ 75%**(SD1.5 的对应量是 41%)。
- 输出的关键数是 `all_cache_speedup_ceiling = T_total / (T_total − T_blocks)`,
  即所有 reuse 步的 block 计算全部免费时的上界。SD1.5 是 1.70×。
- 若 DiT 上这个数只有 2× 出头,那么"显著加速"这一半的目标在这个骨干上同样受限,
  应先如实记录再决定是否继续。

### G1 fixed-cache 的 Pareto(`scripts/evaluate_variants.py --variants-file configs/cache_grid.json`)

不训练任何东西,只测 correction-free 的 anchor schedule:interval 2/3/4/5 和两个
front-loaded schedule,4 prompts × 2 seeds,指标全部相对同 seed 的 exact 输出
(MSE / PSNR / Gaussian SSIM / LPIPS-alex),与 SD1.5 arm 用同一批 prompt、同一套
指标实现,可直接对比。

- 这一步给出**基线 Pareto 前沿**,后续任何"带修正"的方案必须在这条前沿之外,
  而不是只跟 exact 比。
- 同时给出 DiT 对缓存的**内在容忍度**:如果 interval-2 的 SSIM 掉得比 SD1.5 少
  很多,说明 DiT 的跨步残差更平滑,方向本身更有希望。

### G2 oracle 上界(决定性,`configs/oracle_*.json`)

两个子步:

1. **per-block 敏感度**:28 个变体,每个只在一个 block 的全部 reuse 步上恢复真值。
   给出每个 block 的"完美修正收益",这是 SD1.5 per-module 敏感度的 DiT 版本。
2. **top-K oracle**:按 1. 排序,在 top-25% 的 block 上恢复真值,与 SD1.5 的
   "oracle top4 / 60 of 240 slots = 25%" 严格对齐。

- **通过条件(预注册)**:top-25% oracle 至少拿回 **40%** 的 SSIM 差距
  (SD1.5 是 13%)。低于 40% 则与 SD1.5 同构地判定:缓存损失均匀分布在所有
  slot 上,修好少数几个无用,**不再训练任何 surrogate**。
- 注意 SD1.5 观察到的一个陷阱:部分 oracle 修正会让 SSIM/LPIPS 变好而 MSE 略变差。
  因此以 SSIM/LPIPS 为主指标,MSE 仅作记录。

### G3(仅在 G0–G2 全过之后)surrogate 训练

沿用 SD1.5 的 `capture → train → rollout` 结构,但预先声明:

- 选型指标不再是残差 MSE,而是**图像级 SSIM/LPIPS 在等速点上的改善**。
- 训练前先用 G2 的 per-block 排序把部署位置限定在有收益的 block 上。
- 速度必须用 CUDA graph + `torch.compile` 实测(SD1.5 上 eager 的 0.887× 是
  launch-overhead 伪影,fusion 后是 1.240×);任何速度数字都不接受 eager 报数。

## 2. 与 SD1.5 arm 的可比性

| 维度 | SD1.5 arm | DiT arm |
|---|---|---|
| 缓存单元 | `Transformer2DModel`(attn1+attn2+ff) | `BasicTransformerBlock`(attn1+attn2+ff) |
| slot 数 | 16 module × 15 reuse step = 240 | 28 block × reuse step |
| 指标实现 | `skimage` Gaussian SSIM / LPIPS-alex | 同一实现(`src/dit_residual_delta/metrics.py`) |
| design prompts | `rollout_holdout4.txt` 前 4 条 | 同一 4 条(`data/prompts/design4.txt`) |
| oracle 机制 | `oracle_refresh_keys` | 同名同语义 |

## 3. 硬件与环境

单 A40 46GB;torch 2.4.1+cu124,diffusers 0.34.0,transformers 4.48.3。
`/workspace` 在当前 pod 上写入 430 MB/s(上一个 pod 是 19.6 MB/s),特征数据可以
直接落 `/workspace`,不必再依赖 `/dev/shm` —— 上一个 pod 重启时 tmpfs 里的 16.9GB
特征和 `/root/sd15-runs` 下的 checkpoint 全部丢失,这次不重复该错误。
