# DiT arm 门控报告(PixArt-Σ-XL-2-512-MS)

日期:2026-08-21 · 512×512,20-step DPM-Solver++,CFG 4.5,fp16 · 单 A40
指标全部相对**同 seed 的 exact 输出**;design set 4 prompts × 2 seeds,
holdout 6 prompts × 2 seeds(prompt 与 seed 都不相交)。

## 0. 一句话结论

**与 SD1.5 相反,这条路在 DiT 上没有被结构性封死,但门槛被精确量化了:**
修正器必须在**每一个**缓存位置上恢复残差差值的 **≥90%**(等价 relative
residual MSE ≤ 0.01)才有意义;只修正一部分位置(哪怕完美)几乎无用
(top-25% block 完美修正只拿回 2.1–5.4% 的质量差距)。作为参照,SD1.5 arm 上
训练出来的 surrogate 达到的是 rel_mse **0.846**,与 0.01 差两个数量级。

## 1. G0 速度天花板 —— 通过(远超门槛)

| 量 | SD1.5(U-Net) | PixArt-Σ(DiT) |
|---|---:|---:|
| 可缓存单元占端到端 wall clock | 41% | **91.6%** |
| 全缓存理论加速上界 | 1.70× | **11.96×** |
| 单图延迟(exact) | — | 0.987 s |
| 峰值显存 | — | 1.94 GiB |

拆分(20 步,GPU 事件计时,hook 开销 0.7%):blocks 911.1 ms / transformer
922.6 ms / VAE decode 59.6 ms。blocks 占 transformer 的 98.8%。
self-attn 260.7 ms + cross-attn 212.5 ms + FF 296.7 ms。

T5 文本编码 173.4 ms 是**一次性**成本,实验里用预计算 embedding 消掉;
若把它算进端到端,blocks 占比 78.1%、上界 4.56×,仍然远高于 SD1.5。

预注册门槛是 blocks ≥75%,实测 91.6%,**通过**。

## 2. G1 correction-free 的 Pareto 前沿(design set)

| schedule | exact 步 | 速度 | Gaussian SSIM | LPIPS | PSNR |
|---|---:|---:|---:|---:|---:|
| front_loaded | 13/20 | 1.407× | 0.6988 | 0.1990 | 20.51 |
| **fixed i2** | 10/20 | **1.747×** | **0.6763** | 0.2138 | 19.67 |
| front_sparse | 10/20 | 1.749× | 0.5913 | 0.2984 | 17.87 |
| fixed i3 | 7/20 | 2.283× | 0.5672 | 0.3341 | 16.87 |
| fixed i4 | 5/20 | 2.856× | 0.4346 | 0.4904 | 13.87 |
| fixed i5 | 4/20 | 3.278× | 0.3867 | 0.5866 | 12.92 |

`front_sparse` 被 `fixed i2` 支配(同速更差),front-loading 在这里**没有**
SD1.5 上那种优势。后续所有比较以 **fixed i2(1.747× / SSIM 0.676)** 为
correction-free 参照点。

**原 FLUX 计划的 segment 设计(只缓存连续 block 段)在这个骨干上被支配:**

| 变体 | 缓存范围 | 速度 | SSIM |
|---|---|---:|---:|
| fixed i2 | 全部 28 block | 1.699× | 0.6763 |
| seg_fixed | block 8–27 | 1.381× | 0.6744 |
| seg14_fixed | block 14–27 | 1.280× | 0.6818 |

留一部分 block 每步都精确算,既没换来质量也损失了速度。

## 3. G2 oracle 上界 —— 预注册门槛未过,但结论与 SD1.5 完全不同

### 3.1 只修正一部分 block:几乎无用(与 SD1.5 同向,程度更极端)

28 个 block 各自单独做完美修正(该 block 的全部 10 个 reuse 步都换成真值):
最好的单个 block 只拿回 **1.3%** 的 SSIM 差距,28 个的**单独效果之和只有 1.3%**;
其中 6 个 block 的单独修正让图像**变差**(block 27:−3.6%,block 2:−2.5%)。

累积 top-K(按上面排序,design set):

| 修正位置 | 占 reuse slot | 速度 | SSIM | 占质量差距 |
|---|---:|---:|---:|---:|
| fixed i2(无修正) | 0/280 | 1.699× | 0.6763 | — |
| oracle top1 | 10/280 | 1.726× | 0.6804 | 1.3% |
| oracle top4 | 40/280 | 1.597× | 0.6852 | 2.7% |
| **oracle top7(25%)** | 70/280 | 1.485× | 0.6937 | **5.4%** |
| oracle top21(75%) | 210/280 | 1.121× | 0.7245 | 14.9% |
| oracle 全部(= exact) | 280/280 | 0.995× | 1.0000 | 100% |

holdout 复现:top7 只拿回 **2.1%**。预注册门槛是 ≥40%,**不通过**。
(SD1.5 的对应数字是 25% slot → 13.1%,即 DiT 在这个轴上更差。)

`oracle 全部` 得到 SSIM 恰好 1.0000、LPIPS 恰好 0.0000,是 harness 的
图像级正确性验证。

### 3.2 全部位置部分修正:这才是有信息量的轴

在**所有** 280 个 slot 上把缓存残差换成 `R_anchor + α·(R_true − R_anchor)`,
α 就是"修正器恢复了残差差值的多少"。这是纯保真度探针,会跑真实 block,
所以它自己的速度(约 0.93×)没有意义。

design set,fixed i2 基线(SSIM 0.6763):

| α | SSIM | 占质量差距 | 配对 t(n=8) |
|---:|---:|---:|---:|
| 0.25 | 0.6878 | 3.5% | 5.6 |
| 0.50 | 0.7106 | 10.6% | 5.4 |
| 0.75 | 0.7613 | 26.2% | 7.1 |
| 0.90 | 0.8401 | 50.6% | 8.4 |

曲线是强凸的:**质量回收大致按 (1−α) 的高次衰减,α 不到 0.9 基本拿不到一半。**

### 3.3 激进 schedule + 修正:目标可达的窗口在这里

| 配置 | 速度 | SSIM(design) | SSIM(holdout) | 对比 i2-fixed |
|---|---:|---:|---:|---|
| i2 fixed(参照点) | 1.73× | 0.6763 | 0.6793 | — |
| i4 fixed | 2.75× | 0.4346 | 0.4982 | 差很多 |
| i5 fixed | 3.33× | 0.3867 | 0.4716 | 差很多 |
| i4 + α=0.90 | — | 0.7409 | 0.7591 | **更好**(11/12 case 胜) |
| i5 + α=0.90 | — | 0.7214 | 0.7283 | **更好**(11/12 case 胜) |
| i5 + α=0.95 | — | 0.8212 | 0.8068 | **更好**(12/12 case 胜) |

**这是与 SD1.5 的决定性区别。** SD1.5 上"激进 schedule + 完美修正"在速度和
质量两个轴上同时劣于基线;这里,α=0.9 的修正器在 i5 上同时给出约 3× 的原始
schedule 速度和优于 1.73× 基线的质量。速度侧也不再有 SD1.5 那个致命约束:
blocks 占 91.6%,i5 下有 16×28=448 次修正调用的预算约 0.10 s 仍能保住 2.5×,
按 SD1.5 fusion 后测到的每次调用约 0.1 ms 估算(448 × 0.1 ms ≈ 45 ms)是够的。

## 4. α 与残差指标的换算(把这条曲线变成可用的训练目标)

若修正器预测 `ΔR̂`,相对"不修正"(`ΔR̂=0`)的 relative residual MSE 为

```text
rel_mse = E||ΔR − ΔR̂||² / E||ΔR||²
```

α 混合对应 `ΔR̂ = α·ΔR`,于是 **rel_mse = (1−α)²**:

| α | rel_mse | 本报告实测回收(i2) |
|---:|---:|---:|
| 0.25 | 0.5625 | 3.5% |
| 0.50 | 0.2500 | 10.6% |
| 0.75 | 0.0625 | 26.2% |
| 0.90 | **0.0100** | 50.6% |
| 0.95 | 0.0025 | 68.4%(i4 上测) |

**SD1.5 arm 训练出的 surrogate 是 rel_mse 0.846(α≈0.08)。** 把它放到这条
曲线上,对应回收约 1% 的质量差距 —— 这与 SD1.5 全量部署实测到的"质量反而变差"
完全一致(修正带来的收益小于它引入的分布偏移与额外计算)。

注意这个换算的方向性假设:α 混合的误差与真值**共线**,而真实修正器的误差是
任意方向的。同样的 rel_mse 下,任意方向的误差通常比共线收缩更有害,
**所以 0.01 是乐观下界,不是充分条件。**

## 5. 结论与下一步

1. G0 **通过**(91.6% / 11.96×),G1 给出 correction-free 前沿,
   G2 的**预注册门槛(top-25% oracle ≥40%)未通过(2.1–5.4%)**。
2. 但 G2 的失败方式与 SD1.5 不同:失败的是"**只修一部分位置**"这个策略,
   不是整个方向。缓存损失均匀分布在所有 slot 上,必须**全体修正**。
3. 因此把 G3 的门槛改写为一个可直接测量的量:**在留出集上,修正器对
   `ΔR` 的 relative MSE ≤ 0.01**(α≥0.90)。低于这个精度不必进入图像级评测,
   曲线已经说明结果。这个门槛比原 FLUX 计划的"改善 20%"(rel_mse ≤ 0.80)
   严格了两个数量级,而后者已被 SD1.5 证明是无效门槛。
4. 下一步(最小可行探针,约半天):在 i5 schedule 的 448 个 slot 上抓
   teacher-forced 特征,训一个小 surrogate,只看 rel_mse 是否有希望接近 0.01。
   **在这个数之前不做任何图像级部署评测,也不做量化。**
5. 若 rel_mse 停在 0.3–0.8(SD1.5 的量级),则本方向在 DiT 上同样判负,
   且这次有定量的判负依据而不只是"部署后变差"。

## 6. 复现

```bash
python scripts/profile_model.py --steps 20 --embeddings data/embeddings/design4.pt \
  --output runs/model_profile.json
python scripts/evaluate_variants.py --config configs/pixart_sigma_512.toml \
  --prompt-file data/prompts/design4.txt --seeds 8201 8202 \
  --variants-file configs/cache_grid.json --embeddings data/embeddings/design4.pt \
  --output-dir runs/cache_grid
# oracle_per_block.json -> oracle_ceiling.json -> segment_cache.json -> aggressive_blend.json
# holdout: --prompt-file data/prompts/unseen12.txt --seeds 11201 11202
```

产物:`runs/{model_profile.json,cache_grid,oracle_per_block,oracle_ceiling,segment_cache,aggressive_blend,holdout_confirm}`。
权重:`/workspace/models/pixart_sigma_512_fp16`(1.4 GB,fp16 transformer + VAE);
prompt embedding 已预计算,不再需要 19 GB 的 fp32 T5。
