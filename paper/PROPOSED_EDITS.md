# `paper/` 与实测冲突之处：提议的修改

**状态：待审阅，未应用。** 本文件不改动任何 `.tex`。论文的科学措辞是作者的，我只给出「原文 → 提议」和支撑证据，你批准哪几处我再应用哪几处。

六处冲突，按严重程度排序。**站点 1 和 3 是必须改的**（现在的文字与实测直接矛盾），其余四处是「陈述已过时」。

每处的证据都可从 `runs/` 复算，出处见 `report/REPORT.md` 附录。

---

## 站点 1（必改）· `results.tex:20–22` · 机制的因果被归错了

**原文：**
```latex
so deviations \emph{multiply}. The more faithfully $C$ imitates the true block, the
larger the feedback gain it reintroduces. Consequently the number of injection sites
per step, not the accuracy at each site, dominates stability.
```

**问题：** 「越忠实 ⟹ 反馈增益越大」现在被三个测量同时否掉了：

| | 残差准度 | K=28 表现 |
|---|---|---|
| Taylor 一阶（状态无关） | 50 步下**比我们准** | 不受影响（±0.0002，t 不显著） |
| Block Caching（状态无关） | 比我们**差 5 倍**（0.2687 vs 0.0537） | 存活（+0.0117，t=2.4） |
| 我们的（状态相关） | **最准** | 崩塌（−0.3939，t=−34.5） |

忠实度既不必要也不充分。真正的判据是 `∂R̂/∂h` 是否为零。

**提议：**
```latex
so deviations \emph{multiply}. What reintroduces the feedback gain is not fidelity but
\emph{state dependence}: a corrector that reads the current hidden state has
$\partial \hat{\dR}/\partial h \neq 0$ and turns the recursion multiplicative, while one
that reads only cached quantities leaves $J_b = 0$ and the accumulation additive, however
accurately it predicts. Consequently the number of \emph{state-dependent} injection sites
per step, not the accuracy at each site, dominates stability.
```

**为什么这么改而不是小修：** 原句把一个可测的结构性质（读不读当前状态）说成了一个连续量（有多忠实）。这不是措辞问题 —— 它让机制做出错误的预测，比如会预测 Block Caching 在 K=28 崩塌，而它不崩。

---

## 站点 2（必改）· `discussion.tex:82–87` · 机制不再是「论证的」

**原文：**
```latex
\paragraph{Mechanism is argued, not proved.} Eq.~\ref{eq:multiplicative} is a
linearisation. We test its predictions --- the granularity inversion, the damping
requirement, and the weaker damping requirement on the backbone with the flatter
$\tau$ profile --- but we do not measure $J_b$ or its spectral radius directly, so the
mechanism should be read as the best available explanation rather than a
demonstrated one.
```

**问题：** `J_b` 现在测了（`scripts/measure_jacobian.py`）。

**支撑数据**（部署算子，含缩放，fp32 反向模式幂迭代，真实采集状态）：

| K | 每点 ρ 中位数 | 每次前向 ∏ρ | ΔSSIM (σ=1, n=192) |
|---|---|---|---|
| 1 | 12.28 | 12.3 | −0.0028 |
| 2 | 5.46 | 19.2 | −0.1649 |
| 4 | 3.64 | 139.9 | −0.3115 |
| 28 | 1.61 | **1.40 × 10⁶** | −0.4142 |
| 任意 K，状态无关族 | 1.00 | 1.00 | +0.0117 |

外加：val 独立选出的 σ 把乘积从跨五个数量级压回 4.5–30 的窄带。

**提议：**
```latex
\paragraph{Mechanism, measured.} Eq.~\ref{eq:multiplicative} is a linearisation, but its
central quantity is now measured rather than inferred. For the three correction families
that read only cached quantities the Jacobian is $I$ analytically, so $\rho = 1$ with
nothing to measure. For ours, power iteration on the deployed operator gives a per-site
$\rho$ that \emph{falls} monotonically with granularity ($12.3 \to 1.6$ from $K{=}1$ to
$K{=}28$) while the product over the $K$ sites a reuse step traverses \emph{climbs} five
orders of magnitude ($12.3 \to 1.4\times10^{6}$), and deployed quality tracks the product,
not the per-site value. Damping selected on validation, which never sees a Jacobian,
independently compresses that product into a $4.5$--$30$ band. What remains inferred is
the step-to-step behaviour: inside the band the product does not order quality, because
damping suppresses the instability and the correction together, so it explains the
collapse and not the gain.
```

**注意我保留了一个「未证明」：** 乘积在窄带内不排序质量。不要把这段写成机制已被完全证明。

---

## 站点 3 · `results.tex:43–68` · `tab:granularity` 应换成受控版本

**问题：** 现在的表是 n=8 design split，而且四行里**只有承载崩塌结论的 K=28 那行配方不同**（w384 d3、rank 128、12k 步、batch 192、关掉 token 混合），其余三行是 w512 d4 rank256 16k 步 batch48。表注已经承认「not matched correctors」。

**提议：** 换成受控阶梯（同配方、n=192、σ=1，`runs/ctrl_ladder_test`）：

| K | blk/seg | 参数 | val rel-MSE | ΔSSIM vs 纯缓存 |
|---|---|---|---|---|
| 1 | 28 | 11.88M | 0.3935 | −0.0028 (t=−0.6, n.s.) |
| 2 | 14 | 13.06M | 0.3149 | −0.1649 (t=−28.7) |
| 4 | 7 | 15.42M | 0.2382 | −0.3115 (t=−51.6) |
| 28 | 1 | 43.74M | **0.0591** | **−0.4142** (t=−34.2) |

**三点必须进表注：**
1. 配方完全相同，K=28 那行不再是异类。
2. **容量与数据都偏向输的一方** —— rank-256 的 per-block 旁路在 K=28 复制 28 份，所以它有 43.74M 参数（K=1 的 3.7 倍）和 8 倍梯度样本，却崩得更彻底。
3. **这张表不再包含「修正有用」的证据** —— 受控 K=1 在 σ=1 下只是中性（−0.0028），因为它用标准损失、只有 1,792 slot。增益的证据在主表（+0.0588，σ 由 val 选出）。旧表的 K=1 行是 +0.0516，换表后这个数字消失，别让读者以为结论变弱了 —— 是两件事被分开了。

数据量仍未严格匹配（K=1/2/4 用 112 张图，K=28 用 32 张图），两个不对称方向相反，应进 Limitations。

---

## 站点 4 · `discussion.tex:22–24` · 对 TaylorSeer 的让步已过时

**原文：**
```latex
The natural question our result raises --- whether a learned
corrector trained with a site-balanced objective could close that gap --- is one we
cannot answer without the baselines we did not run.
```

**问题：** 跑了。而且答案不是「能」或「不能」，是**有一个交叉点**。

**支撑数据**（20 步，n=192，Taylor 一阶 vs 同 i 纯缓存）：i=2 **+0.0084**（t=3.5）、i=3 **+0.0118**（t=3.9）、i=4 **−0.0145**（t=−5.0）、i=5 **−0.0278**（t=−9.6）。50 步 i=5：**+0.0440**（t=8.8）。

决定性变量是**外推跨度相对轨迹光滑度**，步数和 anchor 间隔都在动它。

**提议：**
```latex
We have since implemented that mechanism rather than conceding to it, and the answer is
an interval rather than a verdict. A first-order forecast of the cached residual helps at
mild caching and \emph{hurts} at aggressive caching: on our primary backbone at $20$ steps
it gains $+0.0084$ and $+0.0118$ SSIM at $i{=}2,3$ and loses $-0.0145$ and $-0.0278$ at
$i{=}4,5$ ($n{=}192$ throughout), while at $50$ steps and $i{=}5$ it gains $+0.0440$. The
variable is the extrapolation span relative to trajectory smoothness, which both the step
count and the anchor interval move. We remain uncompetitive in absolute terms, but the gap
is now characterised rather than conceded.
```

---

## 站点 5 · `discussion.tex:174–175` · 「没有给出解法」不再准确

**原文：**
```latex
is that we identified a real problem with the standard objective and did not demonstrate
a solution to it.
```

**问题：** 50 步、i=5 上有一个可证的解：把 corrector 训练在 forecast 留下的残差上，`R_true − R_taylor`。

**支撑数据**（50 步，i=5，n=48）：对基线 **+0.0675**（t=12.0，47/48）、对 Taylor 单独 **+0.0235**（t=7.4）、对 corrector 单独 **+0.0231**（t=4.1）、对朴素叠加 **+0.0125**（t=3.6）。LPIPS 全场最好。双重计数被直接确认：去掉后可承受 σ=0.5 而非 0.25。

**但范围很窄**（20 步下完全不成立，n=192：corrector 单独打败所有叠加变体，i=5 时 +0.0555，t=15.6）。

**提议：**
```latex
is that we identified a real problem with the standard objective, and demonstrated a
solution to it only on a regime narrower than the one we set out to address. Training the
corrector on the residual a training-free forecast \emph{leaves behind} rather than on the
cache error beats both components and their naive composition at $50$ steps and $i{=}5$
($+0.0235$ SSIM over the forecast alone, $t{=}7.4$; $+0.0231$ over the corrector alone).
It does not transfer to the $20$-step operating point this paper is otherwise built on,
where the corrector alone beats every composed variant, because the composition is worth
doing exactly where its base is worth doing and the forecast is a negative base there.
```

---

## 站点 6 · `paper.tex:139–147` · Related Work 现在可以说得更硬

**原文（节选）：**
```latex
\textbf{We are not competitive with these methods in absolute terms and
do not claim to be}; our operating points are chosen to make a diagnostic comparison
clean, not to win a speed/quality benchmark.
```

**问题：** 这一段列了 TaylorSeer、Block Caching、Learning-to-Cache 三个，我们现在**复现了前两个的机制**，而且 Block Caching 的复现是机制的一个**预注册预测确认**（它只读 timestep → 预测它在 K=28 存活 → 确认，+0.4056 对我们，t=38.7，192/192）。

**提议：** 保留那句不竞争的声明（仍然成立），在其后加：
```latex
We do, however, implement the first two of these mechanisms rather than only citing them,
and the second is where the recursion earns its keep: Block Caching's scale--shift is
conditioned on the timestep alone, so $\partial \hat{\dR}/\partial h = 0$ and
Eq.~\ref{eq:multiplicative} predicts \emph{before measurement} that it survives per-block
injection where ours cannot. It does, by $+0.4056$ SSIM at $K{=}28$ ($t{=}38.7$,
$192/192$), with $5\times$ worse residual accuracy, half the parameters and lower FLOPs
than the corrector it beats there.
```

---

## 还需要的结构性改动（我没起草，因为它们改的是论文骨架）

1. **Contributions 列表**要新增一条（机制的预注册预测被确认）并修改现有那条「A large, mechanically explained decoupling」的措辞以匹配站点 1。
2. **Abstract** 现在没有提任何复现，而 Block Caching 那条是全篇最强的机制证据。
3. **Conclusion 的 "Three of our five findings are negative"** 计数需要重算 —— 现在有一个受限的正面方法结果。
4. **`runs/flops.json` 的计价一致性修复**（另一个 session 完成）如果落地，Cost accounting 一节应提一句两侧现在共用同一 dispatch。

## 两处需要你决定、我不该自己定的

1. **50 步的结果要不要进主表。** 现在 n=48、单训练种子，而 20 步那批是 n=192。进主表就得补样本量（清单第 6 项）。
2. **README 里的 4/10 与 9/10 无出处**（清单第 7 项）。全仓库没有任何评审或审计 artifact 支撑。这是全篇唯一一个读者无法核验的断言，而论文的说服力恰恰建立在「每个数字都能从 `runs/` 复算」上。补出处或撤掉，都要你定。
