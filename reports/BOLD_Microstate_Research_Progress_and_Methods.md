# BOLD 微状态研究进展与计算方法

15 名健康对照（HC）各扫描一次，25 名帕金森病患者各在 DBS 开启（ON）和关闭（OFF）时扫描一次，共 65 次扫描。沿用冻结的 AAL78 时序，频段 0.01–0.08 Hz，采样间隔 2.1 秒。

## BOLD 微状态是怎样计算的

**1. 提取脑区信号**　每个时刻取各脑区的体素均值，一次扫描得到 78 列 BOLD 时序。

$$
x_{ti}=\frac{1}{|V_i|}\sum_{v\in V_i} B_v(t),\qquad
\mathbf X=(x_{ti})\in\mathbb R^{T\times78}.
$$

**2. 标准化空间图**　脑区时序去线性趋势得 $`y_{ti}`$，沿时间标准化后逐帧减去空间均值。

$$
z_{ti}=\frac{y_{ti}-\bar y_i}{s_i},\qquad
u_{ti}=z_{ti}-\frac{1}{78}\sum_{j=1}^{78} z_{tj}.
$$

**3. 选取训练峰**　取空间幅度 $`A_t`$ 高于前后有效帧的时刻，用其单位空间图 $`\mathbf q_t`$ 训练。

$$
A_t=\frac{\|\mathbf u_t\|_2}{\sqrt{78}},\qquad
\mathbf q_t=\frac{\mathbf u_t}{\|\mathbf u_t\|_2},\qquad
A_t>\max(A_{t-1}, A_{t+1}).
$$

**4. 学习四张模板**　按人分五折（同人 ON/OFF 同折），在训练折用参与者等权的改进 K 均值将峰图聚为四类。

$$
\widehat{\mathbf M}=\arg\max_{\|\mathbf m_k\|_2=1}
\sum_{p\in\mathcal P_{\mathrm{train}}} w_p
\max_{1\le k\le4}(\mathbf q_p^{\mathsf T}\mathbf m_k)^2.
$$

**5. 逐帧标注状态**　对留出扫描逐帧算空间相关绝对值，在每个连续块内选总代价最低的序列，每切换一次加 0.04。

$$
\begin{aligned}
r_{tk}&=\left|\mathbf q_t^{\mathsf T}\mathbf m_k\right|,\qquad
c_t(k)=\frac{A_t^2}{\langle A^2\rangle_{\mathrm{valid}}}(1-r_{tk}^2),\\
\widehat{\boldsymbol\ell}_B&=\arg\min_{\boldsymbol\ell}
\left[\sum_{t\in B}c_t(\ell_t)+0.04\sum_{t,\,t+1\in B}\mathbb I(\ell_t\ne\ell_{t+1})\right].
\end{aligned}
$$

**6. 汇总每次扫描**　在 $`N`$ 帧内按各态帧数 $`n_k`$、段数 $`b_k`$ 和拟合度，求占用率、平均停留时间与解释方差贡献（GEV）。

$$
C_k=\frac{n_k}{N},\qquad
D_k=2.1\,\frac{n_k}{b_k}\ \mathrm{s},\qquad
\mathrm{GEV}_k=\frac{\sum_{t:\,\hat\ell_t=k} A_t^2 r_{tk}^2}{\sum_{t\in\mathcal W}A_t^2}.
$$

## 微状态统计分析与目前结果

**OFF 患者与 HC 的四态时间占比有差异，且 M4 平均停留更久（+4.31 秒），同一患者的 ON 扫描较 OFF 扫描表现为 M4 平均停留缩短 3.07 秒、M1 的解释方差贡献增加 1.17 个百分点，目前尚未证实四态时间占比向 HC 靠近。**

<details>
<summary>组均值、校正 P 值与完整结果</summary>

| 比较 | 指标 | 效应或组均值 | 置换 Holm P | 头动 FD 调整后 Holm P |
|---|---|---|---:|---:|
| OFF−HC | 四态占用率组成（ILR 整体检验） | 三维整体检验 | 0.00080 | 0.00080 |
| OFF−HC | M4 平均停留 | OFF 11.98 秒，HC 7.67 秒 | 0.00080 | 0.00080 |
| ON−OFF | M4 平均停留 | −3.07 秒（95% CI −4.75 至 −1.43） | 0.00190 | 0.00170 |
| ON−OFF | M1 GEV | +1.17 个百分点（95% CI +0.69 至 +1.66） | 0.00060 | 0.00080 |
| ON−HC | 四态占用率组成（ILR 整体检验） | 三维整体检验 | 0.68423 | 0.65743 |
| 向 HC 靠近 | 到 HC 中心的距离（OFF 减 ON） | +0.124（95% CI −0.191 至 +0.412） | 0.21938 | 0.23878 |

HC 比较按参与者置换组标签，ON−OFF 在患者内翻转配对差值符号，P 值按原定家族校正。可核实顺序的 24 对均先 ON 后 OFF，刺激与顺序的影响难以分开。ON 与 HC 的等效性因缺少预设界值尚未检验。

完整结果含未显著项。[各组全部标量][scalars] · [四态占用整体检验][ilr] · [V6 配对指标][paired] · [转移矩阵与 12 条路径][transitions] · [健康参考][health] · [设计限制][limits]

</details>

<details>
<summary>计算口径、其余指标与模板图</summary>

每次扫描的权重均分给自己的峰，即上式的 $`w_p`$。HC 扫描权重为 1，ON、OFF 各为 0.5。G0 和 G1 两种峰回填方法均保留的帧组成共同窗口 $`\mathcal W`$，共 12,574 帧，每个缺口两侧独立分段。幅度平方的归一化均值取自该扫描全部有效帧。

| 指标 | 计算方法 |
|---|---|
| 出现率（次/分钟） | $`60b_k/(2.1N)`$ |
| 状态内空间离散度 | 对标签为 $`k`$ 的帧取 $`\sqrt{2(1-r_{tk})}`$ 的平均值 |
| 转移概率 | $`P_{ij}=N_{ij}/\sum_{h=1}^{4} N_{ih}`$，只计相邻有效帧，分母包含留在原状态的次数 |
| 全局切换率（次/分钟） | $`60\sum_{i\ne j}N_{ij}/(2.1N)`$ |
| 四态占用率的整体比较 | $`\mathbf H\log(\mathbf n+0.5)`$，用 Helmert 基将四态帧数转换成三个等距对数比（ILR）坐标 |

某状态未出现时，其平均停留时间和空间离散度记为缺失。跨折编号采用 Hungarian 一对一匹配，整张模板同时变号仍视为同一状态。[四态模板图](figures/bold_microstate_reference_patterns.png) · [冻结流程][frozen] · [模板与分段代码][models] · [指标代码][features] · [八帧算例与详细报告][long]

</details>

[scalars]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/results/All_Scalar_Results.csv
[ilr]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/results/Coverage_ILR_Results.csv
[paired]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/results/ON_OFF_Feature_Summary.csv
[transitions]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Transition_Dynamics.md
[health]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Normalization_Analysis.md
[limits]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Final_Scientific_Interpretation.md
[frozen]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/workflow/FROZEN_WORKFLOW.md
[models]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/workflow/source/V4_BOLD_Microstate_Reliability/scripts/v4_models.py
[features]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/scripts/micro_inputs.py
[long]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/2d53eeecfa48ba08d51199b9bf96724424d31060/reports/BOLD_Microstate_Research_Progress_and_Methods.md
