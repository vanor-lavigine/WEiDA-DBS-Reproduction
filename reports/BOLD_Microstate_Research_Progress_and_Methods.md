# BOLD 微状态研究进展与计算方法

**更新日期：2026 年 11 月 26 日。** 本报告汇总已归档的 15 名 HC 与 25 名 ON/OFF 配对患者的 BOLD 微状态分析，按研究问题与计算流程组织。方法和数值对应文末的固定提交；这不是一次新运行的实验。

**阅读说明：** 在 GitHub 文件页面的预览视图阅读。公式使用 GitHub 支持的数学块和行内数学格式，每个关键公式后附中文含义。

## 研究对象与数据

- **研究目标**
  - 使用静息态 BOLD-fMRI 时间序列构建四种微状态（M1–M4）。
  - 比较 PD 患者 DBS-OFF、DBS-ON 与健康对照（HC）的微状态时空特征。
  - 当前重点转向同一患者 DBS-ON 与 DBS-OFF 的配对变化，进一步评估这些变化是否与健康参考状态接近。

- **队列**
  - HC：15 名。
  - PD：25 名，每名患者有 ON、OFF 两次扫描。
  - 共 40 名独立参与者、65 次扫描。
  - 患者 ON/OFF 比较以 25 名患者配对为统计单位；不能把 50 次扫描当成 50 名独立参与者。

- **BOLD 数据**
  - 图谱：AAL78，提取 78 个 ROI 的 BOLD 时间序列。
  - TR：2.1 s。
  - 上游时间滤波：0.01–0.08 Hz。
  - 每次扫描的原始时钟约 210 TR；实际进入共同有效分析窗的帧数因扫描而略有不同。
  - 冻结分析的共同有效窗总计 12,574 帧。
  - 当前主要分析沿用冻结的 ROI 时序、五折模板及原始标签。
  - 这里的“微状态”指 BOLD 空间模式的离散状态与秒级持续过程；TR 为 2.1 秒，不能由此推断 EEG 的毫秒级微状态。

依据：[冻结流程][workflow]、[锁定统计配置][protocol]。

## BOLD 微状态是怎样计算的

### 1. 从 BOLD 到 ROI 时间序列矩阵

对一份扫描，先将预处理后的 BOLD 数据按 AAL78 图谱分区。对第 $`i`$ 个 ROI，在第 $`t`$ 个 TR 内对该 ROI 的有效体素信号取平均，得到：

```math
x_i(t)=\frac{1}{|V_i|}\sum_{v\in V_i} B_v(t),
\qquad i=1,\ldots,78.
```

其中：

- $`B_v(t)`$：体素 $`v`$ 在时间点 $`t`$ 的 BOLD 信号。
- $`V_i`$：AAL78 中第 $`i`$ 个 ROI 所包含的体素集合。
- $`x_i(t)`$：第 $`i`$ 个 ROI 的平均 BOLD 时间序列。

因此，每份扫描可写成一个时间 × ROI 矩阵：

```math
\mathbf X=
\begin{bmatrix}
x_1(1)&x_2(1)&\cdots&x_{78}(1)\\
x_1(2)&x_2(2)&\cdots&x_{78}(2)\\
\vdots&\vdots&\ddots&\vdots\\
x_1(T)&x_2(T)&\cdots&x_{78}(T)
\end{bmatrix}
\in\mathbb R^{T\times78}.
```

每一行代表一个 TR 的全脑 78 维空间活动模式；每一列代表一个 ROI 随时间变化的 BOLD 信号。

### 2. ROI 标准化与每个 TR 的空间向量

当前冻结方案采用 N0 标准化。对每个 ROI，先沿时间维去均值并按时间标准差标准化，再对每个 TR 的 78 维空间向量去除空间均值：

```math
\widetilde x_i(t)=
\frac{x_i(t)-\bar x_i}{s_i},
\qquad
\bar x_i=\frac{1}{T_0}\sum_{t\in\mathcal V_0}x_i(t),
```

```math
s_i=
\sqrt{\frac{1}{T_0-1}
\sum_{t\in\mathcal V_0}(x_i(t)-\bar x_i)^2}.
```

其中 $`\mathcal V_0`$ 是 78 个 ROI 均为有限值的帧集合，$`T_0=|\mathcal V_0|`$。

每个 TR 的空间均值去除为：

```math
u_i(t)=\widetilde x_i(t)-
\frac{1}{78}\sum_{j=1}^{78}\widetilde x_j(t).
```

将 78 个 ROI 组成向量：

```math
\mathbf u(t)=[u_1(t),\ldots,u_{78}(t)]^{\mathsf T}.
```

再计算空间向量的 L2 范数及单位向量：

```math
A(t)=\frac{\|\mathbf u(t)\|_2}{\sqrt{78}},
\qquad
\widehat{\mathbf u}(t)=
\frac{\mathbf u(t)}{\|\mathbf u(t)\|_2}.
```

- $`\widehat{\mathbf u}(t)`$：该 TR 的标准化全脑空间模式，用于与微状态模板计算空间相似度。
- $`A(t)`$：该 TR 的 **BOLD 空间标准差 / GFP 类比幅度**；在本项目实现中为去空间均值后的 ROI 向量范数除以 $`\sqrt{78}`$。它不是 EEG 的电场强度。
- 非有限值、零范数等无效 TR 不参与状态回填；时间缺口不会被跨越连接。

完成上述标准化后，定义后续有效帧集合 $`\mathcal V=\{t\in\mathcal V_0:\|\mathbf u(t)\|_2>0\}`$，$`T_v=|\mathcal V|`$。这一区分保留了代码先估计时间均值和标准差、再排除空间零范数帧的次序。

### 3. GFP 峰值提取

本项目借用 GFP 的空间标准差形式，衡量每个 TR 的 ROI 信号在空间上的离散幅度。根据上式得到 $`A(t)`$，在连续有效时间点中寻找局部峰值：

```math
A(t)>A(t-1),\qquad A(t)>A(t+1).
```

同时要求 $`t-1,t,t+1`$ 都是有效帧。

GFP 峰值代表空间模式幅度相对较大的时间点。当前 G1 模板训练使用这些峰值对应的空间单位向量，而不是直接将所有 TR 等权用于模板聚类。

### 4. 训练集 GFP 峰图与共同微状态模板

当前 G1 方法**直接汇集训练扫描的 GFP 峰单位空间图**，再学习四个共同空间轴。每个峰输入的是 $`\widehat{\mathbf u}_s(t)`$。此步骤没有先为每个人估计四张模板；先个体后群组的分层方法属于用于比较的 G0。

参与者权重设为：HC 的一次扫描总权重为 1，同一患者 ON、OFF 两次扫描各为 0.5。再将每次扫描的权重均分给其 GFP 峰。设扫描 $`s`$ 的峰集合为 $`\mathcal P_s`$，则：

```math
w_{st}=\frac{\omega_s}{|\mathcal P_s|},
\qquad
\omega_s=
\begin{cases}
1, & s\text{ is an HC scan},\\
0.5, & s\text{ is an ON or OFF scan}.
\end{cases}
```

该权重使每名参与者的总贡献相同；HC 与 PD 两组的总权重仍与各自人数有关。

模板学习使用极性不敏感的 Modified K-means，固定 $`K=4`$：

```math
\max_{\|\mathbf m_k\|_2=1}
\sum_{s\in\mathrm{train}}
\sum_{t\in\mathcal P_s}
w_{st}
\max_{k\in\{1,2,3,4\}}
\left(\widehat{\mathbf u}_s(t)^{\mathsf T}\mathbf m_k\right)^2.
```

**中文含义：** 找到四个单位长度空间轴，使训练集各峰与最相似轴之间的加权平方投影总和最大。模板 $`\mathbf m_k`$ 与 $`-\mathbf m_k`$ 表示同一轴。

将峰分到最相似的模板后，各模板更新为所属峰的加权二阶矩矩阵的第一特征向量：

```math
\mathbf S_k=
\sum_{(s,t):c_{st}=k}
w_{st}\widehat{\mathbf u}_s(t)\widehat{\mathbf u}_s(t)^{\mathsf T},
\qquad
\mathbf m_k=\mathrm{eigvec}_{\max}(\mathbf S_k).
```

代码使用 50 个固定随机初值，选择收敛候选中训练目标最高的结果。这里每个峰的权重按扫描均分，没有另乘 GFP²；后续分段和 GEV 另使用幅度平方权重。

四张模板组成 $`\mathbf M\in\mathbb R^{4\times78}`$。M1–M4 是本次共同模板的编号，不能直接对应 EEG 的 A–D，也不能未经空间匹配就与其他模板方案的同号状态合并。

依据：[空间向量准备][prepare]、[模板学习与动态规划实现][models]。

### 5. 五折交叉拟合与状态回填

- 40 名独立参与者按既定五折划分；同一 PD 患者的 ON/OFF 两次扫描始终在同一折。
- 每折使用其余参与者训练四张模板，再对留出折参与者进行状态回填。
- 五折模板以 Fold 0 的四张模板为状态编号参考，通过绝对空间相关和 Hungarian 一对一匹配，将各折状态对应到 M1–M4。
- 每折训练完成的模板用于其留出参与者；原始 65 次扫描标签已冻结并通过逐帧归档核验。
- 因此，五折的作用是获得折外微状态特征，降低同一参与者既参与模板学习又用于自身特征估计的依赖；五折本身不是显著性检验，也不构成独立队列验证。Fold 0 只是命名参考，每份留出扫描实际使用其所属折训练出的模板。

### 6. 每个 TR 与四张模板计算空间相似度

对有效 TR 的单位空间向量 $`\widehat{\mathbf u}(t)`$，与第 $`k`$ 张模板 $`\mathbf m_k`$ 计算空间 Pearson 相关。由于模板和数据已进行空间去均值及 L2 单位化，相关可表示为内积：

```math
r_k(t)=
\widehat{\mathbf u}(t)^{\mathsf T}\mathbf m_k.
```

当前 A3 使用极性不敏感匹配：

```math
s_k(t)=|r_k(t)|.
```

即整体正负号相反但空间形状相同的模式按相同状态相似度处理。

如果只进行逐 TR 最大相似度分类，则：

```math
\ell_t=\arg\max_{k\in\{1,2,3,4\}}s_k(t).
```

但当前正式 A3 并非简单地对每个 TR 独立取最大值，而是继续进行时间连续性约束。

### 7. A3 时间分段：GFP 加权的动态规划

先区分两个时间集合：

- $`\mathcal V`$：原始有效帧，用于幅度归一化。
- $`\mathcal W`$：G0 与 G1 的峰值回填方法共同保留的分析窗口，用于实际分段及指标计算。无效点与边缘剔除按冻结规则保留，$`\mathcal W\subseteq\mathcal V`$。

A3 使用 G1 模板分数，并通过 S1 动态规划对 $`\mathcal W`$ 中的每个连续时间块求解；窗口内的每个 TR 都参与，而非只给 GFP 峰分类。目标函数为：

```math
\widehat{\boldsymbol\ell}
=
\arg\min_{\ell_1,\ldots,\ell_T}
\left[
\sum_{t\in\mathcal W}
\frac{A(t)^2}{\overline{A^2}}
\left(1-s_{\ell_t}(t)^2\right)
+
\lambda\sum_{(t-1,t)\in\mathcal E_{\mathcal W}}
\mathbb I(\ell_t\ne\ell_{t-1})
\right].
```

其中：

```math
\overline{A^2}
=
\frac{1}{T_v}\sum_{t\in\mathcal V}A(t)^2,
\qquad \lambda=0.04.
```

- 第一项：当前 TR 分配给某状态时的空间拟合损失；空间相关越高，损失越小。
- GFP 权重：幅度较大的 TR 对拟合目标贡献更大。
- 第二项：状态切换惩罚；相邻有效 TR 状态发生变化时增加 $`\lambda`$。
- $`\mathcal E_{\mathcal W}`$：原始时钟上相邻且都在 $`\mathcal W`$ 中的帧对。无效时间点两侧不跨缺口连接。
- 动态规划同时考虑空间拟合与时间连续性，因此不是逐 TR 独立分类。
- A3 是 G1 + S1 的组合；K=4，λ=0.04。λ 是无量纲切换代价，当前方案没有“至少持续 2 TR”的硬阈值。
- 参数由已记录的模拟校准回退规则选出；高噪声门槛未通过，因此不能把该选择表述为已经验证的最优方法。

设单帧损失为 $`c_t(k)=A(t)^2(1-s_k(t)^2)/\overline{A^2}`$。每个连续块的动态规划从 $`D_1(k)=c_1(k)`$ 开始：

```math
D_t(k)=c_t(k)+
\min\left\{
D_{t-1}(k),
\min_{j\ne k}\left[D_{t-1}(j)+\lambda\right]
\right\}.
```

从最小终点代价向前回溯，得到该时间块的状态序列。

最终得到分析窗口内每个 TR 的状态标签：

```math
\ell_t\in\{1,2,3,4\}.
```

每个保留时间点被归入 M1、M2、M3 或 M4 中的一个状态。

依据：[共同窗口与分段入口][inputs]、[动态规划实现][models]。

---

## 微状态指标如何从 BOLD 标签计算

设一份扫描在共同分析窗口 $`\mathcal W`$ 中共有 $`N=|\mathcal W|`$ 个有效且已分类的 TR，TR 长度为 $`\Delta t=2.1`$ 秒。设第 $`k`$ 个状态的连续片段数为 $`N_k`$，所有连续片段集合记为 $`\mathcal R_k`$。

### 1. Coverage / Fractional Occupancy

第 $`k`$ 个微状态占据的有效帧数除以总有效帧数：

```math
C_k=
\frac{\sum_{t\in\mathcal W}\mathbb I(\ell_t=k)}
{N}.
```

四个状态的 Coverage 满足：

```math
\sum_{k=1}^{4}C_k=1.
```

解释：某一微状态在整段扫描中占据的时间比例。Coverage 是组成数据，四个状态并非相互独立。

### 2. Mean Duration / Dwell Time

先把连续相同标签定义为一个状态片段。若第 $`j`$ 个片段长度为 $`L_{kj}`$ 帧，则持续时间为：

```math
d_{kj}=L_{kj}\times\Delta t.
```

第 $`k`$ 个状态的平均持续时间：

```math
D_k=
\frac{1}{N_k}\sum_{j=1}^{N_k}d_{kj}.
```

解释：一次进入该状态后，平均连续维持多长时间。若某状态未出现，Duration 和 Dispersion 记为缺失，不能补成 0。当前主指标保留共同有效观察窗首尾处实际可见片段；边界删失版本作为敏感性分析单独报告。

### 3. Occurrence Rate

第 $`k`$ 个状态的连续片段数量除以总有效扫描时间：

```math
O_k=
\frac{N_k}{N\Delta t}.
```

单位为 episodes/s；换算为次/分钟时乘以 60。

解释：单位时间内该状态出现多少次。Coverage 衡量总占用时间，Occurrence 衡量状态段出现的频率。在使用同一窗口和边界规则时，三者满足：

```math
C_k=D_k O_k.
```

其中 $`O_k`$ 使用次/秒；若使用次/分钟，则 $`C_k=D_k O_{k,\mathrm{min}}/60`$。因此这三类指标存在代数关联，不能把它们当成完全独立的证据。

### 4. Global Explained Variance（GEV）

对每个有效 TR，取其被分配状态模板的空间相似度 $`s_{\ell_t}(t)`$，并以 GFP 的平方加权：

```math
GEV_k=
\frac{
\sum_{t\in\mathcal W}
\mathbb I(\ell_t=k)A(t)^2s_k(t)^2
}{
\sum_{t\in\mathcal W}A(t)^2
}.
```

解释：第 $`k`$ 个微状态对整段扫描全局空间方差/幅度加权拟合的贡献。GEV 同时受状态占用、幅度和模板拟合程度影响，不能单独解释为某脑网络活动增强。必须使用最终动态规划标签对应的模板，不能把每帧最大相关直接代入。四态 GEV 的总和是总解释贡献，通常小于 1；它不像 Coverage 那样必须合计为 1。

### 5. Spatial Dispersion

先对每帧空间向量和对应模板进行去空间均值及 L2 单位化。对被分配到状态 $`k`$ 的每一帧，计算极性不敏感的单位向量距离：

```math
d_k(t)=\sqrt{2\left(1-|r_k(t)|\right)}.
```

第 $`k`$ 个状态的 Spatial Dispersion 为该状态所有有效帧距离的算术平均：

```math
SD_k=
\frac{
\sum_{t\in\mathcal W}
\mathbb I(\ell_t=k)d_k(t)
}{
\sum_{t\in\mathcal W}\mathbb I(\ell_t=k)
}.
```

解释：该状态内各帧空间模式与其模板之间的平均距离。数值越大，表示状态内空间模式相对模板越分散。该指标不进行 GFP 加权，不是 BOLD 时间序列方差，也不同于 WEiDA 的 Local Metastability/VAR。

### 6. Transition Probability

在两个相邻且都有效的 TR 之间，若前一帧为 $`i`$、后一帧为 $`j`$，则计为一次 $`i\to j`$ 转换。定义计数：

```math
N_{ij}
=
\sum_{t}
\mathbb I(\ell_t=i,\ell_{t+1}=j)
\mathbb I(t,t+1\text{ 均有效}).
```

以源状态 $`i`$ 所有具有有效后继的相邻帧数为分母：

```math
P_{ij}=
\frac{N_{ij}}{\sum_{j'=1}^{4}N_{ij'}}.
```

- 每一行表示从某个源状态出发，下一有效相邻 TR 落入各状态的条件概率。
- 完整 4×4 矩阵包含自循环 $`P_{ii}`$。
- 主要路径定位检验关注 12 条非对角有向转换；保留上述分母，不删除自转移后再次归一化。
- 有有效后继的源状态行合计为 1；分母为 0 的行不定义。
- 不会跨越无效 TR 缺口构造虚假的状态转换。

### 7. Global Switching Rate

统计相邻有效 TR 标签发生变化的次数：

```math
N_{switch}
=
\sum_t
\mathbb I(\ell_t\ne\ell_{t+1})
\mathbb I(t,t+1\text{ 均有效}).
```

全局切换率：

```math
SR=
\frac{N_{switch}}{N\Delta t}.
```

单位为次/秒；乘以 60 可转换为次/分钟。

### 8. Coverage 的 ILR 组成变换

因为四态 Coverage 总和为 1，不能将四个比例当作四个独立变量。当前整体比较对组成向量进行 ILR 变换。

对四态帧数 $`n_k`$ 加 0.5 的伪计数，构造：

```math
\mathbf z=
\mathbf H
\begin{bmatrix}
\log(n_1+0.5)\\
\log(n_2+0.5)\\
\log(n_3+0.5)\\
\log(n_4+0.5)
\end{bmatrix},
```

其中 $`\mathbf H`$ 为三维 Helmert 正交基：

```math
\mathbf H=
\begin{bmatrix}
1/\sqrt2&-1/\sqrt2&0&0\\
1/\sqrt6&1/\sqrt6&-2/\sqrt6&0\\
1/\sqrt{12}&1/\sqrt{12}&1/\sqrt{12}&-3/\sqrt{12}
\end{bmatrix}.
```

该变换得到三维 ILR 坐标，用于整体组成差异和健康参考距离分析。因为 $`\mathbf H\mathbf 1=0`$，先把加伪计数的帧数归一化成比例再取对数，会得到同样的坐标。

依据：[当前指标与 ILR 的直接计算实现][inputs]。

---

## 微状态统计分析与目前结果

### 1. 统计比较设计

- OFF–HC：患者 OFF 与健康对照的独立组比较。
- ON–HC：患者 ON 与健康对照的独立组比较。
- ON–OFF：同一患者内的配对比较。
- 当前汇总优先报告参与者级置换与相应 Holm 校正。归档的常规参数/非参数检验结果单独保留，不与置换 P 值混用。
- ON–OFF 的置换在患者内进行符号翻转，不把两次扫描拆成独立样本。
- 平均 FD、标准化 DVARS 是分别调整的两个敏感性模型；表中的两列不是同时调整 FD 和 DVARS 的联合模型。
- 多重比较使用 BH-FDR、Holm、置换 Holm；不同家族和更广泛联合校正分别报告。
- ON–OFF 的 95% CI 来自 2,000 次固定模板、患者配对 Bootstrap，条件于原五折模板和标签；区间未按全部检验进行多重调整，不能代替 Holm 判定。
- 校正家族：ON–OFF 主要终点 2 项、次要终点 6 项、探索性整体检验 2 项、转移路径 12 项。OFF–HC 使用对应归档家族；不同家族的显著结果不能计为相互独立的重复发现。

两项配对主要终点沿用同一队列此前已经观察到的结果。本轮聚焦、留一分析和质量敏感性增加了对该队列的描述，不能称为独立确认性验证。

依据：[统计方法][stats]、[配对结果][paired]、[证据解释][interpretation]。

### 2. ON–OFF：当前重点报告的两项主要终点

| 指标 | ON−OFF 效应 | 95% CI | 置换 Holm P | FD Holm P | DVARS Holm P |
|---|---:|---|---:|---:|---:|
| M1 GEV | **+1.17 个百分点** | [+0.69, +1.66] | **0.00060** | **0.00080** | **0.00100** |
| M4 Mean Duration | **−3.07 s** | [−4.75, −1.43] | **0.00190** | **0.00170** | **0.00230** |

- M1 GEV：21/25 名患者 ON 高于 OFF。
- M4 Mean Duration：19/25 名患者 ON 低于 OFF。
- 两项主要终点在逐一留一患者分析中均保持效应方向。
- 这支持 DBS 条件与 BOLD 微状态动力学变化相关，但由于扫描顺序混杂，不能直接表述为完全识别的因果效应。

### 3. ON–OFF：次要及整体结果

| 指标 | ON−OFF 效应 | 置换 Holm P | FD Holm P | DVARS Holm P |
|---|---:|---:|---:|---:|
| M1 Occurrence | +0.40 次/分钟 | **0.02500** | **0.01800** | 0.05749 |
| Global Switching Rate | +0.84 次/分钟 | **0.03040** | **0.00660** | 0.05749 |
| M1 Coverage | +5.93 个百分点 | **0.01980** | 0.06239 | **0.03780** |

| 整体检验 | 原始置换 P | 两项家族 Holm P | FD Holm P | DVARS Holm P |
|---|---:|---:|---:|---:|
| 四态 Coverage ILR（3 维） | 0.00090 | **0.00180** | **0.00680** | **0.00220** |
| 非对角 Transition（12 维） | 0.00180 | **0.00180** | **0.00680** | **0.00220** |

- M4→M1 的条件转移概率增加 **5.66 个百分点**，95% CI [2.35, 9.28]。12 条路径的无协变量模型的置换 Holm P=**0.04440**；FD Holm P=0.05999，DVARS Holm P=0.06119，常规检验 Holm P=0.05816。其他 11 条路径未通过无协变量模型的置换 Holm。
- M3、M4 Coverage 的无协变量模型的置换 Holm P 分别为 0.05499、0.05369，没有通过 0.05；更宽松的 BH 结果属于补充证据。
- M3 Dispersion 只在 FD 分支达到四态 Holm P=0.03960，DVARS Holm P=0.12919，依赖调整设定。
- M4 排除边界段后的平均持续时间差为 −3.28 s，95% CI [−4.85, −1.77]；17 项时长探索家族置换 Holm P=0.01020，FD Holm P=0.01360。它是同一平均停留发现的敏感性支持。

依据：[配对结果与全部标量][paired]、[整体与路径检验][transition]。

### 4. OFF–HC：疾病相关组间结果

| 指标 / 分析 | OFF−HC 效应 | 归档家族置换 Holm P | FD Holm P |
|---|---:|---:|---:|
| 四态 Coverage ILR 整体组成 | 整体组成不同 | **0.00080** | **0.00080** |
| Coverage Aitchison Energy Distance | 整体分布不同 | **0.00240** | **0.00480** |
| M4 Mean Duration | +4.312 s | **0.00080** | **0.00080** |
| M3 GEV | −2.76 个百分点 | **0.00330** | **0.00550** |
| M3 Spatial Dispersion | +0.0673 | **0.00120** | **0.00120** |
| M3 Occurrence | −0.640 次/分钟 | **0.00400** | **0.00800** |
| M4 P75 Dwell Time | +5.31 s，95% CI [2.88, 7.91] | **0.00240** | **0.00720** |

- 两种整体组成检验从不同角度支持 OFF 与 HC 存在差异，不应计作两个独立复现。
- M3 出现率的计数模型得到 IRR=0.673，95% CI [0.555, 0.809]；计数模型家族 Holm P=0.00300，FD Holm P=0.00722。该结果与出现率降低方向一致。
- 这些 P 值保留各自归档的家族定义。扩展到跨指标、跨模型的联合校正后，部分结论会改变；完整联合校正可在原结果表中追溯。

依据：[HC 对比及联合校正][hc-results]、[计数与时长补充分析][optimization]。

### 5. ON–HC 与健康参考

- 整体 Coverage ILR：置换 P=0.6842，FD Holm P=0.6574。
- Coverage Energy：六项 Holm P=0.31587，FD Holm P=0.35816。
- ON 到 HC 健康参考中心的平均距离比 OFF 缩短约 0.124，但 嵌套 95% CI 为 [-0.191, 0.412]，Holm P=0.21938。
- ON–HC 不显著不等于统计等效；目前没有预先定义且有外部依据的等效界值，TOST 未执行。
- 因此目前尚不能声称 DBS-ON 已使微状态组成恢复正常。

### 6. 方法学与设计限制

- 完整模板重训时，M3 状态匹配率为 35.6%，M4 为 72.5%；这表示重训后状态身份可识别比例，不是原始扫描中状态出现比例。
- 高噪声模拟中所有候选时间正则化参数均未达到预设噪声校准门槛；A3 的 S5 高噪声误报为 42/200，Wilson 上界 27.16%。
- 可核实的 24 对患者均为 ON 先扫描、OFF 后扫描，间隔中位数约 109.98 分钟；刺激状态与扫描顺序重合，无法单独估计两者影响。
- 用药时间未完整核实。
- UPDRS-III 评分无法可靠匹配到对应影像访视，临床相关性分析目前 BLOCKED。
- 配对分析的 2,000 次 Bootstrap 条件于冻结模板，不包含重新学习空间模板的不确定性。
- 只保留 24 对扫描顺序已核实的患者，仍不能分离 DBS 条件与扫描顺序的影响。

依据：[稳定性复核][stability]、[优化与噪声报告][optimization]、[最终解释及质量审计入口][interpretation]。

## 当前结论与下一步

- **目前较明确的 ON–OFF 发现：**
  - M1 GEV 增加。
  - M4 Mean Duration 缩短。
  - 两项主要终点通过主要家族 Holm、FD 和 DVARS 敏感性调整。
- **次要发现：**
  - M1 Occurrence 和 Global Switching Rate 增加，但 DVARS 调整后未通过 0.05。
  - 整体 Transition Matrix 和 Coverage ILR 存在配对变化；具体单条转移路径需谨慎解释。
- **OFF–HC：**
  - 整体 Coverage 组成差异及部分 M3/M4 特征有统计支持。
- **尚未建立：**
  - ON 与 HC 等效。
  - DBS 已使微状态正常化。
  - 微状态变化与 UPDRS 改善相关。
  - 这些结果已在独立队列重复验证。

### 下一步工作

- [ ] 补齐逐人 UPDRS-III 与扫描访视的准确匹配，开展微状态变化与临床改善的关联分析。
- [ ] 后续采集平衡或随机化 ON/OFF 扫描顺序的数据，并记录用药及扫描时间。
- [ ] 在独立队列中预先锁定 M1 GEV、M4 Duration 两项终点与检验家族；预处理敏感性另行报告。
- [ ] 继续评估 A3 高噪声误报及 M1–M4 模板身份可重复性。
- [ ] 对 Transition Omnibus 及路径级结果进行预注册式重复验证。
- [ ] ON–HC 作为健康参考/等效性问题处理，不把“不显著”解释为“已恢复正常”。

## 附录：从单次 BOLD 扫描到统计结论

1. 提取 AAL78 的 ROI 时间序列，保留原始 TR 时钟与有效帧信息。
2. 沿时间标准化各 ROI，逐帧空间去均值，得到幅度及单位空间图。
3. 在训练折内汇集参与者加权的 GFP 峰图，学习四态模板。
4. 匹配五折状态编号，在留出扫描的共同窗口内做 Potts 动态规划。
5. 由最终标签计算 Coverage、Duration、Occurrence、GEV、Dispersion、Transition 与切换率。
6. 以参与者为统计单位，进行 HC 组间比较及 ON–OFF 配对检验；分别做 FD、DVARS 敏感性。
7. 按完整家族校正，结合效应、区间、模板稳定性和采集设计给出解释。

## 主要报告与代码来源

本报告仅综合既有分析，没有重跑原始影像或改变已归档数值。以下链接均固定到归档提交，方法编号只用于追溯来源。

- [冻结分析流程][workflow]；[锁定统计配置][protocol]。
- [ROI 标准化、幅度与峰值准备][prepare]；[模板学习与动态规划实现][models]。
- [共同窗口、指标与 ILR 的直接实现][inputs]。
- [统计设计与多重比较方法][stats]；[HC 对比结果及联合校正][hc-results]。
- [模板重训与稳定性][stability]；[计数、时长和噪声补充分析][optimization]。
- [ON–OFF 配对结果][paired]；[整体转移与具体路径][transition]；[健康参考分析][normalization]。
- [科学解释与研究限制][interpretation]。
- [GitHub 数学公式语法说明](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions)。

[workflow]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/workflow/FROZEN_WORKFLOW.md
[protocol]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/configs/protocol.json
[prepare]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/workflow/source/V3_BOLD_Microstate_Method_Optimization/scripts/models_v3.py
[models]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/workflow/source/V4_BOLD_Microstate_Reliability/scripts/v4_models.py
[inputs]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/scripts/micro_inputs.py
[stats]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/reports/Statistical_Methods.md
[hc-results]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/reports/V4_Microstate_Statistical_Results.md
[stability]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V4_Microstate_Statistical_Significance/reports/Microstate_Stability_Report.md
[optimization]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V5_Microstate_Nonsignificant_Endpoint_Optimization/reports/V5_Microstate_Optimization_Report.md
[paired]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_ON_OFF_Microstate_Results.md
[transition]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Transition_Dynamics.md
[normalization]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Normalization_Analysis.md
[interpretation]: https://github.com/vanor-lavigine/WEiDA-DBS-Reproduction/blob/17d3d1261b223da37280238e615c3176adaef684/V6_DBS_ON_OFF_Microstate_Dynamics/reports/V6_Final_Scientific_Interpretation.md
