# 微状态方法迁移到 STN-DBS BOLD：第一轮实验

版本：2026-11-21。此文件在读取本轮真实状态/临床结果前制定；属于预先约定的探索性分析方案，不称为注册报告或独立确认性验证。

## 1. 实验目的

复现 EEG 微状态方法中“空间模式提取—共同模板—全时点回填—持续时间/出现率/覆盖率”的结构，检验类似的状态—临床关系是否能在已有 STN-DBS BOLD 数据中呈现。

主问题是 **BOLD 状态停留时间是否与 OFF 症状严重程度相关**。同时探索同一患者 ON/OFF 的状态指标变化，以及指标变化是否与临床改善相关。本轮不以 WEiDA local VAR 的分解为主线。

方法命名为 **microstate-inspired BOLD spatial states**。BOLD 中所得 M1–M4 不是 EEG A–D，也不是恢复出的毫秒级电生理微状态。使用每帧 BOLD 空间模式的分析与 CAP 方法接近。

## 2. 原始 EEG 参照与假设边界

- Li et al. (2021), *Dysfunctional Brain Dynamics of Parkinson's Disease and the Effect of Acute Deep Brain Stimulation*, DOI [10.3389/fnins.2021.697909](https://doi.org/10.3389/fnins.2021.697909)：研究 STN-DBS 患者的 EEG OFF/ON；OFF 下部分状态的持续时间与 UPDRS-III 有正相关。不是所有状态在 ON 后都缩短，文章也没有证明每种指标变化均与治疗改善率相关。因此本实验把基线关联和治疗反应关联作为不同问题。
- Guo et al. (2024), *Analysis of microstate features for Parkinson's disease based on reliability validation*, DOI [10.1016/j.jneumeth.2024.110115](https://doi.org/10.1016/j.jneumeth.2024.110115)：报告 EEG D 持续时间与 UPDRS 正相关，C 出现率与 MoCA 负相关。公开摘要的量表名称是 UPDRS，不能自行当作 UPDRS-III。本轮不增加 MoCA 检验。
- 上传的 Michel & Brechet EEG 微状态材料是综述。其方法流程可以复现，但不能称为复制了该综述自身的单一患者实验。
- CAP 方法谱系：[Liu, Chang & Duyn, 2013](https://doi.org/10.3389/fnsys.2013.00101)。本实验的有符号球面聚类是明确声明的实现选择，不是对该原文全部实现的逐行复刻。
- [Pycrostates 文档](https://pycrostates.readthedocs.io/en/latest/microstates/)解释了 EEG 的忽略极性匹配。本实验主分支保留 BOLD 正负方向，不能宣称与 Pycrostates 的默认 EEG 模型数值等价。

没有同步 EEG 或独立空间对应依据时，即使 BOLD 第四类与症状相关，也不命名为 EEG D。横断面关联不能自动推出同一人的治疗变化关联。全部检验双侧，方向在结果中如实呈现。

## 3. 已知输入与当前实际执行状态

最终以 [2026-11-21 SyN9 专项报告](../../reports/2026-11-21-v100-syn9-stn-hc.md) 为准，开发期间核对到 main 提交 `2af5f9e24738e61d9da13fcc11b547f7294bd00b`：

- 69 次原始扫描（27 OFF、27 ON、15 HC），68 次通过预处理及 AAL78 技术验收。
- 完整配对后，**26 对 STN-DBS OFF/ON 与 15 名 HC，共 67 次扫描**进入既有 WEiDA 实验。另一对因 OFF 超时，不能把单独 ON 加入配对。
- 每扫描 **210×78 AAL78 皮层时序**，TR 2.1 秒；采用统一的 78 区名称/顺序。
- fMRIPrep 25.2.7 的无场图 SyN、9 自由度 BBR、MNI152NLin6Asym 2 mm；XCP-D 26.1.0 NIfTI 路线、36P、GSR、despike、0.01–0.08 Hz，无 FD 删帧/额外平滑。
- 已有相同样本的作者 MATLAB WEiDA 结果，可作后续同数据方法参照。不能把新 BOLD M2 直接当作原 WEiDA 状态 2。
- **直接复用这 67 次扫描的既有纳入清单和 OFF/ON 配对证据**，不重新要求用户提供已确认的身份。临床表仍需从对应访视记录读取；临床缺失不阻断状态提取或配对组效应。

较早 README 的 126-run / 4S 批次仅为历史背景，不是此次主队列。开发初期曾考虑该路径；在读取任何本轮真实微状态结果前，已依更新证据将主输入固定为上述 AAL78 SyN9 队列。代码保留其他图谱的技术描述能力，但只有本主输入进入 v1 推断。

与既有 WEiDA 比较时须说明：WEiDA 在共同上游后另做 0.04–0.07 Hz 分析带通及 Hilbert 相位，新实验使用原 0.01–0.08 Hz BOLD 空间幅度模式。因此差别同时包括表征和分析频带，不应归因于“只替换聚类算法”。

本次代码在开发环境的验证仅使用手算例子和显式标记的合成数据。此开发环境没有读取服务器上的真实 ROI 数值，也没有运行真实队列的状态或显著性分析。服务器报告必须另行记录实际完成状态。

不重跑 FreeSurfer/fMRIPrep/XCP-D。原始影像和既有导数只读。保留既有来源与 QC 标记，不为技术试跑新设人工复核门槛。

专项报告已提供掩膜/覆盖率和数值验收，同时说明局部影像仍需人工复核。本程序不会把计算成功改写为人工 QC 已通过。保留既有标记，当前推断属于附带这些限制的探索性结果。

## 4. 固定的第一轮设计

| 项目 | 预设 |
|---|---|
| 数据路径 | 同一 SyN9 批次、已用于 WEiDA 的 67 份 NIfTI 派生 AAL78 TSV |
| 技术 pilot 范围 | 先取少量已纳入 run 验证接口，再运行全部 67 份；不根据临床结果选扫描 |
| 科学范围 | 固定全部 78 个 AAL 皮层区及既有名称顺序，不新增 ROI 筛选，不把 STN 靶向等同于 STN 局部信号 |
| ROI 一致性 | 同一 ROI 标签集合，按共同列顺序对齐；不把缺失分区补 0，不逐人改变特征维度 |
| 特征标准化 | 各 ROI 沿时间 z-score（ddof=1），再对每帧做 L2 单位化；保存标准化参数 |
| 空间中心化 | 不额外执行每帧空间去均值/EEG 平均参考 |
| 主聚类 | signed spherical k-means；中心每轮也单位化；100 次初始值，最多 300 轮 |
| 随机种子 | 20261121 |
| K | K=4 是本轮预设探索性基线，不宣称最优或真实状态数 |
| 训练样本 | 全部有效帧；pilot 每 run 总权重相同；analysis 每位确认受试者总权重相同，DBS 的 ON/OFF 等分 |
| 时间操作 | 不做 GFP 峰选择；不新增滤波；不做时间平滑；不套用 EEG 的毫秒最短片段规则 |
| 回填 | 全部有效帧，最大有符号余弦；数值近零帧标记为 -1，形成序列断点 |
| 次级 K | 5、6、7 仅描述状态数量敏感性，不追加临床显著性筛选，不按最小 p 改选主 K |
| 极性对照 | `unsigned` 为可选方法对照；保存每帧方向。不用哪个分支显著来选择主方法 |

共同模板在 analysis 阶段只使用确认的 DBS 配对和 HC，不混入 tTIS。缺少身份表时，pilot 可以处理技术清单中的 run，但其混合队列模板不直接用于患者推断；身份确定后重新建立对应的分析模板。

主程序当前要求一个患者一个条件一个 run；多个 run/重复访视会写明排除原因，不静默挑选、不把重复记录当新患者。若实际确有多 run，应先按明确访视/采集证据选择本次分析，或另行实现并记录按总时长/总片段聚合的扩展。

## 5. 状态指标和边界

以一个 run 为单位，状态 k 的有效帧数为 n_k，片段数为 b_k，全部有效帧为 T：

- coverage = n_k / T。
- duration_sec = n_k × TR / b_k。
- occurrence_per_min = b_k / (T × TR) × 60。

由定义 `coverage = duration_sec * occurrence_per_min / 60`，这些指标不能作为三份独立生物学证据。

首尾片段按实际观察到的长度计入；其长度可能受到采集边界截断。状态不出现时 coverage、occurrence 为 0，duration 为 NA。没有有效观测时全部不可估计。无效帧和 run 边界不能连接为连续片段。

同时保存两种转移：逐帧转移含自转移；事件转移只记录离开一个状态后进入另一个状态。未观察到某源状态退出时，该行概率为 NA。第一轮转移仅描述，不额外进行 16 条转移的显著性筛选。

拟合目标、平均余弦和初始化稳定性按其实际定义命名，不称为 EEG GEV。多个初始化的稳定不等于跨被试/重测可靠性。

## 6. 预设统计问题

每个患者先得到 ON/OFF 指标，统计单位是患者。模板编号使用 M1–M4，不能基于 p 值重命名为某个 EEG 类别。

| 家族 | 问题 | 固定比较数与校正 |
|---|---|---|
| A | 4 状态 × coverage/occurrence/duration 的 ON−OFF | 12 项，家族内 Holm |
| B | 以上 12 项变化与临床总分改善 OFF−ON 的关系 | 12 项，家族内 Holm |
| C（更直接对应 EEG 问题） | OFF 的 4 个状态 duration 与 OFF 同期严重程度 | 4 项，家族内 Holm |

分别报告三个家族，不声称这等同于全 28 项的单一家族错误率控制。不得只对家族 A 显著的状态进行临床相关。

- A：患者差值双侧符号翻转，默认 10,000 次，Monte Carlo p=(extreme+1)/(B+1)。需要零假设下的交换性/对称性；固定 ON→OFF 顺序不会因此成为随机治疗因果证据。
- B/C：Spearman，患者间临床变量置换，双侧。B 的改善定义为 OFF−ON，正值表示改善。
- 临床量表使用一个明确总分终点；显式记录 UPDRS、UPDRS-III 或 MDS-UPDRS-III 的实际名称，不能混合。若有多个候选量表，显式指定 `--clinical-scale`。
- OFF 评分须属于对应扫描/刺激/用药条件。记录时间、访视及药物条件的证据放在元数据来源说明中，不用术前评分替代术后 OFF。
- 置信区间采用患者层 bootstrap，默认 2,000 次。报告每项实际患者数、差值/相关、区间、原始 p、校正 p，以及不可估计原因。
- 这些区间以本轮共同模板固定为条件，不包含重新估计模板的不确定性；未来需要通过患者级模板重训练或独立样本评估泛化。
- 不可估计项仍占预设比较族位置（校正计算以 p=1 占位，输出原始/校正值为 NA），不因缺失缩小家族。
- B/C 的临床对象限于有身份证据及临床分数完整的配对患者；C 不额外要求 ON 该状态必须出现。
- n<3、相关变量恒定等不可估计情况清楚报告，不能输出伪造的相关。

HC 参与共同模板与描述，但 v1 不进行 HC–OFF 额外显著性家族，因此本轮不声称“正常化”。以后如需此问题应单独预设。原文同一数据上的再分析仍属探索，不能以当前固定 K=4 擦除历史选择。

## 7. 元数据和实际路径

从已有 SyN9 67-scan 清单适配 `runs.tsv` 和对应元数据，保留其患者/条件证据。若现有导出索引具有 `roi_time_series/atlas/format/label/n_frames/n_regions`，可用 `index`；否则只写字段适配器，不能为了凑 schema 修改原始导数。`index` 生成的 run_id 是技术标识，须关联回既有的受试者映射，不能猜。

`runs.tsv` 至少包含 `run_id, roi_time_series, tr_sec, atlas, format`；主分析记录 `atlas=AAL78, format=nifti, tr_sec=2.1`。可加 `n_frames_expected=210, n_regions_expected=78`。图谱名称来自已确认的 AAL1ctx 78 区提取，不得将其他图谱改名冒充。

元数据必需字段：

`run_id, subject_id, cohort, condition, mapping_source, clinical_scale, clinical_score`

- cohort：DBS / HC / TIS。
- condition：DBS 用 ON / OFF，HC 用 HC。
- mapping_source：原始扫描记录、临床表、既有确认映射的具体来源。填写空白模板不能自动成为已确认数据。
- clinical_score 可空；没有量表不阻断 A，也不阻断提取状态。
- 日期、访视、用药、FD、DVARS、有效时长等可加列并保留。v1 不自动拟合大量协变量。

元数据优先直接复用已产生 26 对配对和 15 名 HC 的 SyN9/WEiDA 清单，原始来源追溯到服务器 10.249.25.105 的 PD_DBS_zcc 与既有映射；按此前授权只读。不得混入 PD_TI_zcc 的 tTIS 或 TCP 数据。临床评分找不到时具体报告缺失字段，仍完成技术与 ON/OFF 阶段。

头动是具体的待核验混杂：当前没有 FD 删帧，GSR/despike 不能等同于运动混杂已消失。保存已有 FD/DVARS 指标，在解释临床关联前检查其分布与对应关系。不要为了得到显著性临时换滤波、删帧阈值或图谱。若实施额外头动敏感性分析，应单列流程，缺口打断序列。

## 8. 运行方式

使用独立环境。`requirements.txt` 是兼容性下限；记录实际 Python/依赖版本，不声称这些下限构成锁定环境。

```bash
python -m pip install -r experiments/bold_microstates_v1/requirements.txt
python -m unittest discover -s experiments/bold_microstates_v1/tests -v
```

执行时先从 SyN9 专项任务的现有配置/日志定位 AAL78 TSV 和 67-scan 配对清单，建立规范化索引或直接生成 `runs.tsv`。以下是规范化索引场景；占位符须替换为服务器已知真实路径：

```bash
python experiments/bold_microstates_v1/run.py index \
  --index /ABSOLUTE/SYN9_AAL78/adapted_results.tsv \
  --out /ABSOLUTE/NEW_EXPERIMENT/input \
  --atlas AAL78 --format nifti --tr 2.1

python experiments/bold_microstates_v1/run.py fit \
  --runs /ABSOLUTE/NEW_EXPERIMENT/input/runs.tsv \
  --out /ABSOLUTE/NEW_EXPERIMENT/pilot_k4 \
  --mode pilot --k 4 --n-init 100 --seed 20261121
```

pilot 不要求临床信息。导入已经确认的 26 对及 15 名 HC 清单后，重新按患者等权拟合：

```bash
python experiments/bold_microstates_v1/run.py fit \
  --runs /ABSOLUTE/NEW_EXPERIMENT/input/runs.tsv \
  --metadata /ABSOLUTE/NEW_EXPERIMENT/metadata.verified.tsv \
  --out /ABSOLUTE/NEW_EXPERIMENT/analysis_k4 \
  --mode analysis --k 4 --n-init 100 --seed 20261121

python experiments/bold_microstates_v1/run.py stats \
  --fit-dir /ABSOLUTE/NEW_EXPERIMENT/analysis_k4 \
  --metadata /ABSOLUTE/NEW_EXPERIMENT/metadata.verified.tsv \
  --out /ABSOLUTE/NEW_EXPERIMENT/inference_k4 \
  --clinical-scale MDS-UPDRS-III \
  --permutations 10000 --bootstrap 2000 --seed 20261121
```

量表名称必须改为实际清单里的精确值；如果没有量表，可以不指定，配对组效应仍运行，临床项目标记缺失。AAL78 已是固定皮层范围，本轮不使用 `--roi-list` 再筛脑区。

输出目录必须为新目录，避免覆盖旧结果。第一轮稳定运行后，才以同一清单和 ROI 范围分别运行 K=5/6/7；这些只运行 `fit`。`stats` 对 pilot、unsigned 或非 K=4 输出不启用，避免悄悄增加显著性筛选。

## 9. 结果交付

程序输出：输入审计、实际 ROI 标签、模板矩阵、逐帧状态/相似度/方向、逐 run 指标、两种转移、初始化稳定性、概要图、输入哈希和模型参数；推断阶段输出配对审计和完整统计表。

服务器 Codex 应把以下内容回传私有仓库的 `reports/bold_microstates_v1/`：

1. 实际输入路径来源、run/subject/pair 分别的数量，AAL78/NIfTI 范围和每项排除原因汇总；与预期 67 扫描、26 对、15 HC 对照。
2. 算法参数、环境版本、命令和提交 SHA。
3. 模板及状态时间线示例、指标分布、初始化稳定性；不要把初始化稳定称为重测信度。
4. 若元数据齐全，完整 A/B/C 检验族的汇总（含阴性与不可估计项），配对图和临床散点图；明确尚未独立验证。
5. 身份/临床缺失时，具体列明缺哪些字段以及已完成的技术结果，不阻断已能完成的阶段。

原始影像、ROI 时序、患者级临床表、身份映射及敏感绝对路径清单留在服务器，不上传。报告可用去标识汇总，并明确 synthetic / real。不能用合成数据测试代替真实队列结论。
