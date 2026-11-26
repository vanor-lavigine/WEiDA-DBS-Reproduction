# V7 四个冻结微状态的功能网络注释

## 图谱重叠

以本项目实际使用的 2 mm AAL1ctx MNI 图谱为共用网格。读取作者发布的 Yeo 2011 MNI152 7 网络紧掩膜，通过 NIfTI 仿射在模板坐标中最近邻重采样。重叠按每个 AAL ROI 的全部体素作分母，允许一个 ROI 跨多个网络。未标注的皮层体素单列。两套图谱使用各自的 MNI 模板，重叠是基于坐标对照的近似估计。

冻结标签表有 78 个皮层 ROI，左右各 39 个。本图谱没有收录皮层下 ROI，皮层下类别在表中保留为零。78/78 个 ROI 与至少一个 Yeo 网络有体素重叠。61,614/151,885 个 AAL 体素落在七个网络内，占 40.6%。

以每个 ROI 的绝对权重与平方权重分别乘上该 ROI 的体素重叠比例。表中同时给出 ROI 在整个模板内的权重占比、各 ROI 在所属网络份额中的权重占比、各网络在全模板及 Yeo 已覆盖皮层中的权重占比，以及每个等效 ROI 的平均权重。等效 ROI 数是各 ROI 体素比例之和。

## 四个状态的空间分布

下表用折 0 冻结参考模板。前三个脑区按绝对权重排序。网络排序用 Yeo 已覆盖皮层中的平方权重，覆盖率用整个模板作分母。切面图的正负沿用原模板的整体极性约定，网络权重取绝对值或平方。

| 状态 | 绝对权重最高的三个 ROI | Yeo 网络占比前两位 | 被 Yeo 分配的平方权重 | 左 / 右平方权重 |
|---|---|---|---:|---:|
| M1 | Occipital_Sup_L、Lingual_R、Occipital_Sup_R | Visual 38.9%、Default 18.4% | 35.6% | 49.3% / 50.7% |
| M2 | Frontal_Med_Orb_L、Cingulum_Post_L、Frontal_Inf_Oper_R | Default 32.6%、Ventral Attention 14.7% | 39.9% | 57.0% / 43.0% |
| M3 | Frontal_Mid_L、Frontal_Sup_R、Frontal_Sup_L | Default 24.7%、Frontoparietal 15.3% | 40.9% | 48.2% / 51.8% |
| M4 | Angular_L、Frontal_Sup_Medial_L、Postcentral_R | Default 31.3%、Somatomotor 18.1% | 39.8% | 53.6% / 46.4% |

## 五折一致性

对参考折的 16 项跨折匹配有 16/16 项同时保留冻结编号并通过原 V4 的门槛。五折两两对照有 40/40 项通过相同条件。匈牙利匹配使用绝对余弦，得分至少为 0.60，行和列各自的次优差值至少为 0.05。参考折自身的四项记录单列。

| 状态 | 空间 \|r\| 范围 | ROI 绝对权重 r 范围 | Yeo 平方组成 r 范围 | ROI 绝对权重排名 ρ 范围 |
|---|---:|---:|---:|---:|
| M1 | 0.963 至 0.985 | 0.911 至 0.970 | 0.978 至 0.999 | 0.864 至 0.942 |
| M2 | 0.989 至 0.994 | 0.968 至 0.980 | 0.981 至 0.999 | 0.958 至 0.973 |
| M3 | 0.906 至 0.995 | 0.677 至 0.976 | 0.647 至 0.989 | 0.644 至 0.970 |
| M4 | 0.979 至 0.996 | 0.885 至 0.975 | 0.912 至 0.997 | 0.875 至 0.974 |

模板折之间的训练样本有重叠。上述数值描述冻结模板的空间一致性和图谱重叠，支持对模式分布作解剖描述。网络富集和疾病特异性没有在本轮接受推断检验。

图谱来源 [项目 AAL1ctx 原始文件](https://github.com/chenfei-ye/BIDS-fMRIpost/tree/c345a4a82685b67e5b578197927fe0dc90171efd/atlases) · [Yeo 2011 作者发布的图谱](https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011)。

[体素重叠表](../results/AAL78_Yeo7_Overlap.csv) · [逐折注释表](../results/Microstate_Network_Annotation.csv) · [逐折一致性](../results/Microstate_Network_Stability.csv) · [四态网络图](../figures/Figure_05_Microstate_Network_Maps.png) · [空间一致性图](../figures/Figure_06_Five_Fold_Template_Stability.png)
