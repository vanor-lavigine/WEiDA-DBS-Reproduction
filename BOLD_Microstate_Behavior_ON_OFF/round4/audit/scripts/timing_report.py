"""Present the visit audit without disclosing person-level evidence."""

import pandas as pd

from timing_rules import STATUSES


def evidence_inventory(facts):
    score, scans = (facts[key] for key in ("assessment_sources", "scan_sources"))
    entries = [
        ("Frozen cohort and pair manifests", "V6 patient-condition-scan keys and frozen HC keys",
         scans["frozen_patient_scans"] + scans["frozen_hc_scans"], "scan", "Fixes the audit denominator and linkage keys"),
        ("Original named acquisition roster", "Original demographic and acquisition archive; frozen key cross-check",
         scans["patient_identity_confirmed_scans"], "STN scan", "Identity and labeled DBS condition; roster dates are candidates"),
        ("Original DICOM header evidence", "Archived native acquisition date and time tags matched by series",
         scans["patient_dicom_timestamp_scans"], "STN scan", "Best available acquisition date and clock"),
        ("Original BIDS sidecar evidence", "Two archive captures of original sidecars",
         scans["patient_sidecar_clock_scans"], "STN scan", "Corroborates acquisition clock"),
        ("Archived timing workbook", "Derivative archive of acquisition evidence",
         scans["workbook_timestamp_crosschecks"], "STN scan", "Secondary cross-check only"),
        ("Original December 28 manual scoring workbook", "Manual 33 items and assessment date in source worksheet",
         score["raw_stn_rows_per_workbook"]["authoritative"], "STN assessment", "Identity, stimulation, assessment medication and date"),
        ("December 18 clinical comparison workbook", "Current file, original-items worksheet plus derived sheets",
         score["raw_stn_rows_dual_source_exact"], "STN assessment", "Cross-check relevant rows and items"),
        ("Archived manual rescoring and clinical linkage", "V8 raw audit and round-one frozen clinical linkage",
         score["frozen_score_rows_archived_33_items_and_total_exact"], "frozen STN assessment", "Checks score provenance; prior date labels reported separately"),
        ("Demographic and stimulation workbook", "Patient study-code/target and healthy acquisition roster worksheets",
         scans["frozen_hc_scans"], "frozen HC scan", "Corroborates cohort linkage and documents what the roster records"),
        ("HC assessment lookup", "Exact normalized identity lookup across original and archived scoring sources",
         scans["frozen_hc_scans"], "frozen HC scan", "Assessment, genuine zero, scale and visit evidence checked separately"),
        ("Prior behavioral audit and protocol", "Round-one audit, clinical paired results and Run_Manifest; round-two response profile",
         25, "STN pair", "Historical results and input provenance"),
    ]
    return pd.DataFrame(entries, columns=["source_role", "record_scope", "records_checked", "unit", "use_in_audit"])


def count(table, metric, condition=None):
    selected = table.loc[table.metric.eq(metric)]
    if condition is not None:
        selected = selected.loc[selected.condition.eq(condition)]
    return int(selected.n.item())


def interval_paragraph(orders):
    hours = orders.OFF_minus_ON_hours.dropna()
    if hours.empty:
        return "没有可计算的扫描间隔。"
    median, low, high = hours.median(), hours.min(), hours.max()
    return (f"{len(hours)} 对可计算 OFF 减 ON 的采集时间间隔，中位数 {median:.3f} 小时"
            f"（{median * 60:.2f} 分钟），范围 {low:.3f}–{high:.3f} 小时"
            f"（{low * 60:.2f}–{high * 60:.2f} 分钟）。")


def write_report(tables, facts, visits, groups, orders, association):
    links = tables["Clinical_Scan_Visit_Matching.csv"]
    order = tables["Scan_Order_Audit.csv"]
    hc = tables["HC_Clinical_Score_Audit.csv"]
    medications = tables["Medication_Stimulation_Audit.csv"]
    score, scans = (facts[key] for key in ("assessment_sources", "scan_sources"))
    n_patients = len(groups)
    n_assessments = len(visits)
    n_hc = int(hc.denominator.iloc[0])
    hc_evidence = facts["hc_assessment_sources"]
    hc_workbooks = [book for copies in hc_evidence["candidate_workbooks"].values() for book in copies]
    hc_sheets = sum(book["worksheets_checked"] for book in hc_workbooks)
    a, b, c = (int(groups.group.eq(label).sum()) for label in ("A", "B", "C"))
    both_same_day = int(groups[["ON", "OFF"]].eq(STATUSES[1]).all(axis=1).sum())
    hour_n = int(visits.date_difference_hours.notna().sum())
    days = visits.calendar_date_difference_days.dropna()
    status_rows = "\n".join(
        f"| {status} | {count(links, status, 'ON')} | {count(links, status, 'OFF')} | {count(links, status, 'both')} |"
        for status in STATUSES
    )
    if len(days):
        date_summary = (f"{len(days)}/{n_assessments} 次可比较日历日期。按评分日期减采集日期，"
                        f"{int(days.lt(0).sum())} 次为较早日期，{int(days.eq(0).sum())} 次同日，"
                        f"{int(days.gt(0).sum())} 次为较晚日期，中位数 {days.median():g} 天，"
                        f"范围 {days.min():g} 至 {days.max():g} 天。")
    else:
        date_summary = "评分和扫描之间没有可比较的日历日期。"
    if association is None:
        analysis = (f"A 组 {a} 人，未达到预定义关联所需的最低 3 对。本轮未计算同期关联，"
                    "也没有扩大样本或对 B 组另行检验。前三轮 25 人结果保留为历史评分的探索性分析。"
                    "临床变化或微状态变化本身无法建立两者的同期关联。")
    else:
        analysis = (f"在 {a} 名 A 组患者中复用预定义四态 Coverage 变化与五项相对运动改善率的双侧 Spearman，"
                    f"{int(association.passes_fdr.sum())}/20 项通过 BH 校正。"
                    "有效样本和逐项结果见 [同期访视关联表](Confirmed_Visit_Associations.csv)。")
    primary_comparison = (
        "比较工作簿与历史文件收据一致。" if score["comparison_archive_sha_match"] else
        "当前比较工作簿的字节哈希与历史 V8 收据不一致。逐项核对结果针对本轮可访问的版本。"
    )
    return f"""# STN-DBS 临床评分与 BOLD 扫描访视审计

覆盖冻结 A3 队列的 {n_patients} 名 STN 患者、{n_assessments} 次 ON/OFF 扫描及 {n_hc} 次 HC 扫描。原始来源中，A 组 **{a} 人**可确认两种条件的评分与扫描各自属于同一访视。B 组 **{b} 人**的身份、DBS 条件和评分来源可核实，至少一端缺少同次访视凭证。C 组 **{c} 人**的核心链接无法核实。

## 患者评分与扫描对应

| 原始记录支持的状态 | ON（次） | OFF（次） | 合计（次） |
|---|---:|---:|---:|
{status_rows}

两端均有同日证据而缺少明确访视连接的患者有 **{both_same_day}/{n_patients} 人**。本轮要求原始访视标识或原始记录直接关联，才确认同一次访视。未设置近同期小时阈值。

{date_summary}精确评分与采集小时差可计算 **{hour_n}/{n_assessments} 次**。日期精度依据原单元格格式和值核查。跨日评估可从日期确定先后，同日评估的先后无法从日期确定。日差分布及扫描间隔见[时间差汇总](Timing_Difference_Summary.csv)。

两份评分工作簿中核对了 {score['raw_stn_rows_dual_source_exact']} 条原始 STN 评估记录，身份、记录日期、刺激和药物条件以及全部 33 项人工评分逐项相符。冻结分析的 {score['frozen_score_rows_archived_33_items_and_total_exact']} 次评估与归档重算数值核对一致。辅助人口学表有 {score['demographic_name_discrepancy_patients']} 人姓名写法与主来源有差异，原始采集名册与两份评分原件的身份信息吻合。{primary_comparison}原表运动检查总分为 33 项重算，正式量表版本待核实。

原 V8 的日期匹配按日历日期核对。本轮检查了原始单元格批注，发现 {score['score_dates_reclassified_as_acquisition']} 条历史匹配记录的日期值明确指扫描日期。这条记录没有独立评估日期，归为日期不可用。评分来源和身份仍经核实。其他记录采用已匹配的 DICOM 采集日期对照评估日期。旧标签按评估次数列在[运行清单](Clinical_Timing_Audit_Manifest.json)。

## HC 临床评分

{count(hc, 'scan_identity_confirmed')}/{n_hc} 个冻结 HC 扫描键与原始采集名册、健康组登记匹配。健康组登记没有运动评分字段。查阅 {len(hc_workbooks)} 份工作簿的 {hc_sheets} 个工作表，包括原始条目、比较工作簿附表和既往归档，并核查归档临床导出。找到 {count(hc, 'motor_assessment_documented')}/{n_hc} 份可核验的 HC 运动评估及 {count(hc, 'measured_zero_verified')}/{n_hc} 个实测零分。与患者相同计分规则的可核实记录为 {count(hc, 'same_instrument_confirmed')}，评分与扫描同次访视的可核实记录为 {count(hc, 'same_visit_confirmed')}。缺失的 HC 临床分数以 NA 保存。此前所述的 15 个零分无可追溯的评估原件或赋值记录，HC 暂只作影像参照。

## 扫描顺序、刺激和药物

原始 DICOM 头字段归档支持 {scans['patient_dicom_timestamp_scans']}/{n_assessments} 次患者扫描的采集日期与时刻，其中 {scans['dicom_date_disagrees_roster_scans']} 次与名册候选日期有差异。{scans['dicom_bids_exact_clock_matches']} 次 DICOM 时刻与归档原始 BIDS sidecar 时刻一致。BIDS 时刻中的单数字秒按原数值解析。采用经过序列信息确认的 DICOM 日期，并在本地保存两者及其来源。HC 的 BIDS 时刻有 {facts['hc_scan_sources']['hc_sidecar_clock_scans']} 份可查，HC 名册候选日期尚无独立 DICOM 证据。

**{count(order, 'ON first')}/{n_patients} 对先 ON 后 OFF**，{count(order, 'OFF first')} 对先 OFF 后 ON，{count(order, 'unverified')} 对顺序未核实。{interval_paragraph(orders)}

原始评分记录显示 medication OFF 的评估为 ON 条件 {count(medications, 'score_medication_OFF', 'ON')} 次、OFF 条件 {count(medications, 'score_medication_OFF', 'OFF')} 次。两次扫描的药物状态均有记录、因而可比较的人数为 {count(medications, 'scan_medication_comparable', 'patient_pair')}/{n_patients}。原始采集与配对证据支持 DBS 条件标签。可核验的实际切换时刻、刺激状态稳定时长和扫描期间刺激状态记录分别为 {scans['patient_switch_documented_scans']}、{scans['patient_stabilization_documented_scans']}、{scans['patient_stim_continuity_documented_scans']} 次。扫描间隔只表示两次采集的时间差。固定扫描顺序限制 ON/OFF 差异的解释。分条件证据见[药物与刺激审计](Medication_Stimulation_Audit.csv)。

## 分析决定

{analysis}

基线完整性核验覆盖原始评分接收件、冻结 A3 影像特征及前三轮分析。扫描核验使用本地归档的原始采集与头字段。原采集服务器的无交互认证未通过，远端 DICOM 文件本轮未复读。既往采集档案核查中有 {scans['historic_source_coverage']['archived_review_unreadable_directories']} 个目录不可读。同期临床评估无法从可核查档案恢复，是本队列的设计限制。

## 核验材料

- [审计方案](ANALYSIS_PLAN.md)和[配置](configs/protocol.json)
- [评分扫描对应](Clinical_Scan_Visit_Matching.csv)、[HC 临床评分](HC_Clinical_Score_Audit.csv)、[扫描顺序](Scan_Order_Audit.csv)及[药物与刺激](Medication_Stimulation_Audit.csv)
- [证据清单](Timing_Evidence_Inventory.csv)和[运行清单](Clinical_Timing_Audit_Manifest.json)
- [既往评分核查](../../reports/Data_and_Score_Audit.md)、[既往临床配对](../../results/Clinical_Paired_Results.csv)、[第二轮临床变化](../../round2/results/Clinical_Response_Profile.csv)和[第三轮关联](../../round3/reports/ON_OFF_Coverage_Clinical_Rate_Report.md)

逐人 ON/OFF 连接、原始日期精度、源行定位、HC 核查和逐文件 SHA-256 收据保存在 Git 忽略的 `private/round4/audit/`。
"""
