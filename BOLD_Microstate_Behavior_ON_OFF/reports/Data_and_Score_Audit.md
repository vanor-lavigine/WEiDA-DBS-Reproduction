# 数据与人工评分条目核查

确认 STN 临床配对 27 对，冻结影像配对 25 对，患者与两扫描键精确连接 25 对，日期核实 22 对，固定质量窗匹配 25 对。

第一评分文件的实际纳入患者工作表按双层表头确认 33 个人工条目。Q:R、AA:AB、AE 为预测列，AI 为全身自发动作人工评分。AT 只用于对账。第二评分文件复核同一身份和访视的人工条目及旧派生结果。人口学工作簿与既有身份链确认 STN 靶点。

两份评分表的 54 次 STN 访视逐项一致。旧 AT 在 7 人的 14 次访视中多计 4–15 分，差额全部来自排除的预测列。第二评分文件的旧运动徐缓小计混入肢体强直，49 次访视与重算值不同。主样本中 6 人震颤 OFF 为零，仅该改善率留空。

原表运动检查总分，33 项重算，正式版本待核实。人工条目范围 0–4，分值增大对应运动损害加重。步态与平衡仅为步态、步态冻结、姿势平稳度之和。分析未按工作表名称或这三项划分患者亚型。

[评分方向参考](https://mds.movementdisorders.org/UserFiles/New%20UPDRS%207%203%2008%20final.pdf) · [条目字典](../results/Score_Item_Dictionary.csv) · [既有评分核查](../../BOLD_Microstate_STN_Original_Scores/reports/STN_Original_Score_Audit.md)

## 临床来源核查汇总

```json
{
  "score_version": "原表运动检查总分，33项重算，正式版本待核实",
  "score_direction": "higher_is_worse",
  "score_direction_reference": "https://www.movementdisorders.org/MDS-Files1/Resources/PDFs/MDS-UPDRS.pdf",
  "improvement": "OFF minus ON; positive values indicate lower severity at ON",
  "improvement_rate": "(OFF minus ON) divided by OFF; no value when OFF equals zero",
  "scoring_workbook": {
    "file": "PD开关机fMRI评分_202011_gr_20201228.xlsx",
    "sheet": "开关机fMRI实际纳入患者.UPDRS-III",
    "read_range": "A3:AT128",
    "stn_visits": 54,
    "stn_all_33_items_valid": 54,
    "stn_medication_by_visit": {
      "med_off": 54
    }
  },
  "comparison_workbook": {
    "file": "PD开关机fMRI评分_20201218改善率..xlsx",
    "sheet": "开关机fMRI实际纳入患者.UPDRS-III(原始数据)",
    "read_range": "A4:AP113",
    "stn_old_y_mixes_limb_rigidity_count": 49,
    "stn_old_y_formula_includes_limb_rigidity_count": 54
  },
  "demographics": {
    "file": "PD-开关机人口学及刺激参数_2023.07.31(1).xlsx",
    "sheet": "PD-DBS开关机fMRI患者总名单",
    "header_rows": [
      2,
      3
    ],
    "rows_read": [
      4,
      87
    ],
    "columns_read": "A:R",
    "merged_ranges": 171,
    "participants_by_target": {
      "GPi": 33,
      "STN": 31,
      "mixed_or_asymmetric": 2,
      "other": 1
    },
    "exact_names_and_study_codes": 26,
    "adjudicated_name_variants": 1
  },
  "reconciliation": {
    "records_by_join_status": {
      "matched": 94,
      "primary_missing_visit_key": 16,
      "secondary_only": 30,
      "secondary_missing_visit_key": 2
    },
    "primary_stn_records": 54,
    "stn_records_matched_exact_key": 54,
    "stn_records_exact_33_clinical_items": 54,
    "stn_records_exact_all_derived_domains": 54,
    "stn_records_equal_target": 54,
    "stn_at_cached_total_available": 54,
    "stn_at_formula_includes_ai_rows": 20,
    "stn_at_stored_values": 34,
    "stn_at_total_mismatch": 14,
    "stn_at_affected_people": 7,
    "stn_at_mismatch_by_dbs": {
      "OFF": 7,
      "ON": 7
    },
    "stn_at_mismatch_min": 4.0,
    "stn_at_mismatch_max": 15.0,
    "stn_at_mismatch_explained_by_own_ai_sum": 14,
    "stn_at_mismatch_with_formula_including_ai": 14,
    "unmatched_secondary_blank_target_records": 16
  },
  "cohorts": {
    "confirmed_stn_clinical_pairs": 27,
    "matched_imaging_pairs": 25,
    "date_verified_pairs": 22,
    "matched_pair_visit_statuses": {
      "verified_identity_condition_visit": 22,
      "documented_visit_date_conflict": 2,
      "verified_identity_condition_visit_insufficient": 1
    },
    "unmatched_scoring_pair_reasons": {
      "outside frozen cohort; full pair absent from the verified AAL transfer": 1,
      "outside frozen cohort; archived AAL series contains entirely zero ROI columns": 1
    },
    "archived_v8_pairs_and_scores_exact": 25,
    "all_clinical_pairs_zero_off_for_rate": {
      "total": 0,
      "rigidity": 0,
      "bradykinesia": 0,
      "tremor": 6,
      "gait_balance": 0
    },
    "matched_pairs_zero_off_for_rate": {
      "total": 0,
      "rigidity": 0,
      "bradykinesia": 0,
      "tremor": 6,
      "gait_balance": 0
    }
  }
}
```

## 影像源与质量核查汇总

```json
{
  "baseline": {
    "method": "A3",
    "atlas": "AAL78",
    "k": 4,
    "tr": 2.1,
    "band_hz": [
      0.01,
      0.08
    ],
    "band": "F1",
    "normalization": "N0",
    "learner": "G1",
    "decoder": "S1",
    "lambda": 0.04
  },
  "paired_patients": 25,
  "patient_scans": 50,
  "folds": 5,
  "states": [
    "M1",
    "M2",
    "M3",
    "M4"
  ],
  "duration_unit": "s",
  "duration_definition": "2.1 seconds times mean episode length in TR; gaps split episodes, window-boundary episodes are retained",
  "original_source": "V6 Frozen_Cohort_Features.csv, verified against frozen aligned A3 labels",
  "quality_source": "V4 A3 fd_censor archive durations, verified against the V10 fd_censor_s1 replay",
  "quality_rule": {
    "fd_strictly_greater_than_mm": 0.5,
    "exclude_preceding_tr": 1,
    "exclude_following_tr": 2,
    "independent_blocks_for_s1": true
  },
  "max_duration_error_seconds": {
    "original_window": 7.105427357601002e-15,
    "fd_censor_s1": 0.0
  },
  "complete_duration_pairs": {
    "original_window": {
      "M1": 25,
      "M2": 25,
      "M3": 25,
      "M4": 25
    },
    "fd_censor_s1": {
      "M1": 25,
      "M2": 25,
      "M3": 25,
      "M4": 25
    }
  },
  "number_of_short_roi_scans": 3,
  "archived_inputs_rechecked": 195,
  "input_fingerprints": {
    "files": 199,
    "sha256": "4fe2fac82c13edefef302942635ead70d7a6e685878a210ac2e905a4de39733a"
  },
  "source_sha256": {
    "Frozen_Cohort_Features.csv": "0b396250993ee8086fbe4dc0830e2e9047b5904071379c407cce82ea649e0897",
    "ON_OFF_Pair_Manifest.csv": "bb4eb24358b2864399c4f4eee5a9b00e06a9ac11cd7f7839bacaec18d9b3cbe1",
    "V4_Scan_Metrics.csv": "b3768e0f66ed7255d76855fabad90902d29979a5668ea625817fd097a4eec822",
    "V4_Maps.npz": "f63d41cd8d8e95dd5c481b5b2cf69cea051887d9ef128edf9251c73140c47a8f",
    "v6_protocol_sha256": "2d34f042882a82933f5e07aa0545a8b6991ea2fdf5b278c2b79c73ecde93da1e"
  },
  "aligned_g1_mapping_sha256": "3486639ebefd81203b9518b4532e82417df2dcc168501c389fcc241f53b174e7",
  "scan_order": {
    "verified_pairs": 24,
    "on_first": 24,
    "off_first": 0,
    "unknown_order": 1
  },
  "archived_s1_label_replays_exact": 50,
  "v4_censored_window_replays_exact": 50,
  "censored_blocks_replayed_independently": 228,
  "max_archived_gfp_abs_error": 0.0,
  "max_g1_fit_correlation_abs_error": 0.0,
  "max_censored_v4_gev_abs_error": 5.551115123125783e-17
}
```
