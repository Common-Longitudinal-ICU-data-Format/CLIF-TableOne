# Output Reference

A complete map of every artifact under `output/final/`, with the cohort filter that produced it and the module that generates it.

For the **definitions** of each cohort and stratification flag, see [README.md §8](README.md#8-cohort-definitions).

---

## Top-level layout

```
output/final/
├── overall/                Critical-illness cohort         (icu_enc OR death_enc)
├── overall_ward/           Ward cohort, only if --ward     (ward_enc)
├── strata/                 Stratified critical-illness subsets
│   ├── icu/                                                (icu_enc)
│   ├── advanced_resp/                                      (high_support_enc)
│   │   ├── icu/                                            (high_support_icu_enc)
│   │   └── no_icu/                                         (high_support_no_icu_enc)
│   ├── nippv_hfnc/                                         (nippv_hfnc_enc)
│   │   ├── icu/                                            (nippv_hfnc_icu_enc)
│   │   └── no_icu/                                         (nippv_hfnc_no_icu_enc)
│   ├── vaso/                                               (vaso_support_enc)
│   │   ├── icu/                                            (vaso_icu_enc)
│   │   ├── no_icu/                                         (vaso_no_icu_enc)
│   │   ├── ed_icu/                                         (vaso_ed_icu_enc)
│   │   └── ed_ward/                                        (vaso_ed_ward_enc)
│   ├── no_imv/                                             (no_imv_enc)
│   │   ├── icu/                                            (no_imv_icu_enc)
│   │   └── no_icu/                                         (no_imv_no_icu_enc)
│   └── deaths/                                             (death_enc)
├── validation/             Data quality assessment (cohort-agnostic — runs on the raw CLIF tables)
├── stats/                  Per-stratum collection-coverage tables
├── meta/                   Execution reports, workflow logs, run metadata
└── configs/                Snapshot of the config files used for the run
```

Each cohort directory under `overall/`, `overall_ward/`, and `strata/<name>/` shares the same internal layout. The next section explains it once.

---

## Per-cohort artifacts

The same six subdirectories appear under every cohort/stratum directory. The "Cohort filter" column says which encounter blocks the data is restricted to **before** the artifact is computed.

| Subdirectory | Contents | Cohort filter | Generator |
|---|---|---|---|
| `tableone/` | Demographics, mortality, comorbidities, ventilator settings, medications, code status, SOFA, CCI, STROBE counts, upset data — see the per-file table below | Encounter blocks where the parent flag = 1 | `modules/tableone/generator.py` |
| `figures/` | CONSORT, sankey, venn, upset, area curves, ventilator-table PNG/HTML/PDF — see the per-file table below | Same as `tableone/` | `modules/tableone/generator.py` (HTML medication area curves come from `modules/medications/visualizer.py`) |
| `ecdf/{labs,vitals,respiratory_support}/*.parquet` | Empirical CDF — distinct `(value, probability)` pairs | Parent cohort/stratum **AND** value timestamp ∈ `[in_dttm, out_dttm]` where the time window depends on the stratum: ICU stay windows for `overall/`, `icu`, `deaths`; first vasopressor → discharge for `vaso` family; first qualifying device → discharge for `advanced_resp` and `nippv_hfnc` families. See `STRATUM_WINDOW_TYPE` in `generator.py`. | `modules/ecdf/generator.py:compute_ecdf_compact` |
| `bins/{labs,vitals,respiratory_support}/*.parquet` | Quantile bins with auto-extreme splitting (first/last bin split into 5 sub-bins when configured) | Same as `ecdf/` | `modules/ecdf/generator.py` via `modules/ecdf/utils.create_all_bins` |
| `summary_stats/*.{csv,json}` | Per-category mean/median/IQR for labs / vitals / meds / patient_assessments / CRRT | Critical-illness cohort (only generated under `overall/`) | `modules/mcide/collector.py` |
| `mcide/*.csv` | Minimum CDE value counts for categorical columns across CLIF tables | Critical-illness cohort (only generated under `overall/`) | `modules/mcide/collector.py:155` |

> **Note:** `mcide/` and `summary_stats/` are produced once for the critical-illness cohort and live under `overall/` only. They are not regenerated per stratum.

### What "cohort filter" means in plain English

- `overall/` → adults whose encounter block touched an ICU **or** discharged as `expired`/`hospice`. ECDF window: ICU stay.
- `overall_ward/` → adults whose encounter block touched a ward at any point
- `strata/icu/` → encounters in the critical-illness cohort whose `icu_enc == 1`. ECDF window: ICU stay.
- `strata/advanced_resp/` → encounters in the critical-illness cohort that ever received `imv` / `nippv` / `cpap` / `high flow nc`. ECDF window: first qualifying device `recorded_dttm` → `discharge_dttm`.
- `strata/advanced_resp/icu/` → above **AND** `icu_enc == 1`. ECDF window: same as `advanced_resp/`.
- `strata/advanced_resp/no_icu/` → above **AND** `icu_enc == 0` (these are by construction deaths). ECDF window: same as `advanced_resp/`.
- `strata/nippv_hfnc/` → encounters in the critical-illness cohort that ever received `nippv` (BiPAP) or `high flow nc` with `lpm_set >= 30`. ECDF window: first qualifying device `recorded_dttm` → `discharge_dttm`.
- `strata/nippv_hfnc/icu/` → above **AND** `icu_enc == 1`. ECDF window: same as `nippv_hfnc/`.
- `strata/nippv_hfnc/no_icu/` → above **AND** `icu_enc == 0` (by construction deaths). ECDF window: same as `nippv_hfnc/`.
- `strata/vaso/` → encounters in the critical-illness cohort that ever received `norepinephrine`, `epinephrine`, `phenylephrine`, `vasopressin`, `dopamine`, or `angiotensin`. ECDF window: first vasopressor `admin_dttm` → `discharge_dttm`.
- `strata/vaso/icu/` → above **AND** `icu_enc == 1`. ECDF window: same as `vaso/`.
- `strata/vaso/no_icu/` → above **AND** `icu_enc == 0` (by construction deaths). ECDF window: same as `vaso/`.
- `strata/vaso/ed_icu/` → encounters where the **first vasopressor was administered in the ED** and any subsequent ADT location includes ICU. ECDF window: same as `vaso/`.
- `strata/vaso/ed_ward/` → encounters where the **first vasopressor was administered in the ED** and any subsequent ADT location includes ward but **not** ICU. ECDF window: same as `vaso/`.
- `strata/no_imv/` → encounters in the critical-illness cohort that never received invasive mechanical ventilation (`on_vent == 0`). ECDF window: ICU stay.
- `strata/no_imv/icu/` → above **AND** `icu_enc == 1`. ECDF window: same as `no_imv/`.
- `strata/no_imv/no_icu/` → above **AND** `icu_enc == 0` (by construction deaths). ECDF window: same as `no_imv/`.
- `strata/deaths/` → encounters in the critical-illness cohort with `death_enc == 1` (`discharge_category in ('expired', 'hospice')`). ECDF window: ICU stay.

The flag definitions live in `modules/strata.py:24-41`, the inclusion code is at `modules/tableone/generator.py:1547-1549`, and the stratum-to-window mapping is at `modules/ecdf/generator.py:STRATUM_WINDOW_TYPE`.

---

## Worked example: `albumin_g_dL.parquet`

The user's original question — what exactly is in `output/final/overall/ecdf/labs/albumin_g_dL.parquet`?

It is a Polars-written parquet with two columns:

| Column | Type | Meaning |
|---|---|---|
| `value` | float | A distinct albumin value (g/dL) observed in the cohort |
| `probability` | float ∈ [0, 1] | The cumulative probability `P(X ≤ value)` |

The rows that go into this file are every albumin measurement (lab_category = `albumin`, unit = `g/dL`) where:

1. The patient is in the **critical-illness cohort** — adult, encounter block touched an ICU OR discharged as expired/hospice (`generator.py:1549`)
2. The lab `result_dttm` falls inside an ICU stay window for that patient — `in_dttm ≤ result_dttm ≤ out_dttm` for any row in `clif_adt` with `location_category == 'icu'` for that hospitalization (`modules/ecdf/generator.py:507-511`)
3. The value passes outlier handling defined in `modules/ecdf/config/outlier_config.yaml`

The same file under `output/final/strata/icu/ecdf/labs/albumin_g_dL.parquet` is the same computation but additionally restricted to encounters where `icu_enc == 1`. In the critical-illness cohort this just drops the small set of non-ICU deaths.

Under `strata/vaso/`, the temporal window is different: instead of ICU stay windows, it uses **first vasopressor `admin_dttm` → `discharge_dttm`**. So `strata/vaso/ecdf/labs/albumin_g_dL.parquet` contains albumin values drawn between the first vasopressor administration and discharge for patients who received vasopressors. Similarly, `strata/advanced_resp/` and `strata/nippv_hfnc/` use first qualifying device `recorded_dttm` → `discharge_dttm`.

There is **no 168-hour cap**. A 30-day ICU stay contributes all 30 days of albumin draws. The 24h/48h/72h numbers in `stats/collection_statistics.csv` are *coverage statistics on observation counts*, not the ECDF input filter (`modules/ecdf/statistics.py:196-200`).

---

## `tableone/` per-file detail

Generated by `modules/tableone/generator.py`. The same set of files is written under each cohort directory (`overall/tableone/`, `overall_ward/tableone/`, `strata/<name>/tableone/`); some files are skipped under `overall_ward/` and noted below.

| File | Contents |
|---|---|
| `table_one_overall.csv` | Main demographic + clinical Table One for the parent cohort |
| `table_one_by_year.csv` | Same, stratified by admission year |
| `mortality_rates.csv` | In-hospital and discharge mortality counts/rates |
| `strobe_counts.csv` | STROBE enrollment flow counts |
| `upset_data.csv` | Cohort subset membership (ICU, advanced resp, NIPPV/HFNC, vaso, death) for the upset plot |
| `comorbidities_per_1000_hospitalizations.csv` | Charlson/Elixhauser rates normalized per 1000 stays |
| `comorbidities_per_1000_hospitalizations_summary.csv` | Summary stats for the above |
| `code_status_counts_by_encounter_type.csv` | Code status value counts by encounter type |
| `code_status_percentages_by_encounter_type.csv` | Same as %s |
| `code_status_missingness_summary.csv` | Missing-data summary for code status |
| `code_status_combined_summary.csv` | Combined code status summary |
| `demographic_crosstab_race_ethnicity_sex.csv` | 3-way demographic crosstab |
| `sofa_mortality_summary.csv` | SOFA scores by mortality outcome (skipped under `overall_ward/`) |
| `hospice_trends_summary.csv` | Hospice admission trends over time |
| `cci_hospice_mortality_comprehensive_summary.csv` | CCI + hospice + mortality |
| `cci_mortality_hospice_trends_by_year_category_plotdata.csv` | Plot-ready CCI/hospice/mortality data by year |
| `ventilator_settings_by_device_mode.csv` | Ventilator parameters by device + mode (skipped under `overall_ward/`) |
| `ventilator_settings_counts_by_device_mode.csv` | Observation counts by device + mode |
| `ventilator_settings_total_observations.csv` | Total ventilator observations |
| `tidal_volume_volume_control_modes.csv` | Tidal volume stats (volume-control modes only) |
| `tidal_volume_volume_control_modes_mean_sd.csv` | Same binned by hour with mean/SD |
| `pressure_control_pressure_control_mode.csv` | Pressure control stats (pressure-control modes) |
| `pressure_control_pressure_control_mode_mean_sd.csv` | Same binned by hour |
| `mode_proportions_first_24h.csv` | Ventilator mode breakdown in first 24 h of mechanical ventilation |
| `medications_hourly_data.csv` | Paralytic / sedative / vasoactive doses by hour-since-ICU-admit |
| `medications_summary_stats.csv` | Mean/median/IQR for the above |
| `pf_sf_summary_24h.csv` | Per-encounter PF/SF ratios in the first 24 h of respiratory failure onset. Only under `strata/advanced_resp/` and `strata/no_imv/` (with `_icu`/`_no_icu` suffixed variants). |
| `pf_sf_aggregate_stats.csv` | Aggregate PF/SF statistics (n, mean, sd, median, Q25, Q75) segmented by onset device. Same strata as above. |
| `adt_dwell_summary.csv` | Per `location_category`: total dwell hours/days, median stay hours with Q1/Q3, distinct encounters, number of stays. Overall cohort only. |
| `adt_event_capture.csv` | Per `(table, location_category)`: event counts, % of events, events per location-hour. Tables: labs, vitals, respiratory_support. Overall cohort only. |

**Additional rows in strata Table One CSVs:**

Certain strata table ones (`table_one_<stratum>_by_year.csv`) include extra rows that do not appear in the overall Table One:

| Row(s) | Appears in strata | Description |
|---|---|---|
| `Resp. device onset, n (%)` / `Pre-device LOS (days)` / `Post-device LOS (days)` | `advanced_resp`, `nippv_hfnc`, `no_imv` (+ `/icu`, `/no_icu` splits) | Time from admission to first respiratory device onset, and from onset to discharge. Only encounters with a detected onset are counted. |
| `28-day VFD (IMV encounters), n (%)` / `VFD, median [Q1, Q3]` | Any stratum with IMV encounters | 28-day ventilator-free days. VFD = 0 for death within 28 days (uses `death_dttm` or `discharge_dttm` when `discharge_category` is expired/hospice) or if still on IMV at day 28. Intermediate free days between reintubation episodes do not count. |
| `Norepinephrine equivalent (NEE), n (%)` / `Peak NEE` / `Median NEE` | `vaso`, `vaso/icu`, `vaso/no_icu`, `vaso/ed_icu`, `vaso/ed_ward` | Vasopressor intensity per encounter in norepinephrine-equivalent mcg/kg/min. Peak = maximum concurrent intensity; Median = typical intensity. Weights: norepinephrine 1.0, epinephrine 1.0, phenylephrine 0.1, dopamine 0.01, vasopressin 2.5, angiotensin 10.0. Concurrent doses aligned by rounding to nearest hour. |
| `Time to ICU after first pressor (hours)` / `Time to Ward after first pressor (hours)` | `vaso/ed_icu`, `vaso/ed_ward` (respectively) | Hours from first vasopressor `admin_dttm` (in ED) to the first post-pressor ICU or ward `in_dttm`. Reported as median [Q1, Q3]. |

---

## `figures/` per-file detail

Generated by `modules/tableone/generator.py` (with HTML medication curves from `modules/medications/visualizer.py`). The same files appear under each cohort directory; ventilator/SOFA/medication-from-ICU plots are skipped under `overall_ward/`.

| File | Contents |
|---|---|
| `consort_flow_diagram.png` | CONSORT enrollment flow chart |
| `cohort_intersect_upset_plot.png` | UpSet plot of cohort overlaps (ICU, resp, NIPPV/HFNC, vaso, death) |
| `venn_all_4_groups.png` | 4-way Venn of cohort intersections |
| `code_status_stacked_bar_with_missingness_excl_missing_cat.png` | Code status stacked bar |
| `comorbidities_per_1000_barplot.png` | Charlson/Elixhauser bar chart |
| `sankey_matplotlib_icu.png` | Sankey: ICU → outcomes |
| `sankey_matplotlib_high_o2_support.png` | Sankey: advanced respiratory support → outcomes |
| `sankey_matplotlib_high_o2_proc_other.png` | Sankey variant for procedural-only encounters |
| `sankey_matplotlib_vaso_support.png` | Sankey: vasopressor → outcomes |
| `sankey_matplotlib_vaso_proc_other.png` | Sankey variant for procedural-only encounters |
| `sankey_matplotlib_others.png` | Sankey: non-critical-illness encounters |
| `sofa_mortality_histogram.png` | SOFA score distribution by mortality |
| `tidal_volume_volume_control_modes.png` | Tidal volume trends (volume control) |
| `tidal_volume_volume_control_modes_mean_sd.png` | Same with mean/SD overlay |
| `pressure_control_pressure_control_mode.png` | Pressure control trends |
| `pressure_control_pressure_control_mode_mean_sd.png` | Same with mean/SD overlay |
| `mode_proportions_first_24h_vertical.png` | Ventilator mode proportions (vertical bar) for first 24 h |
| `hospice_mortality_combined_trends.png` | Hospice + mortality trends over time |
| `cci_mortality_hospice_comprehensive.png` | CCI + hospice + mortality 3-way analysis |
| `paralytic_area_curve_7d.html` | Interactive 7-day area curve: paralytic dose by hour |
| `paralytic_median_dose_by_hour.html` | Median paralytic dose by hour |
| `sedative_area_curve_7d.html` | Interactive 7-day area curve: sedative dose by hour |
| `sedative_median_dose_by_hour.html` | Median sedative dose by hour |
| `vasoactive_area_curve_7d.html` | Interactive 7-day area curve: vasoactive dose by hour |
| `vasoactive_median_dose_by_hour.html` | Median vasoactive dose by hour |
| `ventilator_settings_table.png` / `.pdf` | Ventilator settings summary table rendered as image + PDF |
| `pf_sf_comparison_overall_icu_noicu.png` | Box plot comparing PF/SF distributions across Overall/ICU/No-ICU splits. Only under `strata/advanced_resp/figures/` and `strata/no_imv/figures/`. |
| `adt_dwell_hours_by_location.png` | Bar chart of cumulative dwell hours per ADT `location_category`. Overall cohort only. |
| `adt_los_distribution_by_location.png` | Pre-aggregated box plot of per-stay durations by location. Overall cohort only. |
| `adt_event_capture_pct.png` | Grouped bar of % events (labs/vitals/respiratory_support) per location. Overall cohort only. |
| `adt_events_per_location_hour.png` | 3-panel instrumentation density (events/hour) per location per table. Overall cohort only. |

---

## `ecdf/` and `bins/` parquet schemas

### `ecdf/{labs,vitals,respiratory_support}/<name>.parquet`

| Column | Type | Meaning |
|---|---|---|
| `value` | float | Distinct numeric value |
| `probability` | float | Cumulative `P(X ≤ value)` |

Naming:
- **Labs:** `{lab_category}_{unit_safe}.parquet` — e.g. `albumin_g_dL.parquet`
- **Vitals:** `{vital_category}.parquet` — e.g. `heart_rate.parquet`
- **Respiratory support:** `{column_name}.parquet` — e.g. `fio2_set.parquet`

### `bins/{labs,vitals,respiratory_support}/<name>.parquet`

Quantile bins computed from the same filtered values. Schema matches the bin format produced by `modules/ecdf/utils.create_all_bins` — bin id, label, lower/upper edge, count, percentage.

---

## `summary_stats/` per-file detail

Generated by `modules/mcide/collector.py`. Each metric is written as both CSV and JSON. Lives under `overall/summary_stats/` only (not regenerated per stratum).

| File | Contents |
|---|---|
| `labs_summary_by_category.csv/json` | Mean/median/std/q25/q75/min/max per `(lab_category, unit)` |
| `vitals_summary_by_category_and_name.csv/json` | Same per `(vital_category, vital_name)` |
| `medication_admin_continuous_dose_by_category_and_unit.csv/json` | Continuous-medication dose stats per `(med_category, dose_unit)` |
| `medication_admin_intermittent_dose_by_category_and_unit.csv/json` | Intermittent-medication dose stats per `(med_category, dose_unit)` |
| `patient_assessments_summary_by_category.csv/json` | Numeric-assessment stats per `assessment_category` |
| `crrt_blood_flow_rate_overall.csv/json` + `_by_mode.csv/json` | CRRT blood flow rate, overall and by `crrt_mode_category` |
| `crrt_dialysate_flow_rate_overall.csv/json` + `_by_mode.csv/json` | CRRT dialysate flow rate, overall and by mode |
| `crrt_ultrafiltration_out_overall.csv/json` + `_by_mode.csv/json` | CRRT ultrafiltration output, overall and by mode |

---

## `mcide/` per-file detail

Generated by `modules/mcide/collector.py:155`. Each file is `{table}_{columns}_mcide.csv` and contains the value count of every distinct combination of the named columns. Lives under `overall/mcide/` only.

Tables represented (one or more files each, depending on which categorical columns are summarized):

`adt`, `clif_crrt_therapy`, `clif_microbiology_culture`, `code_status`, `hospitalization`, `labs`, `medication_admin_continuous`, `medication_admin_intermittent`, `patient`, `patient_assessments`, `position`, `respiratory_support`, `vitals`.

---

## `validation/`

Cohort-agnostic — runs on the raw CLIF tables, not the cohort-filtered views.

| Path | Contents | Generator |
|---|---|---|
| `validation/json_reports/<table>_dqa.json` + supporting CSVs | Per-table DQA results from clifpy (conformance, completeness, plausibility) | `clifpy` invoked from `run_analysis.py` |
| `validation/consolidated/consolidated_validation.csv` | One-row-per-table master status grid | `modules/reports/combined_report_generator.py` |
| `validation/consolidated/<table>_summary_summary.json` | Per-table summary stats | `modules/reports/combined_report_generator.py` |
| `validation/feedback/<table>_validation_response.json` | User-classified errors (Accepted / Rejected / Pending) saved from the web app | `server/routes/feedback_routes.py` |
| `validation/monthly_trends/*.csv` | Monthly admission / data-volume trends | `run_analysis.py` |
| `validation/pdf_reports/<table>_validation_report.pdf` | Per-table validation PDF | `modules/reports/` |
| `validation/pdf_reports/combined_validation_report.pdf` | All-tables-in-one PDF | `modules/reports/combined_report_generator.py` |

---

## `stats/`

| File | Contents | Generator |
|---|---|---|
| `collection_statistics.csv` | Per `(data_type, category, reference_unit)`: total stays, total observations, total distinct values, mean ICU LOS, whole-stay mean/median/IQR, **first 24h / 48h / 72h count distributions**. The cohort is the critical-illness cohort. | `modules/ecdf/statistics.py:421` |
| `collection_statistics_<stratum>.csv` | Same, restricted to each stratum (`icu`, `advanced_resp`, `advanced_resp_icu`, `nippv_hfnc`, `nippv_hfnc_icu`, `vaso`, `vaso_icu`, `vaso_ed_icu`, `vaso_ed_ward`, `no_imv`, `no_imv_icu`, `deaths`) | `modules/ecdf/statistics.py` |

> The 24h/48h/72h numbers here are **coverage stats on observation counts**, not the ECDF input filter. The ECDF/bins parquets cover the full time window for that stratum: ICU stay windows for `overall/`, `icu`, `deaths`; event-onset → discharge for `vaso`, `advanced_resp`, and `nippv_hfnc` families.

---

## `meta/`

| File | Contents | Generator |
|---|---|---|
| `tableone_execution_report.txt` | Per-step memory checkpoints, timing, status | `modules/tableone/runner.py` |
| `tableone_ward_execution_report.txt` | Same, ward-mode run (only present when `--ward` was used) | `modules/tableone/runner.py` |
| `ecdf_execution_report.txt` | ECDF generation summary + structure | `modules/ecdf/runner.py` |
| `unit_mismatches.log` | `(category, unit)` pairs in the data that aren't in `lab_vital_config.yaml` — flags coverage gaps | `modules/ecdf/generator.py` |
| `file_metadata.json` | Snapshot of `config.json` + `tables_path` for the run | `run_project.py` |
| `workflow_logs/workflow_execution_<timestamp>.log`, `workflow_execution_latest.log` | Full pipeline stdout/stderr per run | `run_project.py` |

---

## `configs/`

| File | Contents |
|---|---|
| `config.json` | Snapshot of the `config/config.json` used for the run (site name, tables_path, file_type, timezone) |
| `outlier_config.yaml` | Snapshot of the lab/vital outlier bounds applied during ECDF |
| `lab_vital_config.yaml` | Snapshot of the lab/vital bin definitions used for the bins parquets |

These are written so a downstream consumer (or a future you) can tell exactly which configuration produced a given `output/final/` directory.
