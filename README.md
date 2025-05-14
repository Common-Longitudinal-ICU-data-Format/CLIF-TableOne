# CLIF-TableOne

# CLIF Table One Generator

Generate “Table One” summary (by year + overall) from CLIF parquet data files for all encounters with at least one ICU stay.

## Requirements

* Required table filenames should be `clif_patient`, `clif_hospitalization`, `clif_adt`, `clif_vitals`, `clif_labs`, `clif_medication_admin_continuous`, `clif_respiratory_support`, `clif_patient_assessments`
* Within each table, the following variables and categories are required.

| Table Name | Required Variables | Required Categories |
| --- | --- | --- |
| **patient** | `patient_id`, `race_category`, `ethnicity_category`, `sex_category`, `death_dttm` | - |
| **hospitalization** | `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission` | - |
| **adt** |  `hospitalization_id`, `hospital_id`,`in_dttm`, `out_dttm`, `location_category` | - |
| **vitals** | `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value` | weight_kg |
| **labs** | `hospitalization_id`, `lab_result_dttm`, `lab_category`, `lab_value` | creatinine, bilirubin_total, po2_arterial, platelet_count |
| **medication_admin_continuous** | `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit` | norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, angiotensin(optional) |
| **respiratory_support** | `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `tracheostomy`, `fio2_set`, `lpm_set`, `resp_rate_set`, `peep_set`, `resp_rate_obs`, `tidal_volume_set`, `pressure_control_set`, `pressure_support_set`, `peak_inspiratory_pressure_set`, `tidal_volume_obs` | - |
| **patient_assessments** | `hospitalization_id`, `recorded_dttm` , `assessment_category`, `numerical_value`| `gcs_total` |
| **crrt_therapy** | `hospitalization_id`, `recorded_dttm` | - |

## Configuration

1. Navigate to the `config/` directory.
2. Rename `config_template.json` to `config.json`.
3. Update the `config.json` with site-specific settings. 

Note: For multi-hospital sites that have created a `hospitalization_joined_id` to track patients transferring between hospitals, set `id_col` to `hospitalization_joined_id`. For single-hospital sites, use `hospitalization_id` as the `id_col`.

## Environment setup and project execution

The environment setup code is provided in the `generate_table_one.sh` file for macOS and `generate_table_one.bat` for Windows.

**For macOS:**

1. Make the script executable: 
```bash
chmod +x generate_table_one.sh
```

2. Run the script:
```bash
./generate_table_one.sh
```

**For Windows:**

1. Run the script in the command prompt:

```bat
generate_table_one.bat
```

## Output


