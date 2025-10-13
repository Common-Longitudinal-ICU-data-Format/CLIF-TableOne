# CLIF Table Implementation Checklist

This checklist provides a step-by-step task list for implementing new CLIF table analyzers.

**Reference**: See `dev/DEVELOPER_GUIDE.md` for detailed explanations and code patterns.

---

## **NEW TABLE IMPLEMENTATION TASK LIST**
*Copy this section for each new table you implement*


## **TABLE TRACKING**

### ✅ Completed Tables:
- [x] patient
- [x] hospitalization
- [x] adt
- [x] code_status
- [x] crrt_therapy
- [x] ecmo_mcs
- [x] hospital_diagnosis
- [x] labs
- [x] medication_admin_continuous
- [x] medication_admin_intermittent
- [x] microbiology_culture
- [x] microbiology_nonculture
- [x] microbiology_susceptibility
- [x] patient_assessments
- [x] patient_procedures
- [x] position
- [x] respiratory_support
- [x] vitals

### 🚧 In Progress:
- [ ]

### 📋 Planned:
- [ ]

---

### **Table Name: `___________`**

#### **PHASE 1: RESEARCH & VERIFICATION** ✅
- [ ] 1.1 Load table in test script using clifpy
- [ ] 1.2 Print and document actual columns (`df.columns.tolist()`)
- [ ] 1.3 Print and document actual data types (`df.dtypes`)
- [ ] 1.4 Check for datetime columns (verify they're actually datetime)
- [ ] 1.5 Review schema in `reference/schemas/` for reference only
- [ ] 1.6 Identify which similar analyzer to study (patient/hospitalization/adt/code_status/crrt_therapy)

#### **PHASE 2: CREATE ANALYZER CLASS** 🛠️
- [ ] 2.1 Create file: `modules/tables/{table_name}_analysis.py`
- [ ] 2.2 Import clifpy table class: `from clifpy.tables.{table_name} import {TableClass}`
- [ ] 2.3 Import BaseTableAnalyzer
- [ ] 2.4 Implement `get_table_name()` method
- [ ] 2.5 Implement `load_table()` with both file naming conventions
- [ ] 2.6 Implement `get_data_info()` based on ACTUAL columns
- [ ] 2.7 Implement `analyze_distributions()` for relevant columns
- [ ] 2.8 Implement `check_data_quality()` with table-specific checks

#### **PHASE 3: REGISTER IN CLI** 🔧
- [ ] 3.1 Add import to `modules/cli/runner.py` (line ~6)
- [ ] 3.2 Add to `TABLE_ANALYZERS` dict in `runner.py` (line ~14)

#### **PHASE 4: REGISTER IN CLI ARGS** 📋
- [ ] 4.1 Add `--{table_name}` argument in `run_analysis.py` (line ~88)
- [ ] 4.2 Add to validation check (line ~118)
- [ ] 4.3 Add to `--all` tables list (line ~127)
- [ ] 4.4 Add to conditional append (line ~129+)

#### **PHASE 5: REGISTER IN WEB APP** 🌐
- [ ] 5.1 Add import to `app.py` (line ~21)
- [ ] 5.2 Add to `TABLE_ANALYZERS` dict in `app.py` (line ~137)
- [ ] 5.3 Add to `available_tables` list (line ~227)
- [ ] 5.4 Add quality check definitions in `_get_quality_check_definition()` (line ~661)
- [ ] 5.5 Add summary display section if needed (line ~1700+)

#### **PHASE 6: UPDATE MODULE EXPORTS** 📦
- [ ] 6.1 Add import to `modules/tables/__init__.py`
- [ ] 6.2 Add to `__all__` list

#### **PHASE 7: CLI TESTING** 🧪
- [ ] 7.1 Run: `python run_analysis.py --{table_name} --validate --summary -v`
- [ ] 7.2 Verify table loads without errors
- [ ] 7.3 Verify validation completes
- [ ] 7.4 Verify summary generates
- [ ] 7.5 Check output files in `output/final/`

#### **PHASE 10: FINAL VERIFICATION** ✔️
- [ ] 10.1 Run `--all` flag includes new table
- [ ] 10.2 Run `git status` to review changed files
- [ ] 10.3 Verify no existing tables broke
- [ ] 10.4 Review all outputs look correct

---

**Files Modified (verify all 5):**
- [ ] `modules/tables/{table_name}_analysis.py` (NEW)
- [ ] `modules/tables/__init__.py`
- [ ] `modules/cli/runner.py`
- [ ] `run_analysis.py`
- [ ] `app.py`

---

## **CRITICAL RULES** ❌✅

### **NEVER:**
- ❌ Implement logic for columns that don't exist in actual data
- ❌ Skip verifying data structure first (Phase 1 is mandatory!)
- ❌ Skip CLI implementation (both interfaces required)
- ❌ Use direct datetime comparisons without error handling
- ❌ Hardcode paths or config values
- ❌ Modify cache_manager.py or feedback system without deep understanding

### **ALWAYS:**
- ✅ Verify actual data structure BEFORE implementing
- ✅ Use safe datetime: `pd.to_datetime(errors='coerce')` + try-except
- ✅ Handle both file naming conventions: `{table}.parquet` and `clif_{table}.parquet`
- ✅ Test both CLI and Web App interfaces
- ✅ Follow patterns from existing analyzers
- ✅ Add quality check definitions to app.py

---

## **EXAMPLE TABLES TO STUDY**

| Table Type | File | Use When |
|-----------|------|----------|
| Simple demographics | `patient_analysis.py` | Basic categorical distributions |
| Complex with dates | `hospitalization_analysis.py` | Date ranges, year distributions |
| Relationship tracking | `adt_analysis.py` | Location tracking, complex queries |
| Recent example | `ecmo_mcs_analysis.py` | Latest implementation patterns |

---

## **QUICK TEST COMMANDS**

```bash
# Test single table
python run_analysis.py --{table_name} --validate --summary -v

# Test all tables (verify new table included)
python run_analysis.py --all --validate --summary

# Start web app
streamlit run app.py
```

---


---

## **NOTES & LESSONS LEARNED**

1. **Always verify data structure first** - Schema documentation may not match reality
2. **Datetime handling is critical** - Always use `pd.to_datetime(errors='coerce')` with try-except
3. **File naming varies** - Check for both `table.parquet` and `clif_table.parquet`
4. **CLI registration is separate** - Don't forget `modules/cli/runner.py` registration
5. **Test both interfaces** - Web app and CLI must both work

---

*Last updated: 2025-10-09*
