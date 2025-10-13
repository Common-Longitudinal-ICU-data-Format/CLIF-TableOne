# 📊 How the 1k ICU Sample is Created

The **1k ICU sample** is a stratified random sample of 1,000 hospitalizations that touched the ICU at least once, proportionally representing each admission year. This smart sampling accelerates analysis of large CLIF datasets while maintaining temporal and statistical representativeness.

---

## 🚦 When & Where is the Sample Created?

The 1k sample is created and saved **automatically**, and can also be force-generated on demand. The process occurs at two main points in the system:

### 1. After ADT Table Analysis (`app.py:1011–1042`)
- **Trigger:** After ADT table finishes analyzing
- **Logic:**  
  ```python
  # After ADT analysis completes
  if table_name == 'adt' and analyzer.table is not None:
      if not sample_exists(output_dir):
          # Generate sample automatically
  ```

### 2. On-Demand During Full Report Generation (`app.py:358–404`)
- **Trigger:** When user runs a full report with the “Use ICU Sample” option on
- **Logic:**  
  ```python
  if st.session_state.get('use_sample_full_report', False):
      sample_filter = load_sample_list(output_dir)
      if sample_filter is None:
          st.info("📊 Creating 1k ICU sample from ADT and Hospitalization tables...")
  ```

---

## 🛠️ Step-by-Step Sample Creation Algorithm

The process has **4 main steps**:

### **Step 1: Identify ICU Hospitalizations (`get_icu_hospitalizations_from_adt`)**

- **Goal:** Find all hospitalizations that ever touched the ICU
- **How:**
  ```python
  icu_mask = adt_df['location_category'] == 'icu'
  icu_hosp_ids = set(adt_df[icu_mask]['hospitalization_id'].dropna().unique())
  ```
- **Input:** ADT table (pandas DataFrame)
- **Output:** Set of unique `hospitalization_id` that had at least one ICU stay

### **Step 2: Subset Hospitalization Table (`generate_stratified_sample`)**

- **Goal:** Filter the hospitalization table to ICU patients only
- **How:**
  ```python
  icu_hosps = hospitalization_df[
      hospitalization_df['hospitalization_id'].isin(icu_hosp_ids)
  ].copy()
  ```
- **Input:** Hospitalization table, and the ICU `hospitalization_id` set from Step 1
- **Output:** ICD-only hospitalization table

### **Step 3: Stratified Proportional Random Sampling by Year**

- **Goal:** Pick 1,000 hospitalizations spread across years, proportional to each year’s frequency
- **Algorithm:**
  ```python
  icu_hosps['admission_year'] = pd.to_datetime(icu_hosps['admission_dttm']).dt.year
  year_counts = icu_hosps.groupby('admission_year').size()
  year_samples = (year_counts / total_icu * sample_size).round().astype(int)
  np.random.seed(42)
  for year, n_samples in year_samples.items():
      year_hosps = icu_hosps[icu_hosps['admission_year'] == year]['hospitalization_id'].values
      sampled = np.random.choice(year_hosps, size=n_samples, replace=False)
      sampled_ids.extend(sampled.tolist())
  ```
- **Key Features:**
  - 🗂 **Stratification:** Each admission year is proportionally represented
  - 📈 **Proportional sampling:** (Example: If 2020 = 30% of ICU stays, it contributes ~300 to the sample)
  - 🎲 **Random selection:** From each year, picked at random
  - 🔒 **Reproducibility:** Fixed random seed (`np.random.seed(42)`)
  - ➕ **Rounding adjustment:** Ensures exactly 1,000 in the sample

### **Step 4: Save for Reuse (`save_sample_list`)**

- **Goal:** Write sample hospitalization IDs (with metadata) to disk for future reuse
- **How:**
  ```python
  sample_file = output_dir / 'intermediate' / 'icu_sample_1k_hospitalizations.csv'
  # Add header comments with metadata
  metadata = [
      "# ICU Sample - 1k Hospitalizations (Stratified Proportional by Year)",
      f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
      f"# Total Count: {len(hosp_ids):,}",
      "# Sampling Method: Stratified (proportional by admission year)"
  ]
  ```
- **File:** `output/intermediate/icu_sample_1k_hospitalizations.csv`
- **Format:** CSV; single column `hospitalization_id`; 1,000 rows  
- **Metadata:** Top-of-file commented lines give creation info/method

---

## 🔍 How is the Sample Used in Analysis?

### **Loading the Sample (`load_sample_list`)**
- **How:**
  ```python
  df = pd.read_csv(sample_file, comment="#")
  hosp_ids = df['hospitalization_id'].dropna().astype(str).tolist()
  ```

### **Running Table Analysis with the Sample**
- **Pattern:** When loading tables, applying the `sample_filter` sends only those ICU sample IDs to clifpy
- **Example (`vitals_analysis.py`):**
  ```python
  if sample_filter is not None:
      self.table = Vitals.from_file(
          data_directory=self.data_dir,
          filetype=self.filetype,
          timezone=self.timezone,
          output_directory=clifpy_output_dir,
          filters={'hospitalization_id': list(sample_filter)}
      )
  ```
- **What this means:**
  - The sample acts as a fast, memory-efficient filter—clifpy loads **only the rows** included in the ICU sample set
  - **Impact:** Greatly speeds up and reduces resource usage for analysis on large datasets

---

