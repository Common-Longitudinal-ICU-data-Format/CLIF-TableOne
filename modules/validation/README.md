# CLIF Validation Module

## Overview

This module provides validation utilities for CLIF tables using clifpy's validation capabilities. Tables are loaded and validated using clifpy's `from_file()` method which handles schema validation automatically.

## Components

### 1. Schema Files (`schemas/`)

Contains YAML definitions for all CLIF tables (copied from clifpy):
- Column definitions (names, types, requirements)
- Permissible values for categorical columns
- Range specifications (e.g., vitals ranges)

**Example** (`vitals_schema.yaml`):
```yaml
columns:
  - name: vital_category
    data_type: VARCHAR
    required: true
    is_category_column: true
    permissible_values:
      - temp_c
      - heart_rate
      - sbp
      ...

vital_ranges:
  temp_c:
    min: 25.0
    max: 44.0
  heart_rate:
    min: 0
    max: 300
```

### 2. Schema Validator (`schema_validator.py`)

Core validation logic that mirrors clifpy's validation.

**Functions:**
- `load_schema(table_name)` - Load YAML schema
- `validate_dataframe(df, schema)` - Run all validations
- `validate_required_columns(df, schema)` - Check required columns exist
- `validate_data_types(df, schema)` - Check column types match schema
- `validate_categories(df, schema)` - Check categorical values are permissible
- `validate_ranges(df, schema)` - Check numeric values within ranges (vitals)

**Returns:** List of error dictionaries matching clifpy's format:
```python
[
    {
        'type': 'missing_required_column',
        'column': 'vital_value',
        'message': "Required column 'vital_value' is missing"
    },
    {
        'type': 'invalid_category_values',
        'column': 'vital_category',
        'invalid_values': ['invalid_vital'],
        'count': 5,
        'message': "Column 'vital_category' contains 1 invalid category value(s)"
    }
]
```

### 3. Polars Loader (`polars_loader.py`)

Memory-efficient data loading using Polars lazy evaluation and streaming.

## Usage

Tables are validated through the analyzer classes in `modules/tables/`. Each analyzer:
1. Loads the table using clifpy's `from_file()` method
2. Runs validation via the `validate()` method in `BaseTableAnalyzer`
3. Returns validation results in a standardized format

```bash
# Run validation on all tables
uv run python run_analysis.py --all --validate

# Run validation on specific tables
uv run python run_analysis.py --patient --hospitalization --validate
```
