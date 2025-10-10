# Developer Guide: Adding New CLIF Tables

This guide provides comprehensive instructions for adding new CLIF 2.1 table analyzers to the validation and summarization system.

## Reference Materials

When implementing a new table analyzer, use these resources:

### 1. CLIF Schema Definition
**Location**: `reference/schemas/`
- Complete CLIF 2.1 schema definitions
- Required columns for each table
- Data types and categorical value sets
- Use this to understand table structure and validation requirements

### 2. Project Objective
**Location**: `reference/goal.txt`
- Overall project goals and requirements
- Context for what the tool aims to achieve
- Helps understand the bigger picture

### 3. Validation Implementation Examples
**Primary References**:
- `legacy/generate_clif_report_card.py` - Original validation implementation patterns
- `code/clif_report_card.py` - Reference validation code with error handling
- Study these to understand validation logic and error classification

### 4. clifpy Documentation
**Source**: Context7 MCP server (`mcp__context7__get-library-docs`)
- Use for clifpy API reference
- Table loading patterns
- Built-in validation methods
- Example usage patterns
- you can also use `reference/clifpy.txt` to get the clifpy table class name

## Core Requirements

### Must-Have Features for Every New Table

1. **Dual Implementation**: Update BOTH interfaces
   - Web App: `app.py`
   - CLI: `run_analysis.py`

2. **Inherit from BaseTableAnalyzer**
   - Location: `modules/tables/base_table_analyzer.py`
   - Provides core validation and analysis framework

3. **Integrate with Feedback System**
   - Support user accept/reject decisions on validation errors
   - Properly classify errors as status-affecting vs informational
   - Maintain feedback persistence via `modules/utils/feedback.py`

4. **Follow Established UI Sections**
   - Validation tab with error classification
   - Summary tab with data overview, missingness, distributions
   - Data quality checks section
   - Maintain consistent layout and interaction patterns

5. **Support Persistent State/Caching**
   - Integration with `modules/utils/cache_manager.py`
   - Results persist across sessions
   - Support re-analysis workflow

6. **Error Classification**
   - **Status-Affecting Errors**: Impact validation status, require user review
     - Missing required columns
     - Data type mismatches (non-castable)
     - 100% missing values in required columns
     - Invalid categorical values in required columns
   - **Informational Issues**: For awareness only, don't affect status
     - Extra columns
     - Missing optional columns
     - Minor data quality observations

## Step-by-Step Implementation

### Step 1: Create Table Analyzer Class

Create `modules/tables/{table_name}_analysis.py`:

```python
from clifpy.tables.{table_name} import {TableClass}
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any
import pandas as pd

class {TableName}Analyzer(BaseTableAnalyzer):
    """Analyzer for {Table Name} table using clifpy."""

    def load_table(self):
        """Load {Table Name} table using clifpy."""
        self.table = {TableClass}.from_file(
            data_directory=self.data_dir,
            filetype=self.filetype,
            timezone=self.timezone,
            output_directory=self.output_dir
        )

    def get_data_info(self) -> Dict[str, Any]:
        """Get {table}-specific data information."""
        if self.table.df is None:
            return {'error': 'No data available'}

        # Return table-specific metrics
        return {
            'row_count': len(self.table.df),
            'column_count': len(self.table.df.columns),
            # Add table-specific metrics here
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze {table}-specific distributions."""
        if self.table.df is None:
            return {}

        # Analyze relevant categorical and numeric distributions
        return {
            # Add distribution analysis here
        }

    def check_data_quality(self) -> Dict[str, Any]:
        """Perform {table}-specific data quality checks."""
        if self.table.df is None:
            return {'error': 'No data available'}

        checks = {}

        # Add table-specific quality checks
        # Example pattern:
        # duplicate_ids = self.table.df.duplicated(subset=['id_column'])
        # checks['duplicate_ids'] = {
        #     'status': 'fail' if duplicate_ids.any() else 'pass',
        #     'count': int(duplicate_ids.sum()),
        #     'percentage': round(duplicate_ids.sum() / len(self.table.df) * 100, 2),
        #     'examples': self.table.df[duplicate_ids].head(10) if duplicate_ids.any() else None
        # }

        return checks
```

### Step 2: Register Table in Both Interfaces

#### A. Update `app.py`

Add to `TABLE_ANALYZERS` dict (around line 126):
```python
TABLE_ANALYZERS = {
    'patient': PatientAnalyzer,
    'hospitalization': HospitalizationAnalyzer,
    'adt': ADTAnalyzer,
    '{table_name}': {TableName}Analyzer,  # Add this line
}
```

Add to `TABLE_DISPLAY_NAMES` dict (around line 132):
```python
TABLE_DISPLAY_NAMES = {
    'patient': 'Patient',
    'hospitalization': 'Hospitalization',
    'adt': 'ADT',
    '{table_name}': '{Display Name}',  # Add this line
}
```

Import the analyzer (around line 18):
```python
from modules.tables import PatientAnalyzer, HospitalizationAnalyzer, ADTAnalyzer, {TableName}Analyzer
```

#### B. Update `run_analysis.py`

Add same entries to TABLE_ANALYZERS and TABLE_DISPLAY_NAMES dicts.

Add import statement at top of file.

Add command-line argument (in argument parser section):
```python
parser.add_argument('--{table_name}', action='store_true',
                   help='Analyze {table_name} table')
```

### Step 3: Update Module Exports

Edit `modules/tables/__init__.py`:
```python
from .{table_name}_analysis import {TableName}Analyzer

__all__ = [
    'BaseTableAnalyzer',
    'PatientAnalyzer',
    'HospitalizationAnalyzer',
    'ADTAnalyzer',
    '{TableName}Analyzer',  # Add this
]
```

### Step 4: Implement Table-Specific Logic

#### A. Data Info Method
Return metrics relevant to the table:
- Row count
- Column count
- Unique ID counts
- Date ranges
- Table-specific counts (e.g., unique locations for ADT)

#### B. Distribution Analysis
Analyze relevant columns:
- Categorical distributions (use `get_categorical_distribution()` from utils)
- Numeric distributions for key metrics
- Date/time patterns if applicable

#### C. Data Quality Checks
Implement checks specific to the table:
- Duplicate records
- Invalid date sequences
- Missing required values
- Business logic violations (e.g., discharge before admission)

Return format:
```python
{
    'check_name': {
        'status': 'pass' | 'warning' | 'fail',
        'count': int,
        'percentage': float,
        'examples': pd.DataFrame or None
    }
}
```

### Step 5: Add Quality Check Definitions

In `app.py`, add definitions to `_get_quality_check_definition()` function (around line 229):
```python
def _get_quality_check_definition(check_name: str) -> str:
    definitions = {
        # ... existing definitions ...
        '{new_check_name}': 'Description of what this check identifies',
    }
    return definitions.get(check_name, 'No definition available')
```

## Code Patterns to Follow

### Error Classification Pattern
From `modules/utils/validation.py`:

```python
# Status-affecting errors
if error_type == 'missing_columns':
    category = 'schema'
    display_type = 'Missing Required Columns'
elif error_type in ['datatype_mismatch', 'datatype_castable']:
    category = 'schema'
    display_type = 'Datatype Casting Error'
elif error_type == 'null_values':
    # Check if required column
    if column in required_columns:
        category = 'data_quality'
    else:
        category = 'informational'
```

### Missingness Analysis Pattern
Use utility function:
```python
from modules.utils.missingness import calculate_missingness

def get_summary_statistics(self) -> Dict[str, Any]:
    return {
        'data_info': self.get_data_info(),
        'missingness': calculate_missingness(self.table.df),
        'distributions': self.analyze_distributions()
    }
```

### Date Range Calculation Pattern
From hospitalization and ADT analyzers:
```python
if datetime_col in self.table.df.columns:
    self.table.df[datetime_col] = pd.to_datetime(self.table.df[datetime_col])
    first_year = self.table.df[datetime_col].dt.year.min()
    last_year = self.table.df[datetime_col].dt.year.max()
```

## Integration Checklist

Before submitting a new table analyzer:

### Code Implementation
- [ ] Created analyzer class inheriting from BaseTableAnalyzer
- [ ] Implemented all required methods (load_table, get_data_info, analyze_distributions)
- [ ] Added data quality checks with examples
- [ ] Added to TABLE_ANALYZERS in app.py
- [ ] Added to TABLE_ANALYZERS in run_analysis.py
- [ ] Added to TABLE_DISPLAY_NAMES in both files
- [ ] Added CLI argument in run_analysis.py
- [ ] Updated __init__.py exports
- [ ] Added quality check definitions

### Functionality Testing
- [ ] Table loads successfully from clifpy
- [ ] Validation runs and produces results
- [ ] Errors classified correctly (status-affecting vs informational)
- [ ] Summary statistics generate correctly
- [ ] Missingness analysis works
- [ ] Distribution plots render
- [ ] Data quality checks execute
- [ ] Examples display for failed checks

### Integration Testing
- [ ] Web app displays table in dropdown
- [ ] Validation tab shows errors correctly
- [ ] Summary tab shows all sections
- [ ] Feedback system works (accept/reject errors)
- [ ] Status recalculation works correctly
- [ ] Caching persists results
- [ ] Re-analysis clears old data
- [ ] CLI runs table successfully
- [ ] PDF/JSON outputs generate

### Edge Cases
- [ ] Handles empty dataframe
- [ ] Handles missing optional columns
- [ ] Handles all-null columns gracefully
- [ ] Handles invalid datetime formats
- [ ] Error messages are clear and actionable

## What NOT to Do

### ❌ Breaking Changes to Avoid

1. **Don't Modify Cache Manager Logic**
   - The cache_manager.py handles state persistence
   - Changes here affect ALL tables
   - Understand the flow before any modifications

2. **Don't Break Feedback System**
   - Error IDs must be unique and stable
   - Don't change error_id generation logic
   - Maintain status recalculation rules

3. **Don't Create Inconsistent Error Classifications**
   - Follow established patterns for status-affecting vs informational
   - Don't introduce new categories without discussion
   - Keep error types aligned with clifpy

4. **Don't Skip CLI Implementation**
   - Every table must work in both interfaces
   - CLI is used for automation and batch processing
   - Test both interfaces thoroughly

5. **Don't Hardcode Paths or Config Values**
   - Use config.json for all site-specific settings
   - Use output_dir from config
   - Support both parquet and csv filetypes

6. **Don't Ignore Timezone Handling**
   - Use timezone from config
   - Convert all datetime columns properly
   - Handle timezone-aware comparisons

## Example: Study Existing Analyzers

### Simple Table: Patient
- `modules/tables/patient_analysis.py`
- Basic demographics
- Categorical distributions
- Simple quality checks

### Complex Table: Hospitalization
- `modules/tables/hospitalization_analysis.py`
- Date range analysis
- Year distributions
- Multiple quality checks

### Relationship Table: ADT
- `modules/tables/adt_analysis.py`
- Location tracking
- ICU identification logic
- Complex queries (ICU-only, ED-only, etc.)

Study these to understand patterns and best practices.

## Getting Help

1. **Check Existing Code**: Look at patient_analysis.py, hospitalization_analysis.py, adt_analysis.py
2. **Review Legacy Code**: Check legacy/generate_clif_report_card.py for validation patterns
3. **Consult clifpy Docs**: Use Context7 MCP for API reference
4. **Test Incrementally**: Build and test each method before moving to next
5. **Use CLIF Schema**: Reference clif_2_1_data_dict.yaml for table structure

## Summary

Adding a new table requires:
1. Understanding the CLIF schema for that table
2. Creating an analyzer class with all required methods
3. Integrating with both web app and CLI
4. Following established error classification patterns
5. Maintaining feedback system compatibility
6. Testing thoroughly across all workflows

The system is designed to be modular - follow the patterns established by existing analyzers and you'll integrate smoothly with all existing features.
