# User Feedback & Caching System

## Overview
The CLIF Table One Analysis system now includes:
1. **User Feedback System**: Accept/reject validation errors with site-specific justifications
2. **Persistent State**: Cached analyses persist across sessions until explicitly re-run
3. **Adjusted Status**: Automatic recalculation of table status based on user decisions

## Features Implemented

### 1. User Feedback on Validation Errors

#### How It Works:
1. When validation finds errors, users can review each one individually
2. Three decision options for each error:
   - **Pending**: Not yet reviewed (default)
   - **Accepted**: Valid issue that needs attention
   - **Rejected**: Site-specific, not considered an issue
3. Rejected errors require a reason (e.g., "Site-specific category")
4. Status is automatically adjusted based on accepted errors only

#### Files Created:
- `modules/utils/feedback.py` - Core feedback logic
- Feedback saved as: `output/final/<table>_validation_response.json`

#### Key Functions:
- `create_feedback_structure()` - Initialize feedback for validation results
- `update_user_decision()` - Record user's accept/reject decision
- `recalculate_status()` - Determine new status based on accepted errors
- `save_feedback()` / `load_feedback()` - Persist decisions

### 2. Persistent Table State (Caching)

#### How It Works:
1. Analysis results cached in Streamlit session state
2. Cache includes:
   - Analyzer object
   - Validation results
   - Summary statistics
   - User feedback (if any)
   - Timestamp
3. Cache automatically invalidated if config changes
4. Users can force re-analysis with "Re-analyze table" checkbox

#### Files Created:
- `modules/utils/cache_manager.py` - Cache management

#### Key Functions:
- `cache_analysis()` - Store analysis results
- `get_cached_analysis()` - Retrieve cached results
- `is_table_cached()` - Check if table has cache
- `clear_all_cache()` - Reset all cached data
- `get_table_status()` - Get status considering feedback

### 3. Visual Indicators

#### Sidebar Status Display:
Each table shows its current state:
- ⭕ **Not analyzed** - Table hasn't been run yet
- ✅ **COMPLETE** - No issues (with timestamp)
- ⚠️ **PARTIAL** - Minor issues (with timestamp)
- ❌ **INCOMPLETE** - Critical issues (with timestamp)

Example:
```
✅ Patient - COMPLETE (Today at 2:30 PM)
```

#### Cache Info:
- Shows when analysis was last run
- Displays "Using cached analysis from..." when loading from cache
- "Re-analyze table" checkbox appears for cached tables

## Usage Guide

### For Users:

#### Running First Analysis:
1. Configure your `config/config.json`
2. Select a table from the sidebar
3. Click "Run Analysis"
4. View results in Validation and Summary tabs

#### Reviewing Validation Errors:
1. In the Validation tab, click "Review Validation Errors"
2. For each error:
   - Select "Accepted" if it's a valid issue
   - Select "Rejected" if it's site-specific (provide reason)
   - Leave as "Pending" if unsure
3. Click "Save Feedback"
4. Status automatically adjusts based on accepted errors

#### Re-running Analysis:
1. Check "Re-analyze table" checkbox in sidebar
2. Click "Run Analysis"
3. This forces fresh validation (clears cache)

#### Clearing Cache:
- Click "Clear All Cache" button in sidebar
- Resets all tables to "Not analyzed" state

### For Developers:

#### Adding Feedback to New Tables:
The system automatically works with any table analyzer that:
1. Inherits from `BaseTableAnalyzer`
2. Implements validation via clifpy
3. Returns errors in the standard format

No additional code needed!

#### Customizing Status Logic:
Edit `recalculate_status()` in `modules/utils/feedback.py` to change how accepted errors affect status.

## Output Files

### 1. Validation Response File
**Location**: `output/final/<table>_validation_response.json`

**Structure**:
```json
{
  "table": "patient",
  "timestamp": "2025-10-09T12:00:00",
  "original_status": "partial",
  "adjusted_status": "complete",
  "total_errors": 5,
  "accepted_count": 0,
  "rejected_count": 5,
  "pending_count": 0,
  "user_decisions": {
    "invalid_category_a1b2c3d4": {
      "error_type": "Invalid Categories",
      "description": "Column 'race_category' contains invalid values: Other",
      "decision": "rejected",
      "reason": "Site-specific category approved by IRB",
      "timestamp": "2025-10-09T12:01:00"
    }
  }
}
```

### 2. Raw Validation Results
**Location**: `output/final/<table>_summary_validation.json`

Contains unmodified clifpy validation output.

### 3. Summary Statistics
**Location**: `output/final/<table>_summary_summary.json`

Contains missingness, distributions, and data quality checks.

## API Reference

### Feedback System

```python
from modules.utils import (
    create_feedback_structure,
    update_user_decision,
    get_feedback_summary,
    save_feedback,
    load_feedback
)

# Create initial feedback
feedback = create_feedback_structure(validation_results, "patient")

# Update a decision
feedback = update_user_decision(
    feedback,
    error_id="invalid_category_abc123",
    decision="rejected",
    reason="Site uses custom categories"
)

# Save
save_feedback(feedback, "output", "patient")

# Load existing
feedback = load_feedback("output", "patient")
```

### Cache System

```python
from modules.utils import (
    cache_analysis,
    get_cached_analysis,
    is_table_cached,
    clear_all_cache
)

# Cache results
cache_analysis("patient", analyzer, validation, summary, feedback)

# Check if cached
if is_table_cached("patient"):
    cached = get_cached_analysis("patient")

# Clear everything
clear_all_cache()
```

## Benefits

### For Sites:
1. **Flexibility**: Accept/reject errors based on site context
2. **Documentation**: All decisions logged with reasons
3. **Efficiency**: No need to re-run analysis repeatedly
4. **Transparency**: Clear audit trail of validation decisions

### For Consortium:
1. **Site Context**: Understand site-specific variations
2. **Data Quality**: True validation status after review
3. **Compliance**: Documented justifications for deviations
4. **Comparison**: Can compare original vs adjusted status

## Future Enhancements

Possible additions:
1. Export feedback report to PDF
2. Multi-user feedback (track who made decisions)
3. Feedback templates for common rejections
4. Batch accept/reject for similar errors
5. Compare feedback across sites
6. Auto-sync feedback when data updates