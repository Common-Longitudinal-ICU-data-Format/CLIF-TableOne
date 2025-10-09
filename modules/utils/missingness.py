"""
Missingness Analysis Utilities for CLIF Data

This module provides functions to analyze missing data patterns
in CLIF tables.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def calculate_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive missingness statistics for a dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze

    Returns:
    --------
    dict
        Missingness statistics including counts and percentages
    """
    if df is None or df.empty:
        return {
            'error': 'No data available',
            'total_rows': 0,
            'total_columns': 0
        }

    total_rows = len(df)
    total_columns = len(df.columns)

    # Calculate per-column missingness
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / total_rows * 100).round(2)

    # Find columns with any missing data
    columns_with_missing = []
    for col in df.columns:
        if missing_counts[col] > 0:
            columns_with_missing.append({
                'column': col,
                'missing_count': int(missing_counts[col]),
                'missing_percent': float(missing_percentages[col]),
                'complete_count': int(total_rows - missing_counts[col]),
                'complete_percent': float(100 - missing_percentages[col])
            })

    # Sort by missing percentage (descending)
    columns_with_missing.sort(key=lambda x: x['missing_percent'], reverse=True)

    # Find complete columns (no missing data)
    complete_columns = [col for col in df.columns if missing_counts[col] == 0]

    # Calculate overall statistics
    total_cells = total_rows * total_columns
    missing_cells = int(df.isnull().sum().sum())
    complete_cells = total_cells - missing_cells

    # Calculate row-wise missingness
    row_missing_counts = df.isnull().sum(axis=1)
    complete_rows = int((row_missing_counts == 0).sum())
    partial_rows = int(((row_missing_counts > 0) & (row_missing_counts < total_columns)).sum())
    empty_rows = int((row_missing_counts == total_columns).sum())

    return {
        'total_rows': total_rows,
        'total_columns': total_columns,
        'total_cells': total_cells,
        'columns_with_missing': columns_with_missing,
        'complete_columns': complete_columns,
        'complete_columns_count': len(complete_columns),
        'columns_with_missing_count': len(columns_with_missing),
        'missing_cells': missing_cells,
        'complete_cells': complete_cells,
        'overall_missing_percentage': round((missing_cells / total_cells * 100) if total_cells > 0 else 0, 2),
        'overall_complete_percentage': round((complete_cells / total_cells * 100) if total_cells > 0 else 0, 2),
        'complete_rows': complete_rows,
        'partial_rows': partial_rows,
        'empty_rows': empty_rows,
        'complete_rows_percentage': round((complete_rows / total_rows * 100) if total_rows > 0 else 0, 2)
    }


def get_high_missingness_columns(df: pd.DataFrame, threshold: float = 50.0) -> List[Dict[str, Any]]:
    """
    Get columns with missingness above a specified threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    threshold : float
        Percentage threshold for high missingness (default: 50%)

    Returns:
    --------
    list
        List of columns with high missingness and their statistics
    """
    if df is None or df.empty:
        return []

    total_rows = len(df)
    missing_percentages = (df.isnull().sum() / total_rows * 100)

    high_missingness = []
    for col in df.columns:
        missing_pct = missing_percentages[col]
        if missing_pct > threshold:
            high_missingness.append({
                'column': col,
                'missing_percent': round(missing_pct, 2),
                'missing_count': int(df[col].isnull().sum()),
                'threshold_exceeded_by': round(missing_pct - threshold, 2)
            })

    # Sort by missing percentage (descending)
    high_missingness.sort(key=lambda x: x['missing_percent'], reverse=True)

    return high_missingness


def analyze_missingness_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze patterns in missing data (e.g., correlations between missing columns).

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze

    Returns:
    --------
    dict
        Analysis of missingness patterns
    """
    if df is None or df.empty:
        return {'error': 'No data available'}

    # Create a boolean dataframe of missingness
    missing_df = df.isnull()

    # Find columns that are always missing together
    correlated_missing = []
    columns = list(df.columns)

    for i, col1 in enumerate(columns):
        for col2 in columns[i+1:]:
            # Check if columns are missing together
            both_missing = (missing_df[col1] & missing_df[col2]).sum()
            either_missing = (missing_df[col1] | missing_df[col2]).sum()

            if either_missing > 0:
                correlation = both_missing / either_missing
                if correlation > 0.8:  # High correlation threshold
                    correlated_missing.append({
                        'column1': col1,
                        'column2': col2,
                        'correlation': round(correlation, 3),
                        'both_missing_count': int(both_missing)
                    })

    # Sort by correlation
    correlated_missing.sort(key=lambda x: x['correlation'], reverse=True)

    return {
        'correlated_missing_pairs': correlated_missing[:10],  # Top 10 pairs
        'total_correlated_pairs': len(correlated_missing)
    }


def get_missingness_summary(df: pd.DataFrame) -> str:
    """
    Generate a text summary of missingness in the dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze

    Returns:
    --------
    str
        Human-readable summary of missingness
    """
    stats = calculate_missingness(df)

    if 'error' in stats:
        return stats['error']

    summary_parts = []

    # Overall missingness
    overall_pct = stats['overall_missing_percentage']
    if overall_pct == 0:
        summary_parts.append("✅ No missing data found")
    elif overall_pct < 5:
        summary_parts.append(f"✅ Low missingness: {overall_pct}% overall")
    elif overall_pct < 20:
        summary_parts.append(f"⚠️  Moderate missingness: {overall_pct}% overall")
    else:
        summary_parts.append(f"❌ High missingness: {overall_pct}% overall")

    # Column statistics
    complete_cols = stats['complete_columns_count']
    missing_cols = stats['columns_with_missing_count']
    total_cols = stats['total_columns']

    summary_parts.append(f"{complete_cols}/{total_cols} complete columns")

    # Row statistics
    complete_rows_pct = stats['complete_rows_percentage']
    summary_parts.append(f"{complete_rows_pct}% complete rows")

    return " | ".join(summary_parts)


def create_missingness_report(df: pd.DataFrame,
                             output_format: str = 'dict') -> Any:
    """
    Create a comprehensive missingness report.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    output_format : str
        Format for output ('dict', 'dataframe', or 'html')

    Returns:
    --------
    dict, pd.DataFrame, or str
        Missingness report in specified format
    """
    # Calculate basic statistics
    stats = calculate_missingness(df)

    if 'error' in stats:
        if output_format == 'dict':
            return stats
        elif output_format == 'dataframe':
            return pd.DataFrame()
        else:
            return "<p>No data available</p>"

    # Get high missingness columns
    high_missing = get_high_missingness_columns(df, threshold=30)

    # Create report
    report = {
        'summary': get_missingness_summary(df),
        'overall_statistics': {
            'total_rows': stats['total_rows'],
            'total_columns': stats['total_columns'],
            'missing_percentage': stats['overall_missing_percentage'],
            'complete_rows_percentage': stats['complete_rows_percentage']
        },
        'column_statistics': {
            'complete_columns': stats['complete_columns_count'],
            'partial_columns': stats['columns_with_missing_count'],
            'high_missingness_columns': len(high_missing)
        },
        'high_missingness_columns': high_missing,
        'detailed_column_missingness': stats['columns_with_missing']
    }

    if output_format == 'dict':
        return report

    elif output_format == 'dataframe':
        # Convert to dataframe
        if stats['columns_with_missing']:
            return pd.DataFrame(stats['columns_with_missing'])
        else:
            return pd.DataFrame({'Message': ['No missing data found']})

    elif output_format == 'html':
        # Create HTML report
        html = f"""
        <h3>Missingness Report</h3>
        <p><strong>Summary:</strong> {report['summary']}</p>
        <h4>Overall Statistics</h4>
        <ul>
            <li>Total Rows: {report['overall_statistics']['total_rows']:,}</li>
            <li>Total Columns: {report['overall_statistics']['total_columns']}</li>
            <li>Missing Data: {report['overall_statistics']['missing_percentage']}%</li>
            <li>Complete Rows: {report['overall_statistics']['complete_rows_percentage']}%</li>
        </ul>
        """

        if high_missing:
            html += "<h4>High Missingness Columns (>30%)</h4><ul>"
            for col in high_missing[:5]:  # Show top 5
                html += f"<li>{col['column']}: {col['missing_percent']}% missing</li>"
            if len(high_missing) > 5:
                html += f"<li>... and {len(high_missing) - 5} more</li>"
            html += "</ul>"

        return html

    else:
        raise ValueError(f"Invalid output format: {output_format}")