"""
Distribution Analysis Utilities for CLIF Data

This module provides functions to analyze data distributions,
including ECDF generation and categorical/numerical distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union
import warnings


def generate_ecdf(series: pd.Series, name: str = None) -> Dict[str, Any]:
    """
    Generate ECDF (Empirical Cumulative Distribution Function) for a series.

    Parameters:
    -----------
    series : pd.Series
        The data series to analyze
    name : str, optional
        Name for the ECDF

    Returns:
    --------
    dict
        ECDF data including x and y values, and summary statistics
    """
    if series is None or series.empty:
        return {'error': 'No data available', 'name': name}

    # Remove NaN values
    clean_data = series.dropna()

    if len(clean_data) == 0:
        return {'error': 'All values are missing', 'name': name}

    # Convert to numeric if possible
    try:
        clean_data = pd.to_numeric(clean_data, errors='coerce').dropna()
    except:
        return {'error': 'Cannot convert to numeric values', 'name': name}

    if len(clean_data) == 0:
        return {'error': 'No numeric values found', 'name': name}

    # Sort data
    sorted_data = np.sort(clean_data)

    # Calculate ECDF
    n = len(sorted_data)
    ecdf_y = np.arange(1, n + 1) / n

    # Calculate percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = {}
    for p in percentiles:
        percentile_values[f'p{p}'] = float(np.percentile(sorted_data, p))

    return {
        'x': sorted_data.tolist(),
        'y': ecdf_y.tolist(),
        'name': name or str(series.name) if hasattr(series, 'name') else 'ECDF',
        'count': n,
        'min': float(sorted_data[0]),
        'max': float(sorted_data[-1]),
        'mean': float(np.mean(sorted_data)),
        'median': float(np.median(sorted_data)),
        'std': float(np.std(sorted_data)),
        'percentiles': percentile_values
    }


def get_categorical_distribution(df: pd.DataFrame, column: str,
                               top_n: int = None) -> Dict[str, Any]:
    """
    Get distribution statistics for a categorical column.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the column
    column : str
        Column name to analyze
    top_n : int, optional
        If specified, only return top N categories

    Returns:
    --------
    dict
        Distribution statistics for the categorical column
    """
    if df is None or df.empty:
        return {'error': 'No data available', 'column': column}

    if column not in df.columns:
        return {'error': f'Column {column} not found', 'column': column}

    # Get value counts
    value_counts = df[column].value_counts(dropna=False)

    # Handle top_n if specified
    if top_n and len(value_counts) > top_n:
        top_values = value_counts.head(top_n)
        other_count = value_counts[top_n:].sum()
        value_counts = pd.concat([top_values, pd.Series({'Other': other_count})])

    # Include NaN count
    nan_count = df[column].isna().sum()

    # Calculate percentages
    total = len(df)
    percentages = (value_counts / total * 100).round(2)

    # Prepare result
    categories = []
    for val, count in value_counts.items():
        if pd.isna(val):
            val_str = 'Missing'
        else:
            val_str = str(val)

        categories.append({
            'value': val_str,
            'count': int(count),
            'percentage': float(percentages[val])
        })

    return {
        'column': column,
        'categories': categories,
        'unique_values': int(df[column].nunique()),
        'total_unique_including_nan': len(value_counts),
        'missing_count': int(nan_count),
        'missing_percentage': round(nan_count / total * 100, 2) if total > 0 else 0,
        'total_rows': total,
        'mode': str(df[column].mode()[0]) if not df[column].mode().empty else None,
        'entropy': calculate_entropy(value_counts)
    }


def get_numeric_distribution(df: pd.DataFrame, column: str,
                           detect_outliers: bool = True) -> Dict[str, Any]:
    """
    Get distribution statistics for a numeric column.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the column
    column : str
        Column name to analyze
    detect_outliers : bool
        Whether to detect outliers using IQR method

    Returns:
    --------
    dict
        Distribution statistics for the numeric column
    """
    if df is None or df.empty:
        return {'error': 'No data available', 'column': column}

    if column not in df.columns:
        return {'error': f'Column {column} not found', 'column': column}

    # Convert to numeric, coercing errors to NaN
    numeric_series = pd.to_numeric(df[column], errors='coerce')

    # Remove NaN values for statistics
    clean_data = numeric_series.dropna()

    if len(clean_data) == 0:
        return {
            'error': 'No numeric values found',
            'column': column,
            'missing_count': len(df),
            'missing_percentage': 100.0
        }

    # Calculate statistics
    stats = {
        'column': column,
        'count': int(len(clean_data)),
        'missing_count': int(len(numeric_series) - len(clean_data)),
        'missing_percentage': round((len(numeric_series) - len(clean_data)) / len(numeric_series) * 100, 2),
        'mean': float(clean_data.mean()),
        'std': float(clean_data.std()),
        'min': float(clean_data.min()),
        'q1': float(clean_data.quantile(0.25)),
        'median': float(clean_data.median()),
        'q3': float(clean_data.quantile(0.75)),
        'max': float(clean_data.max()),
        'iqr': float(clean_data.quantile(0.75) - clean_data.quantile(0.25)),
        'range': float(clean_data.max() - clean_data.min()),
        'cv': float(clean_data.std() / clean_data.mean() * 100) if clean_data.mean() != 0 else None,
        'skewness': float(clean_data.skew()),
        'kurtosis': float(clean_data.kurtosis())
    }

    # Detect outliers if requested
    if detect_outliers:
        outliers = detect_outliers_iqr(clean_data)
        stats['outliers'] = {
            'count': len(outliers),
            'percentage': round(len(outliers) / len(clean_data) * 100, 2),
            'values': outliers[:10] if len(outliers) <= 10 else None,  # Show first 10
            'total_outliers': len(outliers)
        }

    return stats


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> List[float]:
    """
    Detect outliers using the IQR (Interquartile Range) method.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to analyze
    multiplier : float
        IQR multiplier for outlier detection (default: 1.5)

    Returns:
    --------
    list
        List of outlier values
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outliers = series[(series < lower_bound) | (series > upper_bound)].tolist()
    return outliers


def calculate_entropy(value_counts: pd.Series) -> float:
    """
    Calculate Shannon entropy for a categorical distribution.

    Parameters:
    -----------
    value_counts : pd.Series
        Value counts for the categories

    Returns:
    --------
    float
        Shannon entropy value
    """
    if len(value_counts) == 0:
        return 0.0

    # Calculate probabilities
    probabilities = value_counts / value_counts.sum()

    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return round(float(entropy), 3)


def compare_distributions(df1: pd.DataFrame, df2: pd.DataFrame,
                         column: str, test: str = 'auto') -> Dict[str, Any]:
    """
    Compare distributions between two dataframes.

    Parameters:
    -----------
    df1 : pd.DataFrame
        First dataframe
    df2 : pd.DataFrame
        Second dataframe
    column : str
        Column to compare
    test : str
        Statistical test to use ('auto', 'ks', 'chi2', 'mannwhitney')

    Returns:
    --------
    dict
        Comparison results including test statistics
    """
    from scipy import stats

    if column not in df1.columns or column not in df2.columns:
        return {'error': f'Column {column} not found in one or both dataframes'}

    series1 = df1[column].dropna()
    series2 = df2[column].dropna()

    if len(series1) == 0 or len(series2) == 0:
        return {'error': 'One or both series have no data'}

    result = {
        'column': column,
        'n1': len(series1),
        'n2': len(series2)
    }

    # Determine if numeric or categorical
    is_numeric = pd.api.types.is_numeric_dtype(series1) and pd.api.types.is_numeric_dtype(series2)

    if is_numeric:
        # Convert to numeric
        series1 = pd.to_numeric(series1, errors='coerce').dropna()
        series2 = pd.to_numeric(series2, errors='coerce').dropna()

        # Summary statistics
        result['stats1'] = {
            'mean': float(series1.mean()),
            'std': float(series1.std()),
            'median': float(series1.median())
        }
        result['stats2'] = {
            'mean': float(series2.mean()),
            'std': float(series2.std()),
            'median': float(series2.median())
        }

        # Statistical tests
        if test == 'auto' or test == 'ks':
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(series1, series2)
            result['ks_test'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'significant': ks_pvalue < 0.05
            }

        if test == 'auto' or test == 'mannwhitney':
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(series1, series2, alternative='two-sided')
            result['mannwhitney_test'] = {
                'statistic': float(mw_stat),
                'p_value': float(mw_pvalue),
                'significant': mw_pvalue < 0.05
            }

    else:
        # Categorical data
        # Get value counts
        counts1 = series1.value_counts()
        counts2 = series2.value_counts()

        # Align categories
        all_categories = set(counts1.index) | set(counts2.index)
        aligned_counts1 = pd.Series([counts1.get(cat, 0) for cat in all_categories], index=all_categories)
        aligned_counts2 = pd.Series([counts2.get(cat, 0) for cat in all_categories], index=all_categories)

        if test == 'auto' or test == 'chi2':
            # Chi-square test
            chi2_stat, chi2_pvalue, _, _ = stats.chi2_contingency(
                pd.DataFrame({'group1': aligned_counts1, 'group2': aligned_counts2}).T
            )
            result['chi2_test'] = {
                'statistic': float(chi2_stat),
                'p_value': float(chi2_pvalue),
                'significant': chi2_pvalue < 0.05
            }

    return result


def generate_distribution_summary(df: pd.DataFrame,
                                 categorical_columns: List[str] = None,
                                 numeric_columns: List[str] = None) -> Dict[str, Any]:
    """
    Generate a summary of distributions for specified columns.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    categorical_columns : list, optional
        List of categorical column names
    numeric_columns : list, optional
        List of numeric column names

    Returns:
    --------
    dict
        Summary of distributions for all specified columns
    """
    if df is None or df.empty:
        return {'error': 'No data available'}

    summary = {}

    # Process categorical columns
    if categorical_columns:
        summary['categorical'] = {}
        for col in categorical_columns:
            if col in df.columns:
                summary['categorical'][col] = get_categorical_distribution(df, col, top_n=10)

    # Process numeric columns
    if numeric_columns:
        summary['numeric'] = {}
        for col in numeric_columns:
            if col in df.columns:
                summary['numeric'][col] = get_numeric_distribution(df, col)

    # Auto-detect if no columns specified
    if not categorical_columns and not numeric_columns:
        summary['auto_detected'] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                summary['auto_detected'][col] = get_numeric_distribution(df, col)
            else:
                summary['auto_detected'][col] = get_categorical_distribution(df, col, top_n=10)

    return summary