"""
Base Table Analyzer for CLIF 2.1 Tables

This module provides the base class for all table-specific analyzers,
leveraging clifpy's validation capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import os


class BaseTableAnalyzer(ABC):
    """Base class for all table-specific analyzers."""

    def __init__(self, data_dir: str, filetype: str = 'parquet',
                 timezone: str = 'UTC', output_dir: str = None, sample_filter: Optional[List[str]] = None):
        """
        Initialize the base analyzer.

        Parameters:
        -----------
        data_dir : str
            Directory containing the CLIF data files
        filetype : str
            File format (parquet, csv, feather)
        timezone : str
            Timezone for datetime processing
        output_dir : str, optional
            Directory for output files
        sample_filter : list, optional
            List of hospitalization_ids to filter to (for sampling)
        """
        self.data_dir = data_dir
        self.filetype = filetype
        self.timezone = timezone
        self.output_dir = output_dir or 'output'
        self.table = None

        # Ensure output directories exist with new structure
        os.makedirs(os.path.join(self.output_dir, 'intermediate'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'final'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'final', 'reports'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'final', 'results'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'final', 'clifpy'), exist_ok=True)

        # Load the table
        self.load_table(sample_filter)

    @abstractmethod
    def load_table(self, sample_filter=None):
        """Load the specific clifpy table class."""
        pass

    def validate(self) -> Dict[str, Any]:
        """
        Run clifpy validation and format results.

        Returns:
        --------
        dict
            Validation results with status, errors, and data info
        """
        if self.table is None:
            return {
                'is_valid': False,
                'errors': {'schema_errors': [{'type': 'Table Not Loaded',
                                             'description': 'Table could not be loaded',
                                             'category': 'schema'}]},
                'status': 'incomplete',
                'data_info': {}
            }

        # Run clifpy validation
        self.table.validate()

        return {
            'is_valid': self.table.isvalid() if hasattr(self.table, 'isvalid') else False,
            'errors': self.format_errors(self.table.errors if hasattr(self.table, 'errors') else []),
            'status': self.determine_status(),
            'data_info': self.get_data_info()
        }

    def format_errors(self, errors: list) -> Dict[str, list]:
        """
        Format errors into categories like clif_report_card.py.

        Parameters:
        -----------
        errors : list
            List of error dictionaries from clifpy

        Returns:
        --------
        dict
            Categorized errors (schema_errors, data_quality_issues, other_errors)
        """
        schema_errors = []
        data_quality_issues = []
        other_errors = []

        for error in errors:
            formatted = self.format_single_error(error)

            if formatted['category'] == 'schema':
                schema_errors.append(formatted)
            elif formatted['category'] == 'data_quality':
                data_quality_issues.append(formatted)
            else:
                other_errors.append(formatted)

        return {
            'schema_errors': schema_errors,
            'data_quality_issues': data_quality_issues,
            'other_errors': other_errors
        }

    def format_single_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format a single error for display.

        Parameters:
        -----------
        error : dict
            Error dictionary from clifpy

        Returns:
        --------
        dict
            Formatted error with type, description, and category
        """
        from ..utils.validation import format_clifpy_error

        # Get row count if available
        row_count = len(self.table.df) if hasattr(self.table, 'df') and self.table.df is not None else 0

        # Get table name
        table_name = self.get_table_name()

        return format_clifpy_error(error, row_count, table_name)

    def determine_status(self) -> str:
        """
        Determine validation status (complete/partial/incomplete).

        Returns:
        --------
        str
            Status: 'complete', 'partial', or 'incomplete'
        """
        if not hasattr(self.table, 'errors'):
            return 'complete' if self.table and hasattr(self.table, 'df') else 'incomplete'

        errors = self.table.errors

        if not errors:
            return 'complete'

        # Format errors to check their types
        formatted_errors = [self.format_single_error(e) for e in errors]

        # Get required columns from schema if available
        required_columns = []
        if hasattr(self.table, 'schema') and self.table.schema:
            required_columns = self.table.schema.get('required_columns', [])

        # Get table name
        table_name = self.get_table_name()

        from ..utils.validation import determine_validation_status
        return determine_validation_status(formatted_errors, required_columns, table_name)

    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get table-specific data information.

        Returns:
        --------
        dict
            Information about the loaded data
        """
        pass

    @abstractmethod
    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze table-specific distributions.

        Returns:
        --------
        dict
            Distribution analysis results
        """
        pass

    def calculate_missingness(self) -> Dict[str, Any]:
        """
        Calculate missingness for all columns.

        Returns:
        --------
        dict
            Missingness statistics for the table
        """
        from ..utils.missingness import calculate_missingness

        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'total_rows': 0
            }

        return calculate_missingness(self.table.df)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the table.

        Returns:
        --------
        dict
            Summary statistics including distributions and missingness
        """
        return {
            'data_info': self.get_data_info(),
            'missingness': self.calculate_missingness(),
            'distributions': self.analyze_distributions()
        }

    def save_intermediate_data(self, df: pd.DataFrame, suffix: str = ''):
        """
        Save intermediate data to output directory.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        suffix : str
            Suffix for the filename
        """
        if df is None or df.empty:
            return

        table_name = self.get_table_name()
        filename = f"{table_name}{suffix}.{self.filetype}"
        filepath = os.path.join(self.output_dir, 'intermediate', filename)

        if self.filetype == 'parquet':
            df.to_parquet(filepath, index=False)
        elif self.filetype == 'csv':
            df.to_csv(filepath, index=False)
        elif self.filetype == 'feather':
            df.to_feather(filepath)

    def save_summary_data(self, summary: Dict[str, Any], suffix: str = ''):
        """
        Save summary data to output directory.

        Parameters:
        -----------
        summary : dict
            Summary data to save
        suffix : str
            Suffix for the filename
        """
        import json

        table_name = self.get_table_name()
        filename = f"{table_name}_summary{suffix}.json"
        filepath = os.path.join(self.output_dir, 'final', 'results', filename)

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

    def save_validation_feedback(self, feedback: Dict[str, Any]):
        """
        Save validation feedback (user accept/reject decisions).

        Parameters:
        -----------
        feedback : dict
            Feedback structure with user decisions
        """
        from ..utils.feedback import save_feedback

        table_name = self.get_table_name()
        return save_feedback(feedback, self.output_dir, table_name)

    def load_validation_feedback(self) -> Optional[Dict[str, Any]]:
        """
        Load existing validation feedback if available.

        Returns:
        --------
        dict or None
            Feedback structure or None if not found
        """
        from ..utils.feedback import load_feedback

        table_name = self.get_table_name()
        return load_feedback(self.output_dir, table_name)

    def get_table_name(self) -> str:
        """
        Get the table name from the analyzer class.

        Returns:
        --------
        str
            Table name
        """
        return self.__class__.__name__.replace('Analyzer', '').lower()