"""
Hospital Diagnosis table analyzer using clifpy for CLIF 2.1.

This module provides comprehensive analysis for the hospital_diagnosis table,
including CCI (Charlson Comorbidity Index) score calculation.
"""

from clifpy.tables.hospital_diagnosis import HospitalDiagnosis
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class HospitalDiagnosisAnalyzer(BaseTableAnalyzer):
    """
    Analyzer for Hospital Diagnosis table using clifpy.

    Includes analysis of diagnosis codes, distributions, and CCI score calculation.
    """

    def get_table_name(self) -> str:
        """Return the table name."""
        return 'hospital_diagnosis'

    def load_table(self, sample_filter=None):
        """
        Load Hospital Diagnosis table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)

        Handles both naming conventions:
        - hospital_diagnosis.parquet
        - clif_hospital_diagnosis.parquet
        """
        data_path = Path(self.data_dir)
        filetype = self.filetype

        # Check both file naming conventions
        file_without_clif = data_path / f"hospital_diagnosis.{filetype}"
        file_with_clif = data_path / f"clif_hospital_diagnosis.{filetype}"

        file_exists = file_without_clif.exists() or file_with_clif.exists()

        if not file_exists:
            print(f"⚠️  No hospital_diagnosis file found in {self.data_dir}")
            print(f"   Looking for: hospital_diagnosis.{filetype} or clif_hospital_diagnosis.{filetype}")
            self.table = None
            return

        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = HospitalDiagnosis.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                self.table = HospitalDiagnosis.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir
                )
        except FileNotFoundError:
            print(f"⚠️  hospital_diagnosis table file not found in {self.data_dir}")
            self.table = None
        except Exception as e:
            print(f"⚠️  Error loading hospital_diagnosis table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get hospital_diagnosis basic data information.

        Returns:
            Dictionary containing:
            - row_count: Total number of records
            - column_count: Number of columns
            - unique_hospitalizations: Number of unique hospitalizations with diagnoses
            - unique_diagnosis_codes: Number of unique diagnosis codes
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        info = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }

        # Add unique counts for ID columns
        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()

        if 'diagnosis_code' in df.columns:
            info['unique_diagnosis_codes'] = df['diagnosis_code'].nunique()

        # Add diagnoses per hospitalization statistics
        if 'hospitalization_id' in df.columns:
            dx_per_hosp = df.groupby('hospitalization_id').size()
            info['avg_diagnoses_per_hosp'] = float(dx_per_hosp.mean())
            info['median_diagnoses_per_hosp'] = float(dx_per_hosp.median())
            info['max_diagnoses_per_hosp'] = int(dx_per_hosp.max())

        # Add diagnosis format distribution
        if 'diagnosis_code_format' in df.columns:
            format_counts = df['diagnosis_code_format'].value_counts().to_dict()
            info['diagnosis_formats'] = format_counts
            info['icd10cm_count'] = format_counts.get('ICD10CM', 0)
            info['icd9cm_count'] = format_counts.get('ICD9CM', 0)
            info['icd10cm_percentage'] = (format_counts.get('ICD10CM', 0) / len(df) * 100) if len(df) > 0 else 0
            info['icd9cm_percentage'] = (format_counts.get('ICD9CM', 0) / len(df) * 100) if len(df) > 0 else 0

        return info

    def get_diagnosis_format_distribution(self) -> Dict[str, Any]:
        """
        Get distribution of diagnosis code formats (ICD9CM vs ICD10CM).

        Returns:
        --------
        dict
            Distribution of diagnosis formats with counts and percentages
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if 'diagnosis_code_format' not in df.columns:
            return {'error': 'diagnosis_code_format column not found'}

        # Get format counts
        format_counts = df['diagnosis_code_format'].value_counts()
        total_diagnoses = len(df)

        # Get unique hospitalizations per format
        format_hosp_counts = {}
        for format_type in format_counts.index:
            hosp_count = df[df['diagnosis_code_format'] == format_type]['hospitalization_id'].nunique()
            format_hosp_counts[format_type] = hosp_count

        total_hospitalizations = df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 0

        # Calculate percentages
        format_data = []
        for format_type in format_counts.index:
            format_data.append({
                'format': format_type,
                'diagnosis_count': int(format_counts[format_type]),
                'diagnosis_percentage': float(format_counts[format_type] / total_diagnoses * 100),
                'hospitalization_count': format_hosp_counts[format_type],
                'hospitalization_percentage': float(format_hosp_counts[format_type] / total_hospitalizations * 100) if total_hospitalizations > 0 else 0
            })

        return {
            'formats': format_data,
            'total_diagnoses': total_diagnoses,
            'total_hospitalizations': total_hospitalizations
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze hospital_diagnosis distributions.

        Returns:
            Dictionary containing distribution data for top diagnosis codes and format distribution
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        distributions = {}
        df = self.table.df

        # Diagnosis format distribution
        distributions['diagnosis_format'] = self.get_diagnosis_format_distribution()

        # Top diagnosis codes distribution
        if 'diagnosis_code' in df.columns:
            distributions['top_diagnosis_codes'] = self.get_top_diagnosis_codes(n=20)

        return distributions

    def get_top_diagnosis_codes(self, n: int = 20) -> Dict[str, Any]:
        """
        Get the top N most common diagnosis codes.

        Parameters:
        -----------
        n : int
            Number of top codes to return (default: 20)

        Returns:
        --------
        dict
            Top diagnosis codes with counts and percentages
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if 'diagnosis_code' not in df.columns:
            return {'error': 'diagnosis_code column not found'}

        # Get value counts
        code_counts = df['diagnosis_code'].value_counts().head(n)

        # Calculate percentages (of unique hospitalizations, not total records)
        total_hospitalizations = df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else len(df)

        # For each code, count unique hospitalizations
        code_hosp_counts = {}
        for code in code_counts.index:
            hosp_with_code = df[df['diagnosis_code'] == code]['hospitalization_id'].nunique()
            code_hosp_counts[code] = hosp_with_code

        # Convert to percentages
        percentages = [(count / total_hospitalizations * 100) for count in code_hosp_counts.values()]

        return {
            'codes': code_counts.index.tolist(),
            'counts': list(code_hosp_counts.values()),
            'percentages': percentages,
            'total_records': code_counts.values.tolist(),  # Total occurrences of each code
            'total_hospitalizations': total_hospitalizations
        }

    def check_cci_compatibility(self) -> Dict[str, Any]:
        """
        Check if the data is compatible with CCI calculation.

        Returns:
        --------
        dict
            Compatibility information including whether CCI can be calculated
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'compatible': False, 'reason': 'No data available'}

        df = self.table.df

        # Check required columns
        required_cols = ['hospitalization_id', 'diagnosis_code', 'diagnosis_code_format']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'compatible': False, 'reason': f'Missing required columns: {missing_cols}'}

        # Check for ICD10CM codes
        format_counts = df['diagnosis_code_format'].value_counts().to_dict()
        icd10_count = format_counts.get('ICD10CM', 0)

        if icd10_count == 0:
            return {
                'compatible': False,
                'reason': 'No ICD10CM diagnosis codes found',
                'format_distribution': format_counts,
                'message': 'CCI calculation requires ICD10CM diagnosis codes'
            }

        return {
            'compatible': True,
            'icd10_count': icd10_count,
            'total_diagnoses': len(df),
            'format_distribution': format_counts,
            'percentage_icd10': (icd10_count / len(df) * 100) if len(df) > 0 else 0
        }

    def calculate_cci_distribution(self) -> Dict[str, Any]:
        """
        Calculate Charlson Comorbidity Index (CCI) distribution using clifpy.

        Returns:
        --------
        dict
            CCI score statistics and distribution
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        # Check if we have ICD10CM codes (CCI only works with ICD10CM)
        df = self.table.df
        if 'diagnosis_code_format' in df.columns:
            icd10_count = (df['diagnosis_code_format'] == 'ICD10CM').sum()
            if icd10_count == 0:
                # Get format distribution for informative error
                format_counts = df['diagnosis_code_format'].value_counts().to_dict()
                formats_str = ', '.join([f'{k}: {v}' for k, v in format_counts.items()])
                return {'error': f'No ICD10CM diagnosis codes found. CCI requires ICD10CM codes. Found formats: {formats_str}'}

        try:
            from clifpy.utils.comorbidity import calculate_cci

            # Calculate CCI scores with hierarchy
            cci_results = calculate_cci(self.table, hierarchy=True)

            # Check for empty results
            if cci_results is None or cci_results.empty:
                return {'error': 'CCI calculation returned no results. This may occur if there are no valid ICD10CM diagnosis codes.'}

            # Calculate statistics
            cci_scores = cci_results['cci_score']

            # Check if we have valid scores
            if cci_scores.empty or cci_scores.isna().all():
                return {'error': 'No valid CCI scores could be calculated'}

            # Calculate statistics safely
            stats = {
                'count': int(len(cci_scores)),
                'mean': float(cci_scores.mean()) if not cci_scores.empty else 0,
                'std': float(cci_scores.std()) if not cci_scores.empty else 0,
                'min': int(cci_scores.min()) if not cci_scores.empty else 0,
                'q1': float(cci_scores.quantile(0.25)) if not cci_scores.empty else 0,
                'median': float(cci_scores.median()) if not cci_scores.empty else 0,
                'q3': float(cci_scores.quantile(0.75)) if not cci_scores.empty else 0,
                'max': int(cci_scores.max()) if not cci_scores.empty else 0
            }

            # Group into risk categories (only if we have data)
            if not cci_scores.empty:
                risk_categories = pd.cut(
                    cci_scores,
                    bins=[-np.inf, 0, 2, 4, np.inf],
                    labels=['0 (No comorbidity)', '1-2 (Low)', '3-4 (Moderate)', '≥5 (High)']
                )
                risk_dist = risk_categories.value_counts(sort=False)

                stats['risk_categories'] = {
                    'labels': risk_dist.index.tolist(),
                    'counts': risk_dist.values.tolist(),
                    'percentages': (risk_dist / len(cci_scores) * 100).round(2).tolist()
                }

            stats['cci_results_df'] = cci_results  # Store full results for potential export
            return stats

        except ImportError:
            return {'error': 'clifpy.utils.comorbidity module not available'}
        except Exception as e:
            return {'error': f'Error calculating CCI: {str(e)}'}

    def generate_hospital_diagnosis_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the hospital diagnosis table.

        Returns:
        --------
        pd.DataFrame
            Summary table with key metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        summary_data = []

        # Basic counts
        summary_data.append({
            'Metric': 'Total Diagnosis Records',
            'Value': f"{len(df):,}"
        })

        if 'hospitalization_id' in df.columns:
            summary_data.append({
                'Metric': 'Unique Hospitalizations',
                'Value': f"{df['hospitalization_id'].nunique():,}"
            })

        if 'diagnosis_code' in df.columns:
            summary_data.append({
                'Metric': 'Unique Diagnosis Codes',
                'Value': f"{df['diagnosis_code'].nunique():,}"
            })

        # Diagnoses per hospitalization
        if 'hospitalization_id' in df.columns:
            dx_per_hosp = df.groupby('hospitalization_id').size()
            summary_data.append({
                'Metric': 'Diagnoses per Hospitalization (mean ± SD)',
                'Value': f"{dx_per_hosp.mean():.1f} ± {dx_per_hosp.std():.1f}"
            })
            summary_data.append({
                'Metric': 'Diagnoses per Hospitalization (median [IQR])',
                'Value': f"{dx_per_hosp.median():.0f} [{dx_per_hosp.quantile(0.25):.0f}-{dx_per_hosp.quantile(0.75):.0f}]"
            })

        # Diagnosis Format Distribution
        if 'diagnosis_code_format' in df.columns:
            format_counts = df['diagnosis_code_format'].value_counts()
            total = len(df)

            # ICD-10CM
            icd10_count = format_counts.get('ICD10CM', 0)
            if icd10_count > 0:
                icd10_pct = (icd10_count / total * 100)
                summary_data.append({
                    'Metric': 'ICD-10CM Diagnoses',
                    'Value': f"{icd10_count:,} ({icd10_pct:.1f}%)"
                })

            # ICD-9CM
            icd9_count = format_counts.get('ICD9CM', 0)
            if icd9_count > 0:
                icd9_pct = (icd9_count / total * 100)
                summary_data.append({
                    'Metric': 'ICD-9CM Diagnoses',
                    'Value': f"{icd9_count:,} ({icd9_pct:.1f}%)"
                })

            # Other formats if any
            for format_type in format_counts.index:
                if format_type not in ['ICD10CM', 'ICD9CM']:
                    other_count = format_counts[format_type]
                    other_pct = (other_count / total * 100)
                    summary_data.append({
                        'Metric': f'{format_type} Diagnoses',
                        'Value': f"{other_count:,} ({other_pct:.1f}%)"
                    })

        # CCI Score statistics
        cci_stats = self.calculate_cci_distribution()
        if 'error' not in cci_stats:
            summary_data.append({
                'Metric': 'CCI Score (mean ± SD)',
                'Value': f"{cci_stats['mean']:.2f} ± {cci_stats['std']:.2f}"
            })
            summary_data.append({
                'Metric': 'CCI Score (median [IQR])',
                'Value': f"{cci_stats['median']:.0f} [{cci_stats['q1']:.0f}-{cci_stats['q3']:.0f}]"
            })

            # Add CCI risk categories if available
            if 'risk_categories' in cci_stats:
                for i, label in enumerate(cci_stats['risk_categories']['labels']):
                    count = cci_stats['risk_categories']['counts'][i]
                    pct = cci_stats['risk_categories']['percentages'][i]
                    summary_data.append({
                        'Metric': f'CCI Risk - {label}',
                        'Value': f"{count:,} ({pct:.1f}%)"
                    })
        else:
            # Add note about CCI unavailability with reason
            error_msg = cci_stats.get('error', 'Unknown error')
            # Shorten error message if it's too long
            if 'Found formats:' in error_msg:
                error_msg = error_msg.split('. Found formats:')[0]
            summary_data.append({
                'Metric': 'CCI Score',
                'Value': 'Not available'
            })
            summary_data.append({
                'Metric': 'CCI Note',
                'Value': error_msg
            })

        # Top 5 diagnosis codes
        top_codes = self.get_top_diagnosis_codes(n=5)
        if 'error' not in top_codes:
            for i, code in enumerate(top_codes['codes'][:5]):
                count = top_codes['counts'][i]
                pct = top_codes['percentages'][i]
                summary_data.append({
                    'Metric': f'Top Diagnosis #{i+1} - {code}',
                    'Value': f"{count:,} hospitalizations ({pct:.1f}%)"
                })

        return pd.DataFrame(summary_data)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform hospital_diagnosis data quality checks.

        Returns:
            Dictionary of quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        quality_checks = {}
        df = self.table.df

        # Check for duplicate diagnoses (same hospitalization + diagnosis code)
        if 'hospitalization_id' in df.columns and 'diagnosis_code' in df.columns:
            duplicates_mask = df.duplicated(subset=['hospitalization_id', 'diagnosis_code'], keep=False)
            duplicates = duplicates_mask.sum()

            # Get examples of duplicate records
            examples = None
            if duplicates > 0:
                example_cols = ['hospitalization_id', 'diagnosis_code', 'diagnosis_primary', 'poa_present']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[duplicates_mask][example_cols].head(10)

            quality_checks['duplicate_diagnoses'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        # Check for multiple primary diagnoses per hospitalization
        if 'hospitalization_id' in df.columns and 'diagnosis_primary' in df.columns:
            # Count primary diagnoses per hospitalization
            primary_dx = df[df['diagnosis_primary'] == 1].groupby('hospitalization_id').size()
            multiple_primary = (primary_dx > 1).sum()

            # Get examples
            examples = None
            if multiple_primary > 0:
                hosp_with_multiple = primary_dx[primary_dx > 1].index[:5]
                example_cols = ['hospitalization_id', 'diagnosis_code', 'diagnosis_primary']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[
                    (df['hospitalization_id'].isin(hosp_with_multiple)) &
                    (df['diagnosis_primary'] == 1)
                ][example_cols].head(10)

            quality_checks['multiple_primary_diagnoses'] = {
                'count': int(multiple_primary),
                'percentage': round((multiple_primary / df['hospitalization_id'].nunique() * 100)
                                  if df['hospitalization_id'].nunique() > 0 else 0, 2),
                'status': 'pass' if multiple_primary == 0 else 'warning',
                'examples': examples
            }

        # Check for invalid diagnosis code formats (basic length check)
        if 'diagnosis_code' in df.columns and 'diagnosis_code_format' in df.columns:
            invalid_codes = []

            # ICD-9 codes should be 3-5 characters (excluding dots)
            # ICD-10 codes should be 3-7 characters (excluding dots)
            for format_type in ['ICD9CM', 'ICD10CM']:
                format_mask = df['diagnosis_code_format'] == format_type
                if format_mask.sum() > 0:
                    codes = df.loc[format_mask, 'diagnosis_code'].astype(str)
                    # Remove dots for length check
                    codes_no_dots = codes.str.replace('.', '', regex=False)

                    if format_type == 'ICD9CM':
                        invalid_mask = (codes_no_dots.str.len() < 3) | (codes_no_dots.str.len() > 5)
                    else:  # ICD10CM
                        invalid_mask = (codes_no_dots.str.len() < 3) | (codes_no_dots.str.len() > 7)

                    invalid_codes.extend(df.loc[format_mask, :][invalid_mask].index.tolist())

            # Get examples
            examples = None
            if len(invalid_codes) > 0:
                example_cols = ['hospitalization_id', 'diagnosis_code', 'diagnosis_code_format']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df.loc[invalid_codes[:10], example_cols]

            quality_checks['invalid_code_lengths'] = {
                'count': len(invalid_codes),
                'percentage': round((len(invalid_codes) / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if len(invalid_codes) == 0 else 'warning',
                'examples': examples
            }

        return quality_checks