import pandas as pd
import os
import logging
from typing import List, Optional, Dict, Set
import argparse
from pathlib import Path

# Configure logging with UTF-8 encoding support
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("csv_merger.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class FeatureCSVMerger:
    """
    Scalable CSV merger for voice feature extraction outputs.
    Automatically handles column additions/deletions and missing files.
    """
    
    def __init__(self, output_file: str = "combined_voice_features.csv"):
        self.output_file = output_file
        self.merge_key = "filename"  # Primary key for merging
        
    def validate_csv(self, file_path: str) -> Optional[pd.DataFrame]:
        """Validate and load CSV file with error handling"""
        try:
            if not os.path.exists(file_path):
                logging.warning(f"CSV file not found: {file_path}")
                return None
                
            df = pd.read_csv(file_path)
            
            if df.empty:
                logging.warning(f"Empty CSV file: {file_path}")
                return None
                
            if self.merge_key not in df.columns:
                logging.error(f"Merge key '{self.merge_key}' not found in {file_path}")
                return None
                
            # Check for duplicate filenames
            duplicates = df[df[self.merge_key].duplicated()]
            if not duplicates.empty:
                logging.warning(f"Found {len(duplicates)} duplicate filenames in {file_path}")
                # Keep first occurrence of duplicates
                df = df.drop_duplicates(subset=[self.merge_key], keep='first')
                
            logging.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logging.error(f"Error loading {file_path}: {str(e)}")
            return None
    
    def analyze_columns(self, dataframes: List[pd.DataFrame]) -> Dict:
        """Analyze column overlap and differences between DataFrames"""
        if not dataframes:
            return {}
            
        all_columns = [set(df.columns) for df in dataframes]
        
        # Find common columns (excluding merge key)
        common_cols = set.intersection(*all_columns)
        common_cols.discard(self.merge_key)
        
        # Find unique columns for each DataFrame
        unique_cols = []
        for i, cols in enumerate(all_columns):
            unique = cols - set.union(*[all_columns[j] for j in range(len(all_columns)) if j != i])
            unique_cols.append(unique)
        
        analysis = {
            'common_columns': common_cols,
            'unique_columns': unique_cols,
            'total_columns': [len(cols) for cols in all_columns],
            'overlapping_features': len(common_cols)
        }
        
        return analysis
    
    def handle_column_conflicts(self, dataframes: List[pd.DataFrame], 
                              csv_names: List[str]) -> List[pd.DataFrame]:
        """Handle overlapping column names by adding prefixes"""
        if len(dataframes) <= 1:
            return dataframes
        
        # Find columns that exist in multiple DataFrames (excluding merge key and metadata)
        metadata_cols = {'patient_id', 'gender', 'age', 'recording_date', 'recording_time', 
                        'original_duration', 'processed_duration', 'filename'}
        
        all_columns = [set(df.columns) - metadata_cols for df in dataframes]
        
        # Find conflicting columns
        conflicting_cols = set()
        for i in range(len(all_columns)):
            for j in range(i + 1, len(all_columns)):
                conflicts = all_columns[i].intersection(all_columns[j])
                conflicting_cols.update(conflicts)
        
        if conflicting_cols:
            logging.warning(f"Found {len(conflicting_cols)} conflicting columns: {list(conflicting_cols)[:10]}...")
            
            # Add prefixes to conflicting columns
            processed_dfs = []
            for i, (df, csv_name) in enumerate(zip(dataframes, csv_names)):
                df_copy = df.copy()
                
                # Create prefix from filename
                prefix = Path(csv_name).stem.lower()
                if 'praat' in prefix or 'parselmouth' in prefix:
                    prefix = 'praat'
                elif 'mfcc' in prefix:
                    prefix = 'mfcc'
                else:
                    prefix = f'src{i+1}'
                
                # Rename conflicting columns
                rename_dict = {}
                for col in conflicting_cols:
                    if col in df.columns and col not in metadata_cols:
                        rename_dict[col] = f"{prefix}_{col}"
                
                if rename_dict:
                    df_copy = df_copy.rename(columns=rename_dict)
                    logging.info(f"Renamed {len(rename_dict)} columns in {csv_name} with prefix '{prefix}_'")
                
                processed_dfs.append(df_copy)
            
            return processed_dfs
        
        return dataframes
    
    def merge_csvs(self, csv_files: List[str], 
                   merge_strategy: str = 'outer') -> Optional[pd.DataFrame]:
        """
        Merge multiple CSV files on filename column
        
        Args:
            csv_files: List of CSV file paths
            merge_strategy: 'inner', 'outer', 'left', 'right'
        """
        if not csv_files:
            logging.error("No CSV files provided")
            return None
        
        # Load and validate all CSV files
        dataframes = []
        valid_files = []
        
        for csv_file in csv_files:
            df = self.validate_csv(csv_file)
            if df is not None:
                dataframes.append(df)
                valid_files.append(csv_file)
        
        if not dataframes:
            logging.error("No valid CSV files found")
            return None
        
        if len(dataframes) == 1:
            logging.info("Only one valid CSV file found, returning as-is")
            return dataframes[0]
        
        # Analyze columns
        column_analysis = self.analyze_columns(dataframes)
        logging.info(f"Column analysis: {column_analysis['overlapping_features']} overlapping features")
        
        # Handle column conflicts
        dataframes = self.handle_column_conflicts(dataframes, valid_files)
        
        # Perform iterative merging
        merged_df = dataframes[0]
        merge_stats = {
            'initial_rows': len(merged_df),
            'merge_steps': []
        }
        
        for i, df in enumerate(dataframes[1:], 1):
            before_rows = len(merged_df)
            before_cols = len(merged_df.columns)
            
            # Perform merge
            merged_df = pd.merge(
                merged_df, df,
                on=self.merge_key,
                how=merge_strategy,
                suffixes=('', f'_dup{i}')
            )
            
            after_rows = len(merged_df)
            after_cols = len(merged_df.columns)
            
            step_stats = {
                'step': i,
                'source_file': valid_files[i],
                'rows_before': before_rows,
                'rows_after': after_rows,
                'cols_before': before_cols,
                'cols_after': after_cols,
                'rows_change': after_rows - before_rows,
                'cols_added': after_cols - before_cols
            }
            
            merge_stats['merge_steps'].append(step_stats)
            logging.info(f"Merge step {i}: {step_stats['rows_change']:+d} rows, "
                        f"{step_stats['cols_added']:+d} columns")
        
        merge_stats['final_rows'] = len(merged_df)
        merge_stats['final_columns'] = len(merged_df.columns)
        
        # Generate merge report
        self._generate_merge_report(merge_stats, valid_files, column_analysis)
        
        return merged_df
    
    def _generate_merge_report(self, merge_stats: Dict, files: List[str], 
                              column_analysis: Dict):
        """Generate detailed merge report"""
        report_lines = [
            "="*60,
            "VOICE FEATURES CSV MERGE REPORT",
            "="*60,
            f"Input Files: {len(files)}",
            f"Merge Strategy: outer join on '{self.merge_key}'",
            f"Final Dataset: {merge_stats['final_rows']} rows x {merge_stats['final_columns']} columns",
            "",
            "Source Files:"
        ]
        
        for i, file in enumerate(files):
            report_lines.append(f"  {i+1}. {os.path.basename(file)}")
        
        report_lines.extend([
            "",
            "Column Analysis:",
            f"  Common features: {len(column_analysis.get('common_columns', []))}",
            f"  Total unique columns per file: {column_analysis.get('total_columns', [])}",
            ""
        ])
        
        if merge_stats.get('merge_steps'):
            report_lines.append("Merge Steps:")
            for step in merge_stats['merge_steps']:
                report_lines.append(f"  Step {step['step']}: {os.path.basename(step['source_file'])}")
                report_lines.append(f"    Rows: {step['rows_before']} to {step['rows_after']} ({step['rows_change']:+d})")
                report_lines.append(f"    Cols: {step['cols_before']} to {step['cols_after']} ({step['cols_added']:+d})")
        
        report_lines.append("="*60)
        
        report_text = "\n".join(report_lines)
        logging.info(f"\n{report_text}")
        
        # Save report to file with UTF-8 encoding
        try:
            with open(f"{Path(self.output_file).stem}_merge_report.txt", 'w', encoding='utf-8') as f:
                f.write(report_text)
        except Exception as e:
            logging.error(f"Error saving merge report: {str(e)}")
    
    def save_merged_data(self, merged_df: pd.DataFrame, 
                        include_metadata: bool = True) -> bool:
        """Save merged DataFrame with optional metadata organization"""
        try:
            if include_metadata:
                # Organize columns: metadata first, then features
                metadata_cols = ['filename', 'patient_id', 'gender', 'age', 
                               'recording_date', 'recording_time', 
                               'original_duration', 'processed_duration']
                
                # Get existing metadata columns
                existing_metadata = [col for col in metadata_cols if col in merged_df.columns]
                
                # Get feature columns (everything else)
                feature_cols = [col for col in merged_df.columns if col not in existing_metadata]
                feature_cols.sort()  # Sort features alphabetically
                
                # Reorder columns
                final_columns = existing_metadata + feature_cols
                merged_df = merged_df[final_columns]
            
            # Save to CSV
            merged_df.to_csv(self.output_file, index=False)
            
            logging.info(f"Merged dataset saved to: {self.output_file}")
            logging.info(f"Final shape: {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")
            
            # Generate summary statistics
            self._generate_data_summary(merged_df)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving merged data: {str(e)}")
            return False
    
    def _generate_data_summary(self, df: pd.DataFrame):
        """Generate summary statistics of merged dataset"""
        summary = {
            "Total Records": len(df),
            "Total Features": len(df.columns),
            "Missing Values": df.isnull().sum().sum(),
            "Duplicate Filenames": df[self.merge_key].duplicated().sum(),
            "Unique Patients": df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A',
            "Gender Distribution": df['gender'].value_counts().to_dict() if 'gender' in df.columns else 'N/A'
        }
        
        logging.info("Dataset Summary:")
        for key, value in summary.items():
            logging.info(f"  {key}: {value}")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Merge voice feature CSV files")
    parser.add_argument('csv_files', nargs='+', help="CSV files to merge")
    parser.add_argument('--output', '-o', default="combined_voice_features.csv",
                       help="Output filename (default: combined_voice_features.csv)")
    parser.add_argument('--strategy', choices=['inner', 'outer', 'left', 'right'],
                       default='outer', help="Merge strategy (default: outer)")
    
    args = parser.parse_args()
    
    # Initialize merger
    merger = FeatureCSVMerger(output_file=args.output)
    
    # Perform merge
    merged_df = merger.merge_csvs(args.csv_files, merge_strategy=args.strategy)
    
    if merged_df is not None:
        success = merger.save_merged_data(merged_df)
        if success:
            logging.info("Merge completed successfully!")
        else:
            logging.error("Failed to save merged data")
    else:
        logging.error("Merge failed")

if __name__ == "__main__":
    # Example usage when running as script
    if len(os.sys.argv) == 1:  # No command line arguments
        # Default behavior - look for common CSV names
        default_files = ["voice_features.csv"]  # Add more default names as needed
        
        existing_files = [f for f in default_files if os.path.exists(f)]
        
        if existing_files:
            merger = FeatureCSVMerger()
            merged_df = merger.merge_csvs(existing_files)
            if merged_df is not None:
                merger.save_merged_data(merged_df)
        else:
            logging.info("No default CSV files found. Use command line arguments:")
            logging.info("python csv_merger.py file1.csv file2.csv [--output combined.csv]")
    else:
        main()