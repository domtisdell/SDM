"""
Steel Demand ML Model - Data Loader Module
Loads and validates all data from CSV files with no hardcoded values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

class SteelDemandDataLoader:
    """
    Data loader for Australian Steel Demand Model ML system.
    All parameters and assumptions loaded from CSV configuration files.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize data loader with configuration path.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path)
        self.data = {}
        self.config = {}
        self.logger = self._setup_logging()
        
        # Load configurations first
        self._load_configurations()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_configurations(self) -> None:
        """Load all configuration files."""
        try:
            # Model configuration
            model_config_path = self.config_path / "model_config.csv"
            if model_config_path.exists():
                config_df = pd.read_csv(model_config_path, comment='#')
                self.config['model'] = dict(zip(config_df['parameter'], config_df['value']))
                self.logger.info(f"Loaded {len(self.config['model'])} model parameters")
            
            # Data sources configuration
            data_sources_path = self.config_path / "data_sources.csv"
            if data_sources_path.exists():
                self.config['data_sources'] = pd.read_csv(data_sources_path)
                self.logger.info(f"Loaded {len(self.config['data_sources'])} data source definitions")
            
            # Steel categories configuration
            steel_categories_path = self.config_path / "steel_categories.csv"
            if steel_categories_path.exists():
                self.config['steel_categories'] = pd.read_csv(steel_categories_path)
                self.logger.info(f"Loaded {len(self.config['steel_categories'])} steel categories")
            
            # Regional factors configuration
            regional_factors_path = self.config_path / "regional_adjustment_factors.csv"
            if regional_factors_path.exists():
                self.config['regional_factors'] = pd.read_csv(regional_factors_path)
                self.logger.info(f"Loaded {len(self.config['regional_factors'])} regional factors")
            
            # Economic indicators configuration
            economic_indicators_path = self.config_path / "economic_indicators.csv"
            if economic_indicators_path.exists():
                self.config['economic_indicators'] = pd.read_csv(economic_indicators_path)
                self.logger.info(f"Loaded {len(self.config['economic_indicators'])} economic indicators")
            
            # Validation benchmarks configuration
            validation_benchmarks_path = self.config_path / "validation_benchmarks.csv"
            if validation_benchmarks_path.exists():
                self.config['validation_benchmarks'] = pd.read_csv(validation_benchmarks_path)
                self.logger.info(f"Loaded {len(self.config['validation_benchmarks'])} validation benchmarks")
                
        except Exception as e:
            self.logger.error(f"Error loading configurations: {str(e)}")
            raise
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all required datasets based on data sources configuration.
        
        Returns:
            Dictionary containing all loaded datasets
        """
        try:
            if 'data_sources' not in self.config:
                raise ValueError("Data sources configuration not loaded")
            
            for _, source in self.config['data_sources'].iterrows():
                source_name = source['source_name']
                file_path = source['file_path']
                
                # Handle relative paths
                if not file_path.startswith('/'):
                    file_path = self.config_path.parent / file_path
                else:
                    file_path = Path(file_path)
                
                if file_path.exists() and file_path.suffix == '.csv':
                    self.data[source_name] = pd.read_csv(file_path)
                    self.logger.info(f"Loaded {source_name}: {self.data[source_name].shape}")
                    
                    # Validate data quality if score provided
                    if 'quality_score' in source and source['quality_score'] < 0.8:
                        self.logger.warning(f"Low quality score for {source_name}: {source['quality_score']}")
                else:
                    self.logger.warning(f"File not found or not CSV: {file_path}")
            
            # Validate required datasets
            self._validate_required_data()
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_required_data(self) -> None:
        """Validate that all required datasets are loaded."""
        required_datasets = [
            'wsa_steel_data',
            'macro_drivers_wm'
        ]
        
        missing_datasets = [ds for ds in required_datasets if ds not in self.data]
        if missing_datasets:
            raise ValueError(f"Missing required datasets: {missing_datasets}")
    
    def get_historical_data(self, start_year: Optional[int] = None, 
                          end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Get historical steel production/consumption and macro driver data.
        Combines WSA steel data with WM macro economic data for 2004-2023.
        
        Args:
            start_year: Start year for data (None for all available, default 2004)
            end_year: End year for data (None for all available, default 2023)
            
        Returns:
            Combined historical dataset
        """
        try:
            # Get WSA steel data
            wsa_data = self.data.get('wsa_steel_data')
            if wsa_data is None:
                raise ValueError("WSA steel data not loaded")
            
            # Get WM macro data 
            macro_data = self.data.get('macro_drivers_wm')
            if macro_data is None:
                raise ValueError("WM macro drivers data not loaded")
            
            # Set default year ranges for historical data
            if start_year is None:
                start_year = 2004
            if end_year is None:
                end_year = 2023
            
            # Filter WSA data for historical period (2004-2023)
            wsa_historical = wsa_data[
                (wsa_data['Year'] >= start_year) & 
                (wsa_data['Year'] <= end_year)
            ].copy()
            
            # Get steel categories from configuration
            steel_categories = self.get_steel_categories()
            target_columns = steel_categories['target_column'].tolist()
            
            # Select required steel columns from WSA data
            wsa_columns = ['Year'] + [col for col in target_columns if col in wsa_data.columns]
            wsa_selected = wsa_historical[wsa_columns].copy()
            
            # Filter macro data for historical period
            macro_historical = macro_data[
                (macro_data['Year'] >= start_year) & 
                (macro_data['Year'] <= end_year)
            ].copy()
            
            # Get economic indicators from configuration
            econ_indicators = self.get_economic_indicators()
            macro_columns = ['Year'] + econ_indicators['column_name'].tolist()
            
            # Select required macro columns (handle missing columns gracefully)
            available_macro_cols = ['Year'] + [col for col in econ_indicators['column_name'].tolist() 
                                             if col in macro_data.columns]
            macro_selected = macro_historical[available_macro_cols].copy()
            
            # Merge steel and macro data
            historical_data = pd.merge(
                wsa_selected,
                macro_selected,
                on='Year',
                how='inner'
            )
            
            # Remove rows with all NaN values in steel columns
            steel_cols = [col for col in target_columns if col in historical_data.columns]
            historical_data = historical_data.dropna(subset=steel_cols, how='all')
            
            self.logger.info(f"Created consolidated historical dataset: {historical_data.shape}")
            self.logger.info(f"Years covered: {historical_data['Year'].min()}-{historical_data['Year'].max()}")
            self.logger.info(f"Steel categories included: {steel_cols}")
            self.logger.info(f"Macro indicators included: {[col for col in available_macro_cols if col != 'Year']}")
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error creating historical dataset: {str(e)}")
            raise
    
    def get_projection_data(self, start_year: Optional[int] = None,
                           end_year: Optional[int] = None) -> pd.DataFrame:
        """
        Get macro driver projection data for forecasting.
        Uses WM macro data for projections 2024-2050.
        
        Args:
            start_year: Start year for projections (default 2024)
            end_year: End year for projections (default 2050)
            
        Returns:
            Projection dataset
        """
        try:
            # Get WM macro data 
            macro_data = self.data.get('macro_drivers_wm')
            if macro_data is None:
                raise ValueError("WM macro drivers data not loaded")
            
            # Set default year ranges for projections
            if start_year is None:
                start_year = 2024
            if end_year is None:
                end_year = 2050
            
            # Filter macro data for projection period
            projections = macro_data[
                (macro_data['Year'] >= start_year) & 
                (macro_data['Year'] <= end_year)
            ].copy()
            
            # Get economic indicators from configuration
            econ_indicators = self.get_economic_indicators()
            
            # Select required macro columns (handle missing columns gracefully)
            available_macro_cols = ['Year'] + [col for col in econ_indicators['column_name'].tolist() 
                                             if col in macro_data.columns]
            projections_selected = projections[available_macro_cols].copy()
            
            self.logger.info(f"Created projection dataset: {projections_selected.shape}")
            self.logger.info(f"Projection years: {projections_selected['Year'].min()}-{projections_selected['Year'].max()}")
            self.logger.info(f"Macro indicators included: {[col for col in available_macro_cols if col != 'Year']}")
            
            return projections_selected
            
        except Exception as e:
            self.logger.error(f"Error creating projection dataset: {str(e)}")
            raise
    
    def get_steel_categories(self) -> pd.DataFrame:
        """Get steel category definitions and properties."""
        return self.config['steel_categories'].copy()
    
    def get_economic_indicators(self) -> pd.DataFrame:
        """Get economic indicator definitions and weights."""
        return self.config['economic_indicators'].copy()
    
    def get_regional_factors(self) -> pd.DataFrame:
        """Get regional adjustment factors."""
        return self.config['regional_factors'].copy()
    
    def get_model_config(self, parameter: Optional[str] = None) -> Any:
        """
        Get model configuration parameters.
        
        Args:
            parameter: Specific parameter name (None for all)
            
        Returns:
            Parameter value or all parameters dict
        """
        if parameter is None:
            return self.config['model'].copy()
        
        if parameter not in self.config['model']:
            raise KeyError(f"Parameter '{parameter}' not found in model configuration")
        
        # Parse parameter value based on type
        value = self.config['model'][parameter]
        
        # Handle list parameters
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                # Parse list string to actual list
                import ast
                return ast.literal_eval(value)
            except:
                return value
        
        # Try to convert to appropriate type
        try:
            # Try int first
            if '.' not in str(value):
                return int(value)
            else:
                return float(value)
        except:
            # Handle boolean
            if str(value).lower() in ['true', 'false']:
                return str(value).lower() == 'true'
            # Return as string
            return str(value)
    
    def get_validation_benchmarks(self, category: Optional[str] = None) -> pd.DataFrame:
        """
        Get validation benchmarks for model performance.
        
        Args:
            category: Steel category (None for all categories)
            
        Returns:
            Benchmark dataset
        """
        benchmarks = self.config['validation_benchmarks'].copy()
        
        if category is not None:
            benchmarks = benchmarks[benchmarks['category'] == category]
        
        return benchmarks
    
    def validate_data_quality(self) -> Dict[str, Any]:
        """
        Validate data quality across all loaded datasets.
        
        Returns:
            Data quality report
        """
        quality_report = {
            'datasets': {},
            'overall_quality': 'PASS',
            'issues': []
        }
        
        try:
            for dataset_name, df in self.data.items():
                dataset_quality = {
                    'shape': df.shape,
                    'missing_values': df.isnull().sum().sum(),
                    'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
                    'duplicates': df.duplicated().sum(),
                    'data_types': df.dtypes.to_dict()
                }
                
                # Check for critical issues
                if dataset_quality['missing_percentage'] > 10:
                    quality_report['issues'].append(f"{dataset_name}: High missing values ({dataset_quality['missing_percentage']:.1f}%)")
                    quality_report['overall_quality'] = 'WARNING'
                
                if dataset_quality['duplicates'] > 0:
                    quality_report['issues'].append(f"{dataset_name}: {dataset_quality['duplicates']} duplicate rows")
                
                quality_report['datasets'][dataset_name] = dataset_quality
            
            # Check for critical configuration issues
            if 'steel_categories' not in self.config:
                quality_report['issues'].append("Steel categories configuration missing")
                quality_report['overall_quality'] = 'FAIL'
            
            if 'economic_indicators' not in self.config:
                quality_report['issues'].append("Economic indicators configuration missing")
                quality_report['overall_quality'] = 'FAIL'
            
            self.logger.info(f"Data quality validation completed: {quality_report['overall_quality']}")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Error in data quality validation: {str(e)}")
            quality_report['overall_quality'] = 'FAIL'
            quality_report['issues'].append(f"Validation error: {str(e)}")
            return quality_report
    
    def export_data_summary(self, output_path: str = "data_summary.csv") -> None:
        """
        Export summary of all loaded data to CSV.
        
        Args:
            output_path: Path for output CSV file
        """
        try:
            summary_data = []
            
            for dataset_name, df in self.data.items():
                summary_data.append({
                    'dataset': dataset_name,
                    'rows': df.shape[0],
                    'columns': df.shape[1],
                    'missing_values': df.isnull().sum().sum(),
                    'data_types': str(df.dtypes.value_counts().to_dict()),
                    'date_range': f"{df['Year'].min()}-{df['Year'].max()}" if 'Year' in df.columns else 'N/A'
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_path, index=False)
            
            self.logger.info(f"Data summary exported to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting data summary: {str(e)}")
            raise