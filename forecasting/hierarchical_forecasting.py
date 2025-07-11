"""
Hierarchical Steel Demand Forecasting Framework
Implements Level 0-3 hierarchical forecasting with consistency constraints.
Integrates renewable energy and maintains current model compatibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os

# Add parent directories to path for imports
sys.path.append('..')
sys.path.append('../..')

from data.data_loader import SteelDemandDataLoader
from data.renewable_energy_features import RenewableEnergyFeatureEngineering
from data.hierarchical_features import HierarchicalProductFeatures
from models.ensemble_models import EnsembleSteelModel
from training.model_trainer import SteelDemandModelTrainer

class HierarchicalSteelForecastingFramework:
    """
    Comprehensive hierarchical forecasting framework for Australian steel demand.
    Implements Level 0-3 product taxonomy with renewable energy integration.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize hierarchical forecasting framework.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = SteelDemandDataLoader(config_path)
        self.renewable_features = RenewableEnergyFeatureEngineering(f"{config_path}/renewable_energy_config.csv")
        self.hierarchical_features = HierarchicalProductFeatures(config_path)
        
        # ML ensemble weights (baseline approach)
        self.ml_ensemble_weights = {
            'xgboost': 0.45,
            'random_forest': 0.40,
            'linear_regression': 0.15
        }
        
        # Initialize models dictionary
        self.models = {}
        self.is_trained = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare all data sources for hierarchical forecasting.
        Uses consolidated data approach consistent with Track A.
        
        Returns:
            Tuple of (historical_data, forecast_data)
        """
        self.logger.info("Loading and preparing data for hierarchical forecasting...")
        
        # Use consolidated data approach (same as Track A) to ensure feature consistency
        # First load all data if not already loaded
        self.data_loader.load_all_data()
        
        historical_data = self.data_loader.get_historical_data()
        projection_data = self.data_loader.get_projection_data()
        
        # Convert projection data to same format as historical for feature engineering
        forecast_data = projection_data.copy()
        
        # Add renewable energy features to both datasets
        historical_renewable = self.renewable_features.create_renewable_features(historical_data)
        forecast_renewable = self.renewable_features.create_renewable_features(forecast_data)
        
        # Add hierarchical product features
        historical_hierarchical = self.hierarchical_features.create_hierarchical_features(historical_renewable)
        forecast_hierarchical = self.hierarchical_features.create_hierarchical_features(forecast_renewable)
        
        self.logger.info(f"Prepared data with {len(historical_hierarchical.columns)} features")
        
        return historical_hierarchical, forecast_hierarchical
    
    def train_sectoral_models(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train ML models for each sector (construction, infrastructure, manufacturing, renewable).
        
        Args:
            historical_data: Historical data with all features
            
        Returns:
            Dictionary of trained sectoral models
        """
        self.logger.info("Training sectoral models...")
        
        sectoral_models = {}
        
        # Define sector targets (proxy variables from historical data)
        sector_targets = {
            'construction': 'construction_intensity_gdp',
            'infrastructure': 'infrastructure_intensity_population', 
            'manufacturing': 'manufacturing_intensity_ip',
            'renewable_energy': 'total_renewable_steel_demand'
        }
        
        # Train model for each sector
        for sector, target_column in sector_targets.items():
            if target_column in historical_data.columns:
                self.logger.info(f"Training {sector} model with target {target_column}")
                
                # Prepare features and target - use only macro indicators like Track A
                econ_indicators = self.data_loader.get_economic_indicators()
                feature_columns = econ_indicators['column_name'].tolist()
                
                # Ensure all feature columns exist in the data
                feature_columns = [col for col in feature_columns if col in historical_data.columns]
                
                X = historical_data[feature_columns].fillna(0)
                y = historical_data[target_column].fillna(0)
                
                # Train ensemble model for this sector
                trainer = SteelDemandModelTrainer(self.config_path)
                sector_model = trainer.train_model_for_category(X, y, category=f"{sector}_model")
                sectoral_models[sector] = sector_model
                
        self.models['sectoral'] = sectoral_models
        return sectoral_models
    
    def load_track_a_apparent_steel_use(self) -> pd.DataFrame:
        """
        Load Track A apparent steel use results to use as Level 0 baseline.
        
        Returns:
            DataFrame with Track A apparent steel use forecasts
        """
        self.logger.info("Loading Track A apparent steel use results for Level 0 alignment...")
        
        # Find the latest Track A forecast results
        import glob
        import os
        
        track_a_pattern = "forecasts/track_a_*/Ensemble_Forecasts_2025-2050.csv"
        track_a_files = glob.glob(track_a_pattern)
        
        if not track_a_files:
            raise FileNotFoundError("No Track A forecast results found. Run Track A first.")
        
        # Use the most recent Track A results
        latest_track_a = max(track_a_files, key=os.path.getctime)
        self.logger.info(f"Using Track A results from: {latest_track_a}")
        
        track_a_data = pd.read_csv(latest_track_a)
        
        # Extract apparent steel use (crude steel equivalent) as Level 0 total
        if 'Apparent Steel Use (crude steel equivalent)_Ensemble' in track_a_data.columns:
            level_0_data = track_a_data[['Year', 'Apparent Steel Use (crude steel equivalent)_Ensemble']].copy()
            level_0_data.rename(columns={'Apparent Steel Use (crude steel equivalent)_Ensemble': 'total_steel_demand'}, inplace=True)
        else:
            raise ValueError("Track A results missing 'Apparent Steel Use (crude steel equivalent)_Ensemble' column")
        
        return level_0_data

    def forecast_level_0_total_demand(self, 
                                    forecast_data: pd.DataFrame,
                                    sectoral_models: Dict[str, Any]) -> pd.DataFrame:
        """
        Use Track A apparent steel use as Level 0 total demand with stable sectoral breakdown.
        
        Args:
            forecast_data: Forecast period data (for year information)
            sectoral_models: Not used in new approach
            
        Returns:
            DataFrame with Level 0 total steel demand from Track A
        """
        self.logger.info("Using Track A apparent steel use as Level 0 total demand...")
        
        # Load Track A results directly
        level_0_forecasts = self.load_track_a_apparent_steel_use()
        
        # Add stable sectoral breakdown based on Australian steel industry proportions
        level_0_forecasts['sectoral_breakdown'] = None
        
        for i, row in level_0_forecasts.iterrows():
            year = row['Year']
            total_demand = row['total_steel_demand']
            
            # Use stable sectoral weights (no time variation)
            sectoral_weights = self.hierarchical_features.get_sectoral_weights_for_year(year)
            
            # Calculate sectoral breakdown maintaining Track A total
            sector_breakdown = {}
            for sector, weight in sectoral_weights.items():
                sector_key = sector.replace('gdp_', '').replace('wm_', '').replace('_ip', '')
                sector_breakdown[sector_key] = total_demand * weight
            
            level_0_forecasts.at[i, 'sectoral_breakdown'] = sector_breakdown
        
        self.logger.info(f"Level 0 aligned with Track A: {level_0_forecasts['total_steel_demand'].iloc[0]:.1f} kt in 2025")
        return level_0_forecasts
    
    def forecast_level_1_categories(self, level_0_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Disaggregate Level 0 to Level 1 product categories.
        
        Args:
            level_0_forecasts: Level 0 total demand forecasts
            
        Returns:
            DataFrame with Level 1 category forecasts
        """
        self.logger.info("Forecasting Level 1 product categories...")
        
        level_1_forecasts = level_0_forecasts[['Year']].copy()
        
        # Initialize Level 1 category columns
        level_1_categories = ['SEMI_FINISHED', 'FINISHED_FLAT', 'FINISHED_LONG', 'TUBE_PIPE']
        for category in level_1_categories:
            level_1_forecasts[category] = 0.0
        
        # Disaggregate each year
        for i, row in level_0_forecasts.iterrows():
            year = row['Year']
            total_demand = row['total_steel_demand']
            sectoral_breakdown = row['sectoral_breakdown']
            
            if sectoral_breakdown is not None:
                # Disaggregate using hierarchical features
                level_1_products = self.hierarchical_features.disaggregate_level_0_to_level_1(
                    total_demand, sectoral_breakdown, year
                )
                
                # Update forecasts
                for category, demand in level_1_products.items():
                    level_1_forecasts.at[i, category] = demand
        
        return level_1_forecasts
    
    def forecast_level_2_products(self, level_1_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Disaggregate Level 1 to Level 2 detailed products.
        
        Args:
            level_1_forecasts: Level 1 category forecasts
            
        Returns:
            DataFrame with Level 2 product forecasts
        """
        self.logger.info("Forecasting Level 2 detailed products...")
        
        level_2_forecasts = level_1_forecasts[['Year']].copy()
        
        # Get all Level 2 product codes
        level_2_products = self.hierarchical_features.level_2_config['level_2_code'].unique()
        
        # Initialize Level 2 product columns
        for product in level_2_products:
            level_2_forecasts[product] = 0.0
        
        # Disaggregate each year
        for i, row in level_1_forecasts.iterrows():
            level_1_demands = {
                'SEMI_FINISHED': row['SEMI_FINISHED'],
                'FINISHED_FLAT': row['FINISHED_FLAT'],
                'FINISHED_LONG': row['FINISHED_LONG'],
                'TUBE_PIPE': row['TUBE_PIPE']
            }
            
            # Disaggregate using hierarchical features
            level_2_demands = self.hierarchical_features.disaggregate_level_1_to_level_2(level_1_demands)
            
            # Update forecasts
            for product, demand in level_2_demands.items():
                level_2_forecasts.at[i, product] = demand
        
        return level_2_forecasts
    
    def forecast_level_3_specifications(self, level_2_forecasts: pd.DataFrame) -> pd.DataFrame:
        """
        Disaggregate Level 2 to Level 3 client product specifications.
        
        Args:
            level_2_forecasts: Level 2 product forecasts
            
        Returns:
            DataFrame with Level 3 specification forecasts (client products only)
        """
        self.logger.info("Forecasting Level 3 client product specifications...")
        
        level_3_forecasts = level_2_forecasts[['Year']].copy()
        
        # Get all Level 3 specification codes
        level_3_specs = self.hierarchical_features.level_3_config['level_3_code'].unique()
        
        # Initialize Level 3 specification columns
        for spec in level_3_specs:
            level_3_forecasts[spec] = 0.0
        
        # Disaggregate each year
        for i, row in level_2_forecasts.iterrows():
            level_2_demands = {}
            
            # Extract Level 2 demands from row
            for product in self.hierarchical_features.level_2_config['level_2_code'].unique():
                if product in row:
                    level_2_demands[product] = row[product]
            
            # Disaggregate using hierarchical features
            level_3_demands = self.hierarchical_features.disaggregate_level_2_to_level_3(level_2_demands)
            
            # Update forecasts
            for spec, demand in level_3_demands.items():
                level_3_forecasts.at[i, spec] = demand
        
        return level_3_forecasts
    
    def generate_hierarchical_forecasts(self, 
                                      start_year: int = 2025, 
                                      end_year: int = 2050) -> Dict[str, pd.DataFrame]:
        """
        Generate complete hierarchical forecasts for all levels.
        
        Args:
            start_year: Start year for forecasting
            end_year: End year for forecasting
            
        Returns:
            Dictionary with forecasts for all hierarchy levels
        """
        self.logger.info(f"Generating hierarchical forecasts {start_year}-{end_year}...")
        
        # Load and prepare data
        historical_data, forecast_data = self.load_and_prepare_data()
        
        # Filter forecast data to requested period
        forecast_data = forecast_data[
            (forecast_data['Year'] >= start_year) & 
            (forecast_data['Year'] <= end_year)
        ].copy()
        
        # Train sectoral models if not already trained
        if not self.is_trained:
            self.train_sectoral_models(historical_data)
            self.is_trained = True
        
        # Generate forecasts for each level
        level_0_forecasts = self.forecast_level_0_total_demand(
            forecast_data, self.models['sectoral']
        )
        
        level_1_forecasts = self.forecast_level_1_categories(level_0_forecasts)
        level_2_forecasts = self.forecast_level_2_products(level_1_forecasts)
        level_3_forecasts = self.forecast_level_3_specifications(level_2_forecasts)
        
        # Validate hierarchical consistency
        self.validate_forecast_consistency(
            level_0_forecasts, level_1_forecasts, level_2_forecasts, level_3_forecasts
        )
        
        return {
            'level_0': level_0_forecasts,
            'level_1': level_1_forecasts,
            'level_2': level_2_forecasts,
            'level_3': level_3_forecasts
        }
    
    def validate_forecast_consistency(self,
                                    level_0: pd.DataFrame,
                                    level_1: pd.DataFrame,
                                    level_2: pd.DataFrame,
                                    level_3: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate hierarchical consistency across all forecast levels.
        
        Args:
            level_0: Level 0 forecasts
            level_1: Level 1 forecasts  
            level_2: Level 2 forecasts
            level_3: Level 3 forecasts
            
        Returns:
            Dictionary of validation results
        """
        self.logger.info("Validating hierarchical forecast consistency...")
        
        validation_results = {}
        errors = []
        
        for idx, (i, row) in enumerate(level_0.iterrows()):
            year = row['Year']
            
            # Extract demands for this year
            l0_demand = row['total_steel_demand']
            
            l1_demands = {
                'SEMI_FINISHED': level_1.iloc[idx]['SEMI_FINISHED'],
                'FINISHED_FLAT': level_1.iloc[idx]['FINISHED_FLAT'],
                'FINISHED_LONG': level_1.iloc[idx]['FINISHED_LONG'],
                'TUBE_PIPE': level_1.iloc[idx]['TUBE_PIPE']
            }
            
            l2_demands = {}
            for product in self.hierarchical_features.level_2_config['level_2_code'].unique():
                if product in level_2.columns:
                    l2_demands[product] = level_2.iloc[idx][product]
            
            l3_demands = {}
            for spec in self.hierarchical_features.level_3_config['level_3_code'].unique():
                if spec in level_3.columns:
                    l3_demands[spec] = level_3.iloc[idx][spec]
            
            # Validate consistency for this year
            year_validation = self.hierarchical_features.validate_hierarchical_consistency(
                l0_demand, l1_demands, l2_demands, l3_demands
            )
            
            validation_results[year] = year_validation
            
            # Check for errors
            if not all(year_validation.values()):
                errors.append(f"Year {year}: Consistency issues - {year_validation}")
        
        if errors:
            self.logger.warning(f"Found {len(errors)} consistency issues")
            for error in errors[:5]:  # Show first 5 errors
                self.logger.warning(error)
        else:
            self.logger.info("All hierarchical consistency checks passed")
        
        return validation_results
    
    def export_forecasts(self, forecasts: Dict[str, pd.DataFrame], output_dir: str = None):
        """
        Export hierarchical forecasts to CSV files.
        
        Args:
            forecasts: Dictionary of forecast DataFrames
            output_dir: Output directory for forecast files (if None, creates timestamped directory)
        """
        # Create timestamped output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"forecasts/hierarchical_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Exporting hierarchical forecasts to {output_dir}...")
        
        for level, forecast_df in forecasts.items():
            filename = f"{output_dir}/{level}_steel_demand_forecasts_2025_2050.csv"
            forecast_df.to_csv(filename, index=False)
            self.logger.info(f"Exported {level} forecasts to {filename}")
        
        # Export client product summary
        if 'level_3' in forecasts:
            client_products = self.hierarchical_features.get_client_product_mapping()
            client_summary = forecasts['level_3'][['Year']].copy()
            
            for product_code, product_info in client_products.items():
                # Sum Level 3 specifications for each client product
                level_3_specs = [spec for spec in forecasts['level_3'].columns 
                               if spec in self.hierarchical_features.level_3_config['level_3_code'].values]
                
                product_total = 0
                for spec in level_3_specs:
                    spec_row = self.hierarchical_features.level_3_config[
                        self.hierarchical_features.level_3_config['level_3_code'] == spec
                    ]
                    if not spec_row.empty and spec_row.iloc[0]['level_2_code'] == product_code:
                        if spec in forecasts['level_3'].columns:
                            product_total += forecasts['level_3'][spec]
                
                client_summary[product_info['name']] = product_total
            
            client_filename = f"{output_dir}/client_product_portfolio_forecasts_2025_2050.csv"
            client_summary.to_csv(client_filename, index=False)
            self.logger.info(f"Exported client product summary to {client_filename}")
        
        self.logger.info("Hierarchical forecast export completed")