"""
Renewable Energy Feature Engineering Module
Creates renewable energy specific features for steel demand forecasting.
Handles corrected steel intensities for distributed solar.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

class RenewableEnergyFeatureEngineering:
    """
    Feature engineering specifically for renewable energy steel demand.
    Implements corrected steel intensities and capacity-to-steel conversion.
    """
    
    def __init__(self, renewable_config_path: str = "config/renewable_energy_config.csv"):
        """
        Initialize renewable energy feature engineering.
        
        Args:
            renewable_config_path: Path to renewable energy configuration file
        """
        self.renewable_config = pd.read_csv(renewable_config_path)
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def calculate_renewable_steel_demand(self, renewable_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate steel demand from renewable energy capacity data.
        Uses corrected steel intensities for distributed solar.
        
        Args:
            renewable_data: DataFrame with renewable capacity columns
            
        Returns:
            DataFrame with renewable steel demand features
        """
        renewable_features = renewable_data.copy()
        
        # Get steel intensity coefficients from config
        steel_intensities = {}
        grid_intensities = {}
        
        for _, row in self.renewable_config.iterrows():
            tech = row['technology']
            steel_intensities[tech] = row['steel_intensity_tonnes_per_mw']
            grid_intensities[tech] = row['grid_infrastructure_tonnes_per_mw']
        
        # Calculate annual capacity additions
        renewable_features['wind_onshore_additions'] = renewable_features['Wind_Onshore'].diff().fillna(0)
        renewable_features['wind_offshore_additions'] = renewable_features['Wind_Offshore'].diff().fillna(0)
        renewable_features['solar_grid_additions'] = renewable_features['Solar_Grid'].diff().fillna(0)
        renewable_features['solar_distributed_additions'] = renewable_features['Solar_Distributed'].diff().fillna(0)
        
        # Calculate steel demand from new capacity (corrected intensities)
        # Convert from tonnes to kilotonnes (divide by 1000)
        renewable_features['steel_demand_wind_onshore'] = (
            renewable_features['wind_onshore_additions'] * steel_intensities['wind_onshore'] +
            renewable_features['wind_onshore_additions'] * grid_intensities['wind_onshore']
        ) / 1000
        
        renewable_features['steel_demand_wind_offshore'] = (
            renewable_features['wind_offshore_additions'] * steel_intensities['wind_offshore'] +
            renewable_features['wind_offshore_additions'] * grid_intensities['wind_offshore']
        ) / 1000
        
        renewable_features['steel_demand_solar_grid'] = (
            renewable_features['solar_grid_additions'] * steel_intensities['solar_grid'] +
            renewable_features['solar_grid_additions'] * grid_intensities['solar_grid']
        ) / 1000
        
        # Corrected: Distributed solar has minimal steel intensity and NO grid infrastructure
        renewable_features['steel_demand_solar_distributed'] = (
            renewable_features['solar_distributed_additions'] * steel_intensities['solar_distributed']
            # No grid infrastructure component for distributed solar
        ) / 1000
        
        # Total renewable energy steel demand
        renewable_features['total_renewable_steel_demand'] = (
            renewable_features['steel_demand_wind_onshore'] +
            renewable_features['steel_demand_wind_offshore'] +
            renewable_features['steel_demand_solar_grid'] +
            renewable_features['steel_demand_solar_distributed']
        )
        
        return renewable_features
    
    def create_renewable_features(self, renewable_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive renewable energy features for steel demand modeling.
        
        Args:
            renewable_data: DataFrame with renewable capacity data
            
        Returns:
            DataFrame with engineered renewable energy features
        """
        features = renewable_data.copy()
        
        # Calculate steel demand first
        features = self.calculate_renewable_steel_demand(features)
        
        # Total renewable capacity
        features['total_renewable_capacity'] = (
            features['Wind_Onshore'] + features['Wind_Offshore'] + 
            features['Solar_Grid'] + features['Solar_Distributed']
        )
        
        # Grid-connected vs distributed renewables
        features['grid_connected_renewables'] = (
            features['Wind_Onshore'] + features['Wind_Offshore'] + features['Solar_Grid']
        )
        features['distributed_renewables'] = features['Solar_Distributed']
        
        # Technology mix features
        features['wind_share'] = (
            (features['Wind_Onshore'] + features['Wind_Offshore']) / 
            features['total_renewable_capacity'].replace(0, np.nan)
        ).fillna(0)
        
        features['solar_share'] = (
            (features['Solar_Grid'] + features['Solar_Distributed']) / 
            features['total_renewable_capacity'].replace(0, np.nan)
        ).fillna(0)
        
        features['offshore_wind_ratio'] = (
            features['Wind_Offshore'] / 
            (features['Wind_Onshore'] + features['Wind_Offshore']).replace(0, np.nan)
        ).fillna(0)
        
        features['distributed_solar_ratio'] = (
            features['Solar_Distributed'] / 
            (features['Solar_Grid'] + features['Solar_Distributed']).replace(0, np.nan)
        ).fillna(0)
        
        # Steel intensity features
        features['weighted_steel_intensity'] = (
            (features['Wind_Onshore'] * 265 +  # 250 + 15 grid
             features['Wind_Offshore'] * 425 +  # 400 + 25 grid
             features['Solar_Grid'] * 55 +      # 45 + 10 grid
             features['Solar_Distributed'] * 8)  # 8 + 0 grid (corrected)
            / features['total_renewable_capacity'].replace(0, np.nan)
        ).fillna(0)
        
        # Growth rate features
        features['renewable_growth_rate'] = features['total_renewable_capacity'].pct_change().fillna(0)
        features['wind_onshore_growth_rate'] = features['Wind_Onshore'].pct_change().fillna(0)
        features['solar_distributed_growth_rate'] = features['Solar_Distributed'].pct_change().fillna(0)
        
        # Steel demand growth patterns
        features['steel_intensive_capacity_growth'] = (
            features['wind_onshore_additions'] + features['wind_offshore_additions'] + 
            features['solar_grid_additions']
        )
        features['steel_light_capacity_growth'] = features['solar_distributed_additions']
        
        # Policy milestone features
        features['offshore_wind_phase'] = (features['Wind_Offshore'] > 0).astype(int)
        features['high_solar_penetration'] = (features['Solar_Distributed'] > 15000).astype(int)
        
        # Rolling averages for smoothing
        features['renewable_capacity_ma_3'] = features['total_renewable_capacity'].rolling(window=3, min_periods=1).mean()
        features['renewable_steel_demand_ma_3'] = features['total_renewable_steel_demand'].rolling(window=3, min_periods=1).mean()
        
        return features
    
    def get_renewable_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get mapping of renewable energy features to their importance categories.
        
        Returns:
            Dictionary mapping feature names to importance categories
        """
        return {
            'total_renewable_steel_demand': 'primary_target',
            'steel_demand_wind_onshore': 'technology_specific',
            'steel_demand_wind_offshore': 'technology_specific', 
            'steel_demand_solar_grid': 'technology_specific',
            'steel_demand_solar_distributed': 'technology_specific',
            'weighted_steel_intensity': 'steel_efficiency',
            'grid_connected_renewables': 'infrastructure_intensive',
            'distributed_renewables': 'infrastructure_light',
            'renewable_growth_rate': 'market_dynamics',
            'offshore_wind_phase': 'policy_milestone',
            'steel_intensive_capacity_growth': 'high_steel_demand',
            'steel_light_capacity_growth': 'low_steel_demand'
        }