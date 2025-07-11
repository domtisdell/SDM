"""
Hierarchical Product Features Engineering Module
Creates features for hierarchical steel product taxonomy forecasting.
Supports Level 1-3 product disaggregation and sector mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

class HierarchicalProductFeatures:
    """
    Feature engineering for hierarchical steel product taxonomy.
    Creates features for Level 1-3 product forecasting and sector mapping.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize hierarchical product feature engineering.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Load configuration files
        self.level_1_config = pd.read_csv(f"{config_path}/level_1_categories.csv")
        self.level_2_config = pd.read_csv(f"{config_path}/level_2_products.csv")
        self.level_3_config = pd.read_csv(f"{config_path}/level_3_specifications.csv")
        self.sector_mapping = pd.read_csv(f"{config_path}/sector_to_level1_mapping.csv")
        self.sectoral_weights = pd.read_csv(f"{config_path}/sectoral_weights.csv")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def get_sectoral_weights_for_year(self, year: int) -> Dict[str, float]:
        """
        Get research-based sectoral weights with evidence-based time variation.
        Weights based on Infrastructure Australia data and empirical analysis.
        
        Args:
            year: Forecast year
            
        Returns:
            Dictionary of sectoral weights for the year
        """
        # Use research-based time periods reflecting renewable energy growth
        if year <= 2030:
            period = "2025-2030"
        elif year <= 2040:
            period = "2031-2040" 
        else:
            period = "2041-2050"
            
        period_weights = self.sectoral_weights[self.sectoral_weights['period'] == period].iloc[0]
        
        weights = {
            'gdp_construction': period_weights['gdp_construction'],
            'infrastructure_traditional': period_weights['infrastructure_traditional'],
            'manufacturing_ip': period_weights['manufacturing_ip'],
            'wm_renewable_energy': period_weights['wm_renewable_energy']
        }
        
        # Add other_sectors if present in configuration
        if 'other_sectors' in period_weights.index:
            weights['other_sectors'] = period_weights['other_sectors']
            
        return weights
    
    def disaggregate_level_0_to_level_1(self, 
                                       total_steel_demand: float,
                                       sectoral_breakdown: Dict[str, float],
                                       year: int) -> Dict[str, float]:
        """
        Disaggregate total steel demand to Level 1 product categories.
        
        Args:
            total_steel_demand: Total steel demand for the year
            sectoral_breakdown: Steel demand by sector
            year: Forecast year
            
        Returns:
            Dictionary of Level 1 product demands
        """
        level_1_products = {
            'SEMI_FINISHED': 0.0,
            'FINISHED_FLAT': 0.0,
            'FINISHED_LONG': 0.0,
            'TUBE_PIPE': 0.0
        }
        
        # Apply sector-to-product mapping
        for _, mapping_row in self.sector_mapping.iterrows():
            sector = mapping_row['sector']
            level_1_code = mapping_row['level_1_code']
            share = mapping_row['share']
            
            if sector in sectoral_breakdown:
                level_1_products[level_1_code] += sectoral_breakdown[sector] * share
        
        # Hierarchical consistency check
        level_1_total = sum(level_1_products.values())
        if abs(level_1_total - total_steel_demand) > 0.01 * total_steel_demand:
            self.logger.warning(f"Level 1 total ({level_1_total:.1f}) differs from Level 0 ({total_steel_demand:.1f})")
            
            # Normalize to ensure consistency
            normalization_factor = total_steel_demand / level_1_total if level_1_total > 0 else 1.0
            level_1_products = {k: v * normalization_factor for k, v in level_1_products.items()}
        
        return level_1_products
    
    def disaggregate_level_1_to_level_2(self, level_1_forecasts: Dict[str, float]) -> Dict[str, float]:
        """
        Disaggregate Level 1 categories to Level 2 products.
        
        Args:
            level_1_forecasts: Dictionary of Level 1 category demands
            
        Returns:
            Dictionary of Level 2 product demands
        """
        level_2_products = {}
        
        for _, product_row in self.level_2_config.iterrows():
            level_1_code = product_row['level_1_code']
            level_2_code = product_row['level_2_code']
            share = product_row['share_of_level_1']
            
            if level_1_code in level_1_forecasts:
                level_2_products[level_2_code] = level_1_forecasts[level_1_code] * share
        
        return level_2_products
    
    def disaggregate_level_2_to_level_3(self, level_2_forecasts: Dict[str, float]) -> Dict[str, float]:
        """
        Disaggregate Level 2 products to Level 3 specifications (client products only).
        
        Args:
            level_2_forecasts: Dictionary of Level 2 product demands
            
        Returns:
            Dictionary of Level 3 product demands
        """
        level_3_products = {}
        
        for _, spec_row in self.level_3_config.iterrows():
            level_2_code = spec_row['level_2_code']
            level_3_code = spec_row['level_3_code']
            share = spec_row['share_of_level_2']
            
            if level_2_code in level_2_forecasts:
                level_3_products[level_3_code] = level_2_forecasts[level_2_code] * share
        
        return level_3_products
    
    def create_hierarchical_features(self, base_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create hierarchical product features from base macro data.
        
        Args:
            base_data: DataFrame with macro economic data
            
        Returns:
            DataFrame with hierarchical product features
        """
        features = base_data.copy()
        
        # Sector intensity features (scaled to realistic steel consumption levels in kt)
        # Based on Australian steel consumption matching Track A apparent steel use (crude steel equivalent)
        # Calibrated to match Track A 2025 apparent steel use: 6,968.55 kt
        features['construction_intensity_gdp'] = features.get('GDP_AUD_Real2015', 0) * 4.07  # Calibrated to Track A (4.09 * 0.9952)
        features['infrastructure_intensity_population'] = features.get('Population', 0) * 134  # Calibrated to Track A (135 * 0.9952)
        features['manufacturing_intensity_ip'] = features.get('IP_Index_AUD_Real2015?', 0) * 82  # Calibrated to Track A (82 * 0.9952)
        
        # Product category elasticity features
        features['semi_finished_demand_proxy'] = (
            features['construction_intensity_gdp'] * 0.10 +  # Construction billets
            features['manufacturing_intensity_ip'] * 0.60     # Manufacturing billets/slabs
        )
        
        features['finished_long_demand_proxy'] = (
            features['construction_intensity_gdp'] * 0.70 +   # Structural steel
            features['infrastructure_intensity_population'] * 0.50  # Rails, bridges
        )
        
        features['finished_flat_demand_proxy'] = (
            features['construction_intensity_gdp'] * 0.15 +   # Construction plate
            features['manufacturing_intensity_ip'] * 0.25     # Automotive, appliances
        )
        
        features['tube_pipe_demand_proxy'] = (
            features['construction_intensity_gdp'] * 0.05 +   # Structural hollow sections
            features['infrastructure_intensity_population'] * 0.30  # Pipelines
        )
        
        # Client product concentration features
        features['client_product_intensity'] = (
            features['semi_finished_demand_proxy'] * 1.0 +    # High client exposure
            features['finished_long_demand_proxy'] * 0.5 +    # Medium client exposure
            features['finished_flat_demand_proxy'] * 0.1      # Low client exposure
        )
        
        # Blue Sky product opportunity features
        features['degassed_product_opportunity'] = (
            features.get('automotive_growth_proxy', 0) * 0.6 +  # Automotive degassed demand
            features.get('precision_manufacturing_proxy', 0) * 0.4  # Precision applications
        )
        
        # Market share and positioning features
        features['structural_steel_market_size'] = features['finished_long_demand_proxy'] * 0.45  # UB + UC share
        features['rail_market_size'] = features['finished_long_demand_proxy'] * 0.08  # Rail products share
        features['billet_market_size'] = features['semi_finished_demand_proxy'] * 0.80  # Billet share
        
        return features
    
    def get_client_product_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Get mapping of client products to their market characteristics.
        
        Returns:
            Dictionary mapping client products to market information
        """
        client_products = {}
        
        # Filter for client products only
        client_level_2 = self.level_2_config[self.level_2_config['client_product'] == 'yes']
        
        for _, product in client_level_2.iterrows():
            client_products[product['level_2_code']] = {
                'name': product['level_2_name'],
                'level_1_category': product['level_1_code'],
                'market_share': product['share_of_level_1'],
                'description': product['description']
            }
        
        return client_products
    
    def validate_hierarchical_consistency(self, 
                                        level_0: float,
                                        level_1: Dict[str, float],
                                        level_2: Dict[str, float],
                                        level_3: Dict[str, float]) -> Dict[str, bool]:
        """
        Validate hierarchical consistency across all levels.
        
        Args:
            level_0: Total steel demand
            level_1: Level 1 category demands
            level_2: Level 2 product demands  
            level_3: Level 3 specification demands
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        # Level 0 = Sum of Level 1 (exact)
        level_1_sum = sum(level_1.values())
        validation_results['level_0_1_consistency'] = abs(level_1_sum - level_0) < 0.01 * level_0
        
        # Level 1 = Sum of Level 2 (for complete coverage categories)
        complete_coverage_categories = ['SEMI_FINISHED', 'FINISHED_LONG']
        for category in complete_coverage_categories:
            if category in level_1:
                level_2_sum = sum(v for k, v in level_2.items() 
                                if self.level_2_config[self.level_2_config['level_2_code'] == k]['level_1_code'].iloc[0] == category)
                validation_results[f'{category}_consistency'] = abs(level_2_sum - level_1[category]) < 0.01 * level_1[category]
        
        # Level 2 >= Sum of Level 3 (partial coverage allowed)
        for level_2_code in level_2:
            level_3_sum = sum(v for k, v in level_3.items()
                            if self.level_3_config[self.level_3_config['level_3_code'] == k]['level_2_code'].iloc[0] == level_2_code)
            if level_3_sum > 0:
                validation_results[f'{level_2_code}_level_3_consistency'] = level_3_sum <= level_2[level_2_code]
        
        return validation_results