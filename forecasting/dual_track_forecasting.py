"""
Dual-Track Steel Demand Forecasting System
Maintains current 13-product forecasts while adding new hierarchical taxonomy.
Provides cross-validation and reconciliation between methodologies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import sys
import os
from datetime import datetime

# Add parent directories to path for imports
sys.path.append('..')
sys.path.append('../..')

from forecasting.hierarchical_forecasting import HierarchicalSteelForecastingFramework
from training.model_trainer import SteelDemandModelTrainer
from data.data_loader import SteelDemandDataLoader

class DualTrackSteelForecastingSystem:
    """
    Comprehensive dual-track forecasting system that maintains current model
    capabilities while adding hierarchical product taxonomy forecasting.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize dual-track forecasting system.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = SteelDemandDataLoader(config_path)
        self.hierarchical_framework = HierarchicalSteelForecastingFramework(config_path)
        
        # Current model product list (13 products)
        self.current_products = [
            "Total Production of Crude Steel",
            "Production of Hot Rolled Flat Products",
            "Production of Hot Rolled Long Products", 
            "Apparent Steel Use (crude steel equivalent)",
            "Apparent Steel Use (finished steel products)",
            "True Steel Use (finished steel equivalent)",
            "Production of Railway Track Material",
            "Total Production of Tubular Products",
            "Production of Wire Rod",
            "Production of Hot Rolled Coil, Sheet, and Strip (<3mm)",
            "Production of Non-metallic Coated Sheet and Strip",
            "Production of Other Metal Coated Sheet and Strip",
            "Apparent Steel Use (finished steel equivalent)"  # Duplicate entry
        ]
        
        # Model storage
        self.current_models = {}
        self.is_trained = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def train_current_model_system(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the current 13-product forecasting system using existing methodology.
        
        Args:
            historical_data: Historical data for training
            
        Returns:
            Dictionary of trained models for current products
        """
        self.logger.info("Training current model system (13 products) with renewable energy integration...")
        
        # Load current steel categories configuration
        steel_categories = pd.read_csv(f"{self.config_path}/steel_categories.csv")
        
        trained_models = {}
        
        # Train model for each current product
        for _, category in steel_categories.iterrows():
            category_name = category['category']
            target_column = category['target_column']
            
            if target_column in historical_data.columns:
                self.logger.info(f"Training current model for: {category_name}")
                
                # Prepare features (use traditional macro drivers + renewable energy time series)
                feature_columns = [
                    'Population', 'GDP_AUD_Real2015', 'IP_Index_AUD_Real2015',
                    'Iron_Ore_Production', 'Coal_Production',
                    'Wind_Onshore', 'Wind_Offshore', 'Solar_Grid', 'Solar_Distributed'
                ]
                
                # Add available feature columns
                available_features = [col for col in feature_columns if col in historical_data.columns]
                
                if len(available_features) >= 3:  # Minimum features required
                    X = historical_data[available_features].fillna(method='ffill').fillna(0)
                    y = historical_data[target_column].fillna(method='ffill').fillna(0)
                    
                    # Train using existing methodology
                    trainer = SteelDemandModelTrainer(self.config_path)
                    model = trainer.train_model_for_category(X, y, category=f"current_{category['category_code']}")
                    
                    trained_models[category['category_code']] = {
                        'model': model,
                        'category_name': category_name,
                        'target_column': target_column,
                        'features': available_features
                    }
                else:
                    self.logger.warning(f"Insufficient features for {category_name}")
        
        self.current_models = trained_models
        return trained_models
    
    def forecast_current_products(self, 
                                forecast_data: pd.DataFrame,
                                start_year: int = 2025,
                                end_year: int = 2050) -> pd.DataFrame:
        """
        Generate forecasts for current 13 products using existing methodology.
        
        Args:
            forecast_data: Forecast period data
            start_year: Start year for forecasting
            end_year: End year for forecasting
            
        Returns:
            DataFrame with current product forecasts
        """
        self.logger.info(f"Forecasting current products {start_year}-{end_year} with renewable energy time series...")
        
        # Filter forecast data
        forecast_period = forecast_data[
            (forecast_data['Year'] >= start_year) & 
            (forecast_data['Year'] <= end_year)
        ].copy()
        
        # Initialize results DataFrame
        current_forecasts = forecast_period[['Year']].copy()
        
        # Generate forecasts for each current product
        for product_code, model_info in self.current_models.items():
            model = model_info['model']
            features = model_info['features']
            category_name = model_info['category_name']
            
            # Prepare features for forecasting
            available_features = [col for col in features if col in forecast_period.columns]
            
            if len(available_features) >= 3:
                X_forecast = forecast_period[available_features].fillna(method='ffill').fillna(0)
                
                # Generate forecast
                forecast_values = model.predict(X_forecast)
                current_forecasts[category_name] = forecast_values
                
                self.logger.info(f"Generated forecast for {category_name}")
            else:
                self.logger.warning(f"Cannot forecast {category_name} - insufficient features")
                current_forecasts[category_name] = 0
        
        return current_forecasts
    
    def cross_validate_forecasting_approaches(self,
                                            current_forecasts: pd.DataFrame,
                                            hierarchical_forecasts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Cross-validate current and hierarchical forecasting approaches.
        
        Args:
            current_forecasts: Current model forecasts
            hierarchical_forecasts: Hierarchical model forecasts
            
        Returns:
            Dictionary of cross-validation results
        """
        self.logger.info("Cross-validating forecasting approaches...")
        
        validation_results = {}
        
        # Production vs Consumption reconciliation
        if 'Total Production of Crude Steel' in current_forecasts.columns:
            crude_steel_production = current_forecasts['Total Production of Crude Steel']
            total_steel_consumption = hierarchical_forecasts['level_0']['total_steel_demand']
            
            production_consumption_ratio = crude_steel_production / total_steel_consumption
            
            validation_results['production_consumption_ratio'] = {
                'mean_ratio': production_consumption_ratio.mean(),
                'ratio_range': (production_consumption_ratio.min(), production_consumption_ratio.max()),
                'expected_range': (0.80, 0.95),  # Expected production/consumption ratio
                'validation_passed': (production_consumption_ratio >= 0.75).all() and (production_consumption_ratio <= 1.0).all()
            }
        
        # Flat products reconciliation
        if 'Production of Hot Rolled Flat Products' in current_forecasts.columns:
            flat_production = current_forecasts['Production of Hot Rolled Flat Products']
            flat_consumption = hierarchical_forecasts['level_1']['FINISHED_FLAT']
            
            flat_ratio = flat_production / flat_consumption
            
            validation_results['flat_products_ratio'] = {
                'mean_ratio': flat_ratio.mean(),
                'ratio_range': (flat_ratio.min(), flat_ratio.max()),
                'expected_range': (0.55, 0.85),  # Expected flat production/consumption ratio
                'validation_passed': (flat_ratio >= 0.50).all() and (flat_ratio <= 0.90).all()
            }
        
        # Long products reconciliation
        if 'Production of Hot Rolled Long Products' in current_forecasts.columns:
            long_production = current_forecasts['Production of Hot Rolled Long Products']
            long_consumption = hierarchical_forecasts['level_1']['FINISHED_LONG']
            
            long_ratio = long_production / long_consumption
            
            validation_results['long_products_ratio'] = {
                'mean_ratio': long_ratio.mean(),
                'ratio_range': (long_ratio.min(), long_ratio.max()),
                'expected_range': (0.65, 0.95),  # Expected long production/consumption ratio
                'validation_passed': (long_ratio >= 0.60).all() and (long_ratio <= 1.0).all()
            }
        
        # Rails reconciliation
        if 'Production of Railway Track Material' in current_forecasts.columns:
            rails_production = current_forecasts['Production of Railway Track Material']
            
            # Sum rail products from Level 2
            rails_consumption = (
                hierarchical_forecasts['level_2']['RAILS_STANDARD'] +
                hierarchical_forecasts['level_2']['RAILS_HEAD_HARDENED']
            )
            
            rails_ratio = rails_production / rails_consumption
            
            validation_results['rails_ratio'] = {
                'mean_ratio': rails_ratio.mean(),
                'ratio_range': (rails_ratio.min(), rails_ratio.max()),
                'expected_range': (0.90, 1.05),  # Expected rails production/consumption ratio
                'validation_passed': (rails_ratio >= 0.85).all() and (rails_ratio <= 1.10).all()
            }
        
        # Overall validation summary
        all_validations_passed = all(
            result.get('validation_passed', False) 
            for result in validation_results.values()
        )
        
        validation_results['overall_validation'] = {
            'all_checks_passed': all_validations_passed,
            'number_of_checks': len(validation_results),
            'validation_summary': 'PASS' if all_validations_passed else 'FAIL'
        }
        
        return validation_results
    
    def generate_unified_forecasts(self,
                                 start_year: int = 2025,
                                 end_year: int = 2050) -> Dict[str, Any]:
        """
        Generate unified forecasts using both current and hierarchical approaches.
        
        Args:
            start_year: Start year for forecasting
            end_year: End year for forecasting
            
        Returns:
            Dictionary containing both current and hierarchical forecasts with validation
        """
        self.logger.info(f"Generating unified forecasts {start_year}-{end_year}...")
        
        # Load and prepare data
        all_data = self.data_loader.load_all_data()
        historical_data = all_data['macro_drivers_wm']  # Use macro drivers as historical data
        forecast_data = all_data['macro_drivers_wm']    # Same data for forecasting
        
        # Train models if not already trained
        if not self.is_trained:
            self.logger.info("Training both current and hierarchical models...")
            
            # Train current model system
            self.train_current_model_system(historical_data)
            
            # Hierarchical framework trains itself during forecast generation
            self.is_trained = True
        
        # Generate current model forecasts
        current_forecasts = self.forecast_current_products(
            forecast_data, start_year, end_year
        )
        
        # Generate hierarchical forecasts
        hierarchical_forecasts = self.hierarchical_framework.generate_hierarchical_forecasts(
            start_year, end_year
        )
        
        # Cross-validate approaches
        validation_results = self.cross_validate_forecasting_approaches(
            current_forecasts, hierarchical_forecasts
        )
        
        # Load historical steel data
        steel_data = all_data.get('wsa_steel_data', pd.DataFrame())
        
        # Create unified results
        unified_results = {
            'current_model_forecasts': current_forecasts,
            'hierarchical_forecasts': hierarchical_forecasts,
            'cross_validation': validation_results,
            'historical_data': steel_data,  # Include historical data
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'forecast_period': f"{start_year}-{end_year}",
                'current_products_count': len(self.current_products),
                'hierarchical_levels': 4,  # Level 0-3
                'methodology': 'dual_track_ensemble'
            }
        }
        
        return unified_results
    
    def export_unified_results(self, 
                             unified_results: Dict[str, Any],
                             output_dir: str = None) -> None:
        """
        Export unified forecasting results to files.
        
        Args:
            unified_results: Unified forecasting results
            output_dir: Output directory for files (if None, creates timestamped directory)
        """
        # Create timestamped output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"forecasts/unified_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Exporting unified results to {output_dir}...")
        
        # Export current model forecasts - forecast only version
        current_forecasts = unified_results['current_model_forecasts']
        current_filename = f"{output_dir}/current_model_forecasts_2025_2050.csv"
        current_forecasts.to_csv(current_filename, index=False)
        
        # Export current model forecasts with historical data (2004-2050)
        if 'historical_data' in unified_results:
            historical_data = unified_results['historical_data']
            # Merge historical with forecasts
            complete_current = pd.concat([
                historical_data[historical_data['Year'] < 2025],
                current_forecasts
            ], ignore_index=True).sort_values('Year')
            complete_filename = f"{output_dir}/current_model_forecasts_2004_2050.csv"
            complete_current.to_csv(complete_filename, index=False)
        
        # Export hierarchical forecasts - forecast only versions
        hierarchical_forecasts = unified_results['hierarchical_forecasts']
        for level, forecast_df in hierarchical_forecasts.items():
            hierarchical_filename = f"{output_dir}/hierarchical_{level}_forecasts_2025_2050.csv"
            forecast_df.to_csv(hierarchical_filename, index=False)
            
            # Export with historical data if available
            if 'historical_data' in unified_results:
                # Merge with historical data for complete timeseries
                hist_cols = set(historical_data.columns) & set(forecast_df.columns)
                if len(hist_cols) > 1:  # More than just 'Year'
                    complete_df = pd.concat([
                        historical_data[historical_data['Year'] < 2025][list(hist_cols)],
                        forecast_df[list(hist_cols)]
                    ], ignore_index=True).sort_values('Year')
                    complete_filename = f"{output_dir}/hierarchical_{level}_forecasts_2004_2050.csv"
                    complete_df.to_csv(complete_filename, index=False)
        
        # Export cross-validation results
        validation_results = unified_results['cross_validation']
        validation_df = pd.DataFrame.from_dict(validation_results, orient='index')
        validation_filename = f"{output_dir}/cross_validation_results.csv"
        validation_df.to_csv(validation_filename, index=True)
        
        # Export comprehensive summary
        summary_data = []
        
        # Current model summary
        if not current_forecasts.empty:
            for column in current_forecasts.columns:
                if column != 'Year':
                    summary_data.append({
                        'forecast_type': 'current_model',
                        'product_category': column,
                        'level': 'production_focus',
                        'avg_annual_demand_kt': current_forecasts[column].mean(),
                        'total_2025_2050_mt': current_forecasts[column].sum() / 1000,
                        'growth_rate_cagr': ((current_forecasts[column].iloc[-1] / current_forecasts[column].iloc[0]) ** (1/25) - 1) * 100
                    })
        
        # Hierarchical model summary
        for level, forecast_df in hierarchical_forecasts.items():
            if level != 'level_0':  # Skip level_0 to avoid duplication
                for column in forecast_df.columns:
                    if column != 'Year' and column != 'sectoral_breakdown':
                        summary_data.append({
                            'forecast_type': 'hierarchical_model',
                            'product_category': column,
                            'level': level,
                            'avg_annual_demand_kt': forecast_df[column].mean(),
                            'total_2025_2050_mt': forecast_df[column].sum() / 1000,
                            'growth_rate_cagr': ((forecast_df[column].iloc[-1] / forecast_df[column].iloc[0]) ** (1/25) - 1) * 100 if forecast_df[column].iloc[0] > 0 else 0
                        })
        
        summary_df = pd.DataFrame(summary_data)
        summary_filename = f"{output_dir}/unified_forecasting_summary.csv"
        summary_df.to_csv(summary_filename, index=False)
        
        # Export metadata
        metadata = unified_results['metadata']
        metadata_df = pd.DataFrame.from_dict(metadata, orient='index', columns=['value'])
        metadata_filename = f"{output_dir}/forecasting_metadata.csv"
        metadata_df.to_csv(metadata_filename, index=True)
        
        self.logger.info(f"Unified results exported successfully to {output_dir}")
        
        # Log validation summary
        overall_validation = validation_results.get('overall_validation', {})
        if overall_validation.get('all_checks_passed', False):
            self.logger.info("✅ All cross-validation checks PASSED")
        else:
            self.logger.warning("⚠️  Some cross-validation checks FAILED - review validation results")
    
    def generate_client_insights(self, unified_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate client-specific insights from unified forecasting results.
        
        Args:
            unified_results: Unified forecasting results
            
        Returns:
            Dictionary of client insights and recommendations
        """
        self.logger.info("Generating client-specific insights...")
        
        hierarchical_forecasts = unified_results['hierarchical_forecasts']
        client_insights = {}
        
        # Client product portfolio analysis
        if 'level_2' in hierarchical_forecasts:
            level_2_forecasts = hierarchical_forecasts['level_2']
            
            # Billets analysis
            commercial_billets = level_2_forecasts.get('BILLETS_COMMERCIAL', pd.Series([0]))
            sbq_billets = level_2_forecasts.get('BILLETS_SBQ', pd.Series([0]))
            degassed_billets = level_2_forecasts.get('BILLETS_DEGASSED', pd.Series([0]))
            
            client_insights['billets_portfolio'] = {
                'total_market_2025_2050_mt': (commercial_billets.sum() + sbq_billets.sum() + degassed_billets.sum()) / 1000,
                'commercial_share': commercial_billets.sum() / (commercial_billets.sum() + sbq_billets.sum() + degassed_billets.sum()) * 100,
                'sbq_share': sbq_billets.sum() / (commercial_billets.sum() + sbq_billets.sum() + degassed_billets.sum()) * 100,
                'degassed_growth_opportunity': degassed_billets.sum() / 1000,
                'market_trend': 'premium_shift' if degassed_billets.iloc[-1] > degassed_billets.iloc[0] * 2 else 'stable'
            }
            
            # Structural steel analysis
            beams = level_2_forecasts.get('STRUCTURAL_BEAMS', pd.Series([0]))
            columns = level_2_forecasts.get('STRUCTURAL_COLUMNS', pd.Series([0]))
            channels = level_2_forecasts.get('STRUCTURAL_CHANNELS', pd.Series([0]))
            angles = level_2_forecasts.get('STRUCTURAL_ANGLES', pd.Series([0]))
            
            client_insights['structural_steel_portfolio'] = {
                'total_market_2025_2050_mt': (beams.sum() + columns.sum() + channels.sum() + angles.sum()) / 1000,
                'beams_dominance': beams.sum() / (beams.sum() + columns.sum()) * 100,
                'infrastructure_vs_construction_trend': 'infrastructure_growth' if level_2_forecasts.get('TUBE_PIPE', pd.Series([0])).sum() > 0 else 'construction_focus'
            }
            
            # Rails analysis
            standard_rails = level_2_forecasts.get('RAILS_STANDARD', pd.Series([0]))
            head_hardened_rails = level_2_forecasts.get('RAILS_HEAD_HARDENED', pd.Series([0]))
            
            client_insights['rails_portfolio'] = {
                'total_market_2025_2050_mt': (standard_rails.sum() + head_hardened_rails.sum()) / 1000,
                'heavy_haul_share': head_hardened_rails.sum() / (standard_rails.sum() + head_hardened_rails.sum()) * 100,
                'mining_infrastructure_trend': 'growth' if head_hardened_rails.iloc[-1] > head_hardened_rails.iloc[0] else 'stable'
            }
        
        # Blue Sky opportunities
        if 'level_3' in hierarchical_forecasts:
            level_3_forecasts = hierarchical_forecasts['level_3']
            
            # Premium automotive billets
            premium_auto_billets = level_3_forecasts.get('BILLETS_DEG_PREMIUM_AUTO', pd.Series([0]))
            
            # Enhanced grade structural steel
            ub_300_plus = level_3_forecasts.get('UB_GRADE_300_PLUS', pd.Series([0]))
            uc_300_plus = level_3_forecasts.get('UC_GRADE_300_PLUS', pd.Series([0]))
            
            client_insights['blue_sky_opportunities'] = {
                'premium_automotive_market_mt': premium_auto_billets.sum() / 1000,
                'enhanced_structural_market_mt': (ub_300_plus.sum() + uc_300_plus.sum()) / 1000,
                'ev_driven_growth': premium_auto_billets.iloc[-1] / premium_auto_billets.iloc[0] if premium_auto_billets.iloc[0] > 0 else 1,
                'infrastructure_upgrade_trend': 'premium_shift' if (ub_300_plus.iloc[-1] + uc_300_plus.iloc[-1]) > 0 else 'standard_focus'
            }
        
        # Market positioning insights
        validation_results = unified_results['cross_validation']
        
        client_insights['market_positioning'] = {
            'production_vs_consumption_balance': validation_results.get('production_consumption_ratio', {}).get('mean_ratio', 'unknown'),
            'import_dependency_flat_products': 1 - validation_results.get('flat_products_ratio', {}).get('mean_ratio', 0.7),
            'domestic_strength_long_products': validation_results.get('long_products_ratio', {}).get('mean_ratio', 0.8),
            'rails_market_leadership': validation_results.get('rails_ratio', {}).get('mean_ratio', 0.95)
        }
        
        # Renewable energy impact on current model forecasts
        client_insights['renewable_energy_integration'] = {
            'renewable_features_included': 'Wind_Onshore, Wind_Offshore, Solar_Grid, Solar_Distributed',
            'track_a_enhanced': 'Traditional 13-product models now include renewable energy time series',
            'dual_track_benefit': 'Both production-focused and consumption-focused models benefit from renewable energy data'
        }
        
        return client_insights