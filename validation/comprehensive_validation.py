"""
Comprehensive Validation Framework
Implements cross-validation, consistency checking, and benchmark validation
for the hierarchical steel demand forecasting system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

class ComprehensiveValidationFramework:
    """
    Comprehensive validation framework for hierarchical steel demand forecasting.
    Validates accuracy, consistency, and economic plausibility.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize validation framework.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = config_path
        self.logger = self._setup_logging()
        
        # Load validation benchmarks
        self.validation_benchmarks = pd.read_csv(f"{config_path}/validation_benchmarks.csv")
        
        # Define validation thresholds
        self.accuracy_thresholds = {
            'level_0': {'mape': 3.5, 'r2': 0.94},
            'level_1': {'mape': 6.0, 'r2': 0.88},
            'level_2': {'mape': 10.0, 'r2': 0.80},
            'level_3': {'mape': 15.0, 'r2': 0.70}
        }
        
        # Infrastructure Australia validation constraints
        self.infrastructure_constraints = {
            'total_steel_5_years': 8000000,  # 8M tonnes over 5 years
            'annual_average': 1600000,       # 1.6M tonnes per year average
            'energy_transition_share': 0.15  # 15% of total steel demand
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def calculate_forecast_accuracy_metrics(self, 
                                          actual: pd.Series, 
                                          predicted: pd.Series) -> Dict[str, float]:
        """
        Calculate comprehensive accuracy metrics for forecasts.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Handle missing values
        mask = ~(pd.isna(actual) | pd.isna(predicted))
        actual_clean = actual[mask]
        predicted_clean = predicted[mask]
        
        if len(actual_clean) == 0:
            return {'mape': np.inf, 'rmse': np.inf, 'r2': -np.inf, 'mae': np.inf}
        
        # Calculate metrics
        mape = np.mean(np.abs((actual_clean - predicted_clean) / actual_clean)) * 100
        rmse = np.sqrt(mean_squared_error(actual_clean, predicted_clean))
        r2 = r2_score(actual_clean, predicted_clean)
        mae = mean_absolute_error(actual_clean, predicted_clean)
        
        # Directional accuracy
        actual_changes = np.diff(actual_clean)
        predicted_changes = np.diff(predicted_clean)
        
        if len(actual_changes) > 0:
            directional_accuracy = np.mean(
                np.sign(actual_changes) == np.sign(predicted_changes)
            ) * 100
        else:
            directional_accuracy = np.nan
        
        return {
            'mape': mape,
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'directional_accuracy': directional_accuracy
        }
    
    def validate_hierarchical_consistency(self, 
                                        forecasts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate consistency across hierarchical levels.
        
        Args:
            forecasts: Dictionary of forecasts for all levels
            
        Returns:
            Dictionary of consistency validation results
        """
        self.logger.info("Validating hierarchical consistency...")
        
        consistency_results = {}
        
        if 'level_0' not in forecasts or 'level_1' not in forecasts:
            self.logger.warning("Missing required forecast levels for consistency validation")
            return consistency_results
        
        level_0_df = forecasts['level_0']
        level_1_df = forecasts['level_1']
        
        # Level 0 vs Level 1 consistency
        for i, row in level_0_df.iterrows():
            year = row['Year']
            level_0_total = row['total_steel_demand']
            
            # Sum Level 1 categories
            level_1_total = (
                level_1_df.iloc[i]['SEMI_FINISHED'] +
                level_1_df.iloc[i]['FINISHED_FLAT'] +
                level_1_df.iloc[i]['FINISHED_LONG'] +
                level_1_df.iloc[i]['TUBE_PIPE']
            )
            
            # Calculate consistency error
            consistency_error = abs(level_0_total - level_1_total) / level_0_total * 100
            consistency_results[f'year_{year}_l0_l1_error'] = consistency_error
        
        # Level 1 vs Level 2 consistency (for complete coverage categories)
        if 'level_2' in forecasts:
            level_2_df = forecasts['level_2']
            
            # Semi-finished consistency (should be 100% coverage)
            for i, row in level_1_df.iterrows():
                year = level_0_df.iloc[i]['Year']
                level_1_semi = row['SEMI_FINISHED']
                
                # Sum semi-finished Level 2 products
                level_2_semi = (
                    level_2_df.iloc[i]['BILLETS_COMMERCIAL'] +
                    level_2_df.iloc[i]['BILLETS_SBQ'] +
                    level_2_df.iloc[i]['SLABS_STANDARD'] +
                    level_2_df.iloc[i]['BILLETS_DEGASSED'] +
                    level_2_df.iloc[i]['SLABS_DEGASSED']
                )
                
                consistency_error = abs(level_1_semi - level_2_semi) / level_1_semi * 100 if level_1_semi > 0 else 0
                consistency_results[f'year_{year}_semi_finished_error'] = consistency_error
            
            # Finished long consistency (should be 100% coverage)
            for i, row in level_1_df.iterrows():
                year = level_0_df.iloc[i]['Year']
                level_1_long = row['FINISHED_LONG']
                
                # Sum finished long Level 2 products
                level_2_long = (
                    level_2_df.iloc[i]['STRUCTURAL_BEAMS'] +
                    level_2_df.iloc[i]['STRUCTURAL_COLUMNS'] +
                    level_2_df.iloc[i]['STRUCTURAL_CHANNELS'] +
                    level_2_df.iloc[i]['STRUCTURAL_ANGLES'] +
                    level_2_df.iloc[i]['RAILS_STANDARD'] +
                    level_2_df.iloc[i]['RAILS_HEAD_HARDENED'] +
                    level_2_df.iloc[i]['SLEEPER_BAR'] +
                    level_2_df.iloc[i]['REBAR'] +
                    level_2_df.iloc[i]['WIRE_ROD']
                )
                
                consistency_error = abs(level_1_long - level_2_long) / level_1_long * 100 if level_1_long > 0 else 0
                consistency_results[f'year_{year}_finished_long_error'] = consistency_error
        
        # Summary statistics
        all_errors = [v for k, v in consistency_results.items() if 'error' in k]
        if all_errors:
            consistency_results['summary'] = {
                'max_error_percent': max(all_errors),
                'avg_error_percent': np.mean(all_errors),
                'errors_under_1_percent': sum(1 for e in all_errors if e < 1.0),
                'total_checks': len(all_errors),
                'consistency_score': sum(1 for e in all_errors if e < 1.0) / len(all_errors) * 100
            }
        
        return consistency_results
    
    def validate_renewable_energy_assumptions(self, 
                                            renewable_data: pd.DataFrame,
                                            renewable_steel_demand: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate renewable energy steel demand assumptions and calculations.
        
        Args:
            renewable_data: Renewable energy capacity data
            renewable_steel_demand: Calculated renewable energy steel demand
            
        Returns:
            Dictionary of renewable energy validation results
        """
        self.logger.info("Validating renewable energy assumptions...")
        
        renewable_validation = {}
        
        # Government target validation (2030: 43% renewable electricity)
        if 'total_renewable_capacity' in renewable_data.columns:
            # Approximate 2030 target: ~62 GW total renewable capacity
            renewable_2030 = renewable_data[renewable_data['Year'] == 2030]['total_renewable_capacity']
            
            if not renewable_2030.empty:
                capacity_2030 = renewable_2030.iloc[0]
                government_target_2030 = 62000  # MW
                
                renewable_validation['capacity_2030_target'] = {
                    'forecast_capacity_mw': capacity_2030,
                    'government_target_mw': government_target_2030,
                    'variance_percent': abs(capacity_2030 - government_target_2030) / government_target_2030 * 100,
                    'target_achieved': abs(capacity_2030 - government_target_2030) / government_target_2030 < 0.20  # Within 20%
                }
        
        # Steel intensity validation
        if 'total_renewable_steel_demand' in renewable_steel_demand.columns:
            total_capacity_2030 = renewable_data[renewable_data['Year'] == 2030]['total_renewable_capacity'].iloc[0] if 'total_renewable_capacity' in renewable_data.columns else 0
            total_steel_2030 = renewable_steel_demand[renewable_steel_demand['Year'] == 2030]['total_renewable_steel_demand'].iloc[0]
            
            if total_capacity_2030 > 0:
                weighted_steel_intensity = total_steel_2030 / total_capacity_2030
                
                # International benchmark: ~150-200 tonnes/MW weighted average
                international_benchmark = 175  # tonnes/MW
                
                renewable_validation['steel_intensity_validation'] = {
                    'calculated_intensity': weighted_steel_intensity,
                    'international_benchmark': international_benchmark,
                    'variance_percent': abs(weighted_steel_intensity - international_benchmark) / international_benchmark * 100,
                    'benchmark_alignment': abs(weighted_steel_intensity - international_benchmark) / international_benchmark < 0.30  # Within 30%
                }
        
        # Distributed solar correction validation
        if 'Solar_Distributed' in renewable_data.columns and 'steel_demand_solar_distributed' in renewable_steel_demand.columns:
            solar_distributed_2030 = renewable_data[renewable_data['Year'] == 2030]['Solar_Distributed'].iloc[0]
            steel_distributed_2030 = renewable_steel_demand[renewable_steel_demand['Year'] == 2030]['steel_demand_solar_distributed'].iloc[0]
            
            if solar_distributed_2030 > 0:
                distributed_intensity = steel_distributed_2030 / solar_distributed_2030
                expected_intensity = 8  # tonnes/MW (corrected intensity)
                
                renewable_validation['distributed_solar_correction'] = {
                    'calculated_intensity': distributed_intensity,
                    'expected_intensity': expected_intensity,
                    'correction_applied': abs(distributed_intensity - expected_intensity) < 1.0,
                    'no_grid_infrastructure': steel_distributed_2030 < solar_distributed_2030 * 10  # Should be much less than with grid
                }
        
        return renewable_validation
    
    def validate_infrastructure_australia_alignment(self, 
                                                  level_0_forecasts: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate alignment with Infrastructure Australia pipeline and targets.
        
        Args:
            level_0_forecasts: Level 0 total steel demand forecasts
            
        Returns:
            Dictionary of Infrastructure Australia validation results
        """
        self.logger.info("Validating Infrastructure Australia alignment...")
        
        infrastructure_validation = {}
        
        # 5-year total steel demand (2025-2029)
        five_year_data = level_0_forecasts[
            (level_0_forecasts['Year'] >= 2025) & 
            (level_0_forecasts['Year'] <= 2029)
        ]
        
        if not five_year_data.empty:
            five_year_total = five_year_data['total_steel_demand'].sum()
            infrastructure_target = self.infrastructure_constraints['total_steel_5_years']
            
            infrastructure_validation['five_year_alignment'] = {
                'forecast_total_kt': five_year_total,
                'infrastructure_australia_target_kt': infrastructure_target,
                'variance_percent': abs(five_year_total - infrastructure_target) / infrastructure_target * 100,
                'target_alignment': abs(five_year_total - infrastructure_target) / infrastructure_target < 0.25  # Within 25%
            }
        
        # Annual average validation
        annual_average = level_0_forecasts['total_steel_demand'].mean()
        target_average = self.infrastructure_constraints['annual_average']
        
        infrastructure_validation['annual_average_alignment'] = {
            'forecast_average_kt': annual_average,
            'target_average_kt': target_average,
            'variance_percent': abs(annual_average - target_average) / target_average * 100,
            'target_alignment': abs(annual_average - target_average) / target_average < 0.20  # Within 20%
        }
        
        return infrastructure_validation
    
    def validate_economic_relationships(self, 
                                      forecasts: Dict[str, pd.DataFrame],
                                      macro_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate economic relationships and correlations in forecasts.
        
        Args:
            forecasts: Dictionary of forecasts for all levels
            macro_data: Macro economic data
            
        Returns:
            Dictionary of economic relationship validation results
        """
        self.logger.info("Validating economic relationships...")
        
        economic_validation = {}
        
        if 'level_0' not in forecasts:
            return economic_validation
        
        level_0_df = forecasts['level_0']
        
        # GDP correlation validation
        if 'GDP_AUD_Real2015' in macro_data.columns:
            # Calculate correlation between steel demand and GDP
            merged_data = pd.merge(level_0_df, macro_data[['Year', 'GDP_AUD_Real2015']], on='Year', how='inner')
            
            if len(merged_data) > 5:  # Minimum observations for correlation
                correlation = merged_data['total_steel_demand'].corr(merged_data['GDP_AUD_Real2015'])
                
                economic_validation['gdp_correlation'] = {
                    'correlation_coefficient': correlation,
                    'expected_range': (0.6, 0.9),  # Expected strong positive correlation
                    'correlation_valid': 0.6 <= correlation <= 0.9
                }
        
        # Population correlation validation
        if 'Population' in macro_data.columns:
            merged_data = pd.merge(level_0_df, macro_data[['Year', 'Population']], on='Year', how='inner')
            
            if len(merged_data) > 5:
                correlation = merged_data['total_steel_demand'].corr(merged_data['Population'])
                
                economic_validation['population_correlation'] = {
                    'correlation_coefficient': correlation,
                    'expected_range': (0.4, 0.8),  # Expected moderate positive correlation
                    'correlation_valid': 0.4 <= correlation <= 0.8
                }
        
        # Steel intensity per capita validation
        if 'Population' in macro_data.columns:
            merged_data = pd.merge(level_0_df, macro_data[['Year', 'Population']], on='Year', how='inner')
            merged_data['steel_per_capita'] = merged_data['total_steel_demand'] / merged_data['Population']
            
            # International benchmark: Australia ~300-500 kg per capita
            avg_per_capita = merged_data['steel_per_capita'].mean()
            
            economic_validation['per_capita_intensity'] = {
                'avg_kg_per_capita': avg_per_capita,
                'international_benchmark_range': (300, 500),
                'benchmark_alignment': 250 <= avg_per_capita <= 600  # Reasonable range for developed economy
            }
        
        return economic_validation
    
    def generate_comprehensive_validation_report(self,
                                                forecasts: Dict[str, pd.DataFrame],
                                                current_forecasts: pd.DataFrame = None,
                                                macro_data: pd.DataFrame = None,
                                                renewable_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for all aspects of the forecasting system.
        
        Args:
            forecasts: Hierarchical forecasts dictionary
            current_forecasts: Current model forecasts
            macro_data: Macro economic data
            renewable_data: Renewable energy data
            
        Returns:
            Comprehensive validation report
        """
        self.logger.info("Generating comprehensive validation report...")
        
        validation_report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'forecast_period': f"{forecasts['level_0']['Year'].min()}-{forecasts['level_0']['Year'].max()}",
            'validation_summary': {}
        }
        
        # 1. Hierarchical consistency validation
        consistency_results = self.validate_hierarchical_consistency(forecasts)
        validation_report['hierarchical_consistency'] = consistency_results
        
        # 2. Infrastructure Australia alignment
        infrastructure_results = self.validate_infrastructure_australia_alignment(forecasts['level_0'])
        validation_report['infrastructure_alignment'] = infrastructure_results
        
        # 3. Economic relationships validation
        if macro_data is not None:
            economic_results = self.validate_economic_relationships(forecasts, macro_data)
            validation_report['economic_relationships'] = economic_results
        
        # 4. Renewable energy validation
        if renewable_data is not None:
            renewable_results = self.validate_renewable_energy_assumptions(
                renewable_data, 
                renewable_data if 'total_renewable_steel_demand' in renewable_data.columns else forecasts.get('renewable_features', pd.DataFrame())
            )
            validation_report['renewable_energy'] = renewable_results
        
        # 5. Cross-model validation (if current forecasts available)
        if current_forecasts is not None:
            cross_validation = self._validate_current_vs_hierarchical(current_forecasts, forecasts)
            validation_report['cross_model_validation'] = cross_validation
        
        # 6. Generate validation summary
        validation_summary = self._generate_validation_summary(validation_report)
        validation_report['validation_summary'] = validation_summary
        
        self.logger.info(f"Validation report generated: {validation_summary['overall_score']:.1f}% overall score")
        
        return validation_report
    
    def _validate_current_vs_hierarchical(self,
                                        current_forecasts: pd.DataFrame,
                                        hierarchical_forecasts: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Validate current model forecasts against hierarchical model forecasts.
        
        Args:
            current_forecasts: Current model forecasts
            hierarchical_forecasts: Hierarchical model forecasts
            
        Returns:
            Cross-validation results
        """
        cross_validation = {}
        
        # Production vs consumption validation
        if 'Total Production of Crude Steel' in current_forecasts.columns:
            production = current_forecasts['Total Production of Crude Steel']
            consumption = hierarchical_forecasts['level_0']['total_steel_demand']
            
            # Align data
            merged_data = pd.merge(
                current_forecasts[['Year', 'Total Production of Crude Steel']],
                hierarchical_forecasts['level_0'][['Year', 'total_steel_demand']],
                on='Year', how='inner'
            )
            
            if not merged_data.empty:
                ratio = merged_data['Total Production of Crude Steel'] / merged_data['total_steel_demand']
                
                cross_validation['production_consumption'] = {
                    'mean_ratio': ratio.mean(),
                    'ratio_stability': ratio.std(),
                    'expected_range': (0.80, 0.95),
                    'ratio_valid': (ratio >= 0.75).all() and (ratio <= 1.0).all()
                }
        
        return cross_validation
    
    def _generate_validation_summary(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate overall validation summary with scores and recommendations.
        
        Args:
            validation_report: Complete validation report
            
        Returns:
            Validation summary with scores
        """
        summary = {
            'overall_score': 0.0,
            'category_scores': {},
            'critical_issues': [],
            'recommendations': [],
            'validation_status': 'UNKNOWN'
        }
        
        # Score hierarchical consistency (25% weight)
        consistency = validation_report.get('hierarchical_consistency', {})
        if 'summary' in consistency:
            consistency_score = consistency['summary'].get('consistency_score', 0)
            summary['category_scores']['hierarchical_consistency'] = consistency_score
        else:
            summary['category_scores']['hierarchical_consistency'] = 50  # Default if missing
        
        # Score infrastructure alignment (25% weight)  
        infrastructure = validation_report.get('infrastructure_alignment', {})
        infra_score = 0
        if 'five_year_alignment' in infrastructure:
            if infrastructure['five_year_alignment'].get('target_alignment', False):
                infra_score += 50
        if 'annual_average_alignment' in infrastructure:
            if infrastructure['annual_average_alignment'].get('target_alignment', False):
                infra_score += 50
        summary['category_scores']['infrastructure_alignment'] = infra_score
        
        # Score economic relationships (25% weight)
        economic = validation_report.get('economic_relationships', {})
        econ_score = 0
        checks = 0
        for validation in economic.values():
            if isinstance(validation, dict) and 'correlation_valid' in validation:
                if validation['correlation_valid']:
                    econ_score += 33.33
                checks += 1
        if checks == 0:
            econ_score = 50  # Default if no checks
        summary['category_scores']['economic_relationships'] = min(econ_score, 100)
        
        # Score renewable energy validation (25% weight)
        renewable = validation_report.get('renewable_energy', {})
        renewable_score = 0
        renewable_checks = 0
        for validation in renewable.values():
            if isinstance(validation, dict):
                if validation.get('target_achieved', False) or validation.get('benchmark_alignment', False) or validation.get('correction_applied', False):
                    renewable_score += 33.33
                renewable_checks += 1
        if renewable_checks == 0:
            renewable_score = 50  # Default if no checks
        summary['category_scores']['renewable_energy'] = min(renewable_score, 100)
        
        # Calculate overall score (weighted average)
        weights = {
            'hierarchical_consistency': 0.25,
            'infrastructure_alignment': 0.25,
            'economic_relationships': 0.25,
            'renewable_energy': 0.25
        }
        
        overall_score = sum(
            summary['category_scores'][category] * weight
            for category, weight in weights.items()
        )
        summary['overall_score'] = overall_score
        
        # Determine validation status
        if overall_score >= 80:
            summary['validation_status'] = 'EXCELLENT'
        elif overall_score >= 70:
            summary['validation_status'] = 'GOOD'
        elif overall_score >= 60:
            summary['validation_status'] = 'ACCEPTABLE'
        else:
            summary['validation_status'] = 'NEEDS_IMPROVEMENT'
        
        # Generate recommendations
        if summary['category_scores']['hierarchical_consistency'] < 70:
            summary['recommendations'].append("Review hierarchical disaggregation logic for better consistency")
        
        if summary['category_scores']['infrastructure_alignment'] < 70:
            summary['recommendations'].append("Adjust sectoral weights to better align with Infrastructure Australia targets")
        
        if summary['category_scores']['economic_relationships'] < 70:
            summary['recommendations'].append("Review economic driver correlations and feature engineering")
        
        if summary['category_scores']['renewable_energy'] < 70:
            summary['recommendations'].append("Validate renewable energy steel intensity assumptions against industry data")
        
        return summary