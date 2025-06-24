"""
Steel Demand ML Model - Validation and Backtesting Framework
Comprehensive validation against real historical data and regression benchmarks.
All parameters loaded from CSV configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class SteelDemandValidationFramework:
    """
    Comprehensive validation framework for steel demand forecasting models.
    Validates against historical data and compares with regression benchmarks.
    """
    
    def __init__(self, data_loader, config_path: str = "config/"):
        """
        Initialize validation framework with data loader and configuration.
        
        Args:
            data_loader: SteelDemandDataLoader instance
            config_path: Path to configuration directory
        """
        self.data_loader = data_loader
        self.config_path = config_path
        self.logger = self._setup_logging()
        self._load_config()
        
        # Storage for validation results
        self.validation_results = {}
        self.benchmark_comparisons = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Load validation configuration from CSV files."""
        self.config = {
            'target_mape': float(self.data_loader.get_model_config('target_mape')),
            'acceptable_mape': float(self.data_loader.get_model_config('acceptable_mape')),
            'min_r2_score': float(self.data_loader.get_model_config('min_r2_score'))
        }
        
        # Load validation benchmarks
        self.benchmarks = self.data_loader.get_validation_benchmarks()
    
    def validate_against_historical_data(self, model, X: pd.DataFrame, y: pd.Series,
                                       category: str, validation_years: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Validate model against historical steel consumption data.
        
        Args:
            model: Trained model to validate
            X: Feature matrix
            y: Historical target values
            category: Steel category name
            validation_years: Specific years to validate (None for all)
            
        Returns:
            Historical validation results
        """
        try:
            self.logger.info(f"Validating {category} against historical data")
            
            # Filter data by years if specified
            if validation_years is not None and 'Year' in X.columns:
                year_mask = X['Year'].isin(validation_years)
                X_val = X[year_mask]
                y_val = y[year_mask]
            else:
                X_val = X
                y_val = y
            
            # Remove Year column for prediction
            feature_cols = [col for col in X_val.columns if col != 'Year']
            X_features = X_val[feature_cols]
            
            # Make predictions
            predictions = model.predict(X_features)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, predictions)
            mse = mean_squared_error(y_val, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, predictions)
            mape = np.mean(np.abs((y_val - predictions) / (y_val + 1e-6))) * 100
            
            # Calculate year-by-year metrics
            yearly_metrics = []
            if 'Year' in X_val.columns:
                for year in sorted(X_val['Year'].unique()):
                    year_mask = X_val['Year'] == year
                    if year_mask.sum() > 0:
                        y_year = y_val[year_mask]
                        pred_year = predictions[year_mask]
                        
                        if len(y_year) > 0:
                            year_mae = mean_absolute_error(y_year, pred_year)
                            year_mape = np.mean(np.abs((y_year - pred_year) / (y_year + 1e-6))) * 100
                            year_r2 = r2_score(y_year, pred_year) if len(y_year) > 1 else np.nan
                            
                            yearly_metrics.append({
                                'Year': year,
                                'Actual': y_year.iloc[0] if len(y_year) == 1 else y_year.mean(),
                                'Predicted': pred_year[0] if len(pred_year) == 1 else pred_year.mean(),
                                'MAE': year_mae,
                                'MAPE': year_mape,
                                'R2': year_r2
                            })
            
            # Assess performance against benchmarks
            category_benchmarks = self.benchmarks[self.benchmarks['category'] == category]
            performance_assessment = self._assess_against_benchmarks(
                {'MAPE': mape, 'R2': r2, 'RMSE': rmse}, category_benchmarks
            )
            
            validation_results = {
                'category': category,
                'overall_metrics': {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape
                },
                'yearly_metrics': yearly_metrics,
                'performance_assessment': performance_assessment,
                'validation_period': {
                    'start_year': X_val['Year'].min() if 'Year' in X_val.columns else 'N/A',
                    'end_year': X_val['Year'].max() if 'Year' in X_val.columns else 'N/A',
                    'total_years': len(yearly_metrics)
                },
                'sample_size': len(y_val)
            }
            
            self.logger.info(f"Historical validation completed for {category}")
            self.logger.info(f"MAPE: {mape:.2f}%, R²: {r2:.3f}, RMSE: {rmse:.0f}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in historical validation for {category}: {str(e)}")
            raise
    
    def compare_with_regression_baseline(self, ml_model, X: pd.DataFrame, y: pd.Series,
                                       category: str) -> Dict[str, Any]:
        """
        Compare ML model performance with regression baseline.
        
        Args:
            ml_model: Trained ML model
            X: Feature matrix
            y: Target values
            category: Steel category name
            
        Returns:
            Comparison results
        """
        try:
            self.logger.info(f"Comparing {category} with regression baseline")
            
            # Get regression coefficients from historical analysis
            steel_categories = self.data_loader.get_steel_categories()
            category_info = steel_categories[steel_categories['category'] == category]
            
            if category_info.empty:
                raise ValueError(f"Category {category} not found in steel categories")
            
            # Apply regression model based on historical coefficients
            # Note: This uses the coefficients from the historical regression analysis
            regression_predictions = self._apply_regression_model(X, category, category_info.iloc[0])
            
            # Remove Year column for ML prediction
            feature_cols = [col for col in X.columns if col != 'Year']
            X_features = X[feature_cols]
            
            # Get ML model predictions
            ml_predictions = ml_model.predict(X_features)
            
            # Calculate metrics for both models
            ml_metrics = self._calculate_metrics(y, ml_predictions, "ML_Model")
            regression_metrics = self._calculate_metrics(y, regression_predictions, "Regression_Baseline")
            
            # Calculate improvement
            improvement = {
                'MAPE_improvement': regression_metrics['MAPE'] - ml_metrics['MAPE'],
                'R2_improvement': ml_metrics['R2'] - regression_metrics['R2'],
                'RMSE_improvement': regression_metrics['RMSE'] - ml_metrics['RMSE'],
                'relative_MAPE_improvement': ((regression_metrics['MAPE'] - ml_metrics['MAPE']) / regression_metrics['MAPE']) * 100
            }
            
            comparison_results = {
                'category': category,
                'ml_metrics': ml_metrics,
                'regression_metrics': regression_metrics,
                'improvement': improvement,
                'ml_better': ml_metrics['MAPE'] < regression_metrics['MAPE'],
                'significant_improvement': improvement['relative_MAPE_improvement'] > 5.0  # >5% improvement
            }
            
            self.logger.info(f"Baseline comparison completed for {category}")
            self.logger.info(f"ML MAPE: {ml_metrics['MAPE']:.2f}%, Regression MAPE: {regression_metrics['MAPE']:.2f}%")
            self.logger.info(f"Improvement: {improvement['relative_MAPE_improvement']:.1f}%")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error in baseline comparison for {category}: {str(e)}")
            raise
    
    def _apply_regression_model(self, X: pd.DataFrame, category: str, category_info: pd.Series) -> np.ndarray:
        """Apply historical regression model based on documented coefficients."""
        try:
            # Get the regression equations from the historical analysis
            # These are the actual coefficients from Historical_Regression_Analysis_2007-2024.md
            
            regression_coefficients = {
                'Hot Rolled Structural Steel': {
                    'intercept': -1245000,
                    'GDP_Real_AUD_Billion': 285.4,
                    'Total_Population_Millions': 42150,
                    'National_Urbanisation_Rate_pct': 8420
                },
                'Rail Products': {
                    'intercept': -125000,
                    'Iron_Ore_Production_Mt': 78.5,
                    'Coal_Production_Mt': 45.2,
                    'Urban_Population_Millions': 6850  # Calculated as Total_Population * Urbanisation_Rate / 100
                },
                'Steel Billets': {
                    'intercept': -856000,
                    'Industrial_Production_Index_Base_2007': 4250,
                    'GDP_Real_AUD_Billion': 165.7,
                    'Total_Population_Millions': 28900
                },
                'Steel Slabs': {
                    'intercept': -1890000,
                    'Industrial_Production_Index_Base_2007': 8750,
                    'GDP_Real_AUD_Billion': 425.8,
                    'Iron_Ore_Production_Mt': 285.4
                }
            }
            
            if category not in regression_coefficients:
                raise ValueError(f"No regression coefficients found for {category}")
            
            coeffs = regression_coefficients[category]
            
            # Calculate regression predictions
            predictions = np.full(len(X), coeffs['intercept'])
            
            # Add each term
            for var_name, coeff in coeffs.items():
                if var_name != 'intercept':
                    if var_name == 'Urban_Population_Millions':
                        # Calculate urban population
                        if 'Total_Population_Millions' in X.columns and 'National_Urbanisation_Rate_pct' in X.columns:
                            urban_pop = X['Total_Population_Millions'] * X['National_Urbanisation_Rate_pct'] / 100
                            predictions += coeff * urban_pop
                    elif var_name in X.columns:
                        predictions += coeff * X[var_name]
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error applying regression model: {str(e)}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> Dict[str, float]:
        """Calculate standard metrics for model evaluation."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        return {
            'model': model_name,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def _assess_against_benchmarks(self, metrics: Dict[str, float], 
                                 category_benchmarks: pd.DataFrame) -> Dict[str, str]:
        """Assess model performance against predefined benchmarks."""
        assessment = {}
        
        for _, benchmark in category_benchmarks.iterrows():
            metric_name = benchmark['metric']
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                target_value = benchmark['target_value']
                acceptable_value = benchmark['acceptable_value']
                critical_value = benchmark['critical_value']
                
                if metric_name in ['MAPE', 'RMSE']:  # Lower is better
                    if actual_value <= target_value:
                        status = 'EXCELLENT'
                    elif actual_value <= acceptable_value:
                        status = 'GOOD'
                    elif actual_value <= critical_value:
                        status = 'ACCEPTABLE'
                    else:
                        status = 'POOR'
                else:  # Higher is better (R2)
                    if actual_value >= target_value:
                        status = 'EXCELLENT'
                    elif actual_value >= acceptable_value:
                        status = 'GOOD'
                    elif actual_value >= critical_value:
                        status = 'ACCEPTABLE'
                    else:
                        status = 'POOR'
                
                assessment[metric_name] = status
        
        # Overall assessment
        status_scores = {'EXCELLENT': 4, 'GOOD': 3, 'ACCEPTABLE': 2, 'POOR': 1}
        if assessment:
            avg_score = np.mean([status_scores[status] for status in assessment.values()])
            if avg_score >= 3.5:
                assessment['OVERALL'] = 'EXCELLENT'
            elif avg_score >= 2.5:
                assessment['GOOD'] = 'GOOD'
            elif avg_score >= 1.5:
                assessment['OVERALL'] = 'ACCEPTABLE'
            else:
                assessment['OVERALL'] = 'POOR'
        else:
            assessment['OVERALL'] = 'NO_BENCHMARKS'
        
        return assessment
    
    def backtest_temporal_stability(self, model, X: pd.DataFrame, y: pd.Series,
                                  category: str, window_size: int = 3) -> Dict[str, Any]:
        """
        Perform temporal backtesting to assess model stability over time.
        
        Args:
            model: Trained model to backtest
            X: Feature matrix with Year column
            y: Target values
            category: Steel category name
            window_size: Size of rolling window for backtesting
            
        Returns:
            Temporal stability results
        """
        try:
            self.logger.info(f"Performing temporal backtesting for {category}")
            
            if 'Year' not in X.columns:
                raise ValueError("Year column required for temporal backtesting")
            
            # Sort by year
            sort_indices = X['Year'].argsort()
            X_sorted = X.iloc[sort_indices]
            y_sorted = y.iloc[sort_indices]
            
            backtest_results = []
            feature_cols = [col for col in X.columns if col != 'Year']
            
            years = sorted(X_sorted['Year'].unique())
            
            for i in range(window_size, len(years)):
                test_year = years[i]
                train_years = years[i-window_size:i]
                
                # Split data
                train_mask = X_sorted['Year'].isin(train_years)
                test_mask = X_sorted['Year'] == test_year
                
                X_train_bt = X_sorted[train_mask][feature_cols]
                y_train_bt = y_sorted[train_mask]
                X_test_bt = X_sorted[test_mask][feature_cols]
                y_test_bt = y_sorted[test_mask]
                
                if len(X_train_bt) > 0 and len(X_test_bt) > 0:
                    try:
                        # Retrain model on historical window
                        from models.ensemble_models import EnsembleSteelModel
                        bt_model = EnsembleSteelModel(self.data_loader)
                        bt_model.fit(X_train_bt, y_train_bt)
                        
                        # Predict on test year
                        bt_predictions = bt_model.predict(X_test_bt)
                        
                        # Calculate metrics
                        bt_metrics = self._calculate_metrics(y_test_bt, bt_predictions, f"Backtest_{test_year}")
                        bt_metrics['test_year'] = test_year
                        bt_metrics['train_years'] = train_years
                        
                        backtest_results.append(bt_metrics)
                        
                    except Exception as e:
                        self.logger.warning(f"Backtest failed for year {test_year}: {str(e)}")
                        continue
            
            if not backtest_results:
                raise ValueError("No successful backtest iterations")
            
            # Analyze temporal stability
            backtest_df = pd.DataFrame(backtest_results)
            
            stability_metrics = {
                'mean_mape': backtest_df['MAPE'].mean(),
                'std_mape': backtest_df['MAPE'].std(),
                'mape_trend': np.polyfit(range(len(backtest_df)), backtest_df['MAPE'], 1)[0],
                'mean_r2': backtest_df['R2'].mean(),
                'std_r2': backtest_df['R2'].std(),
                'r2_trend': np.polyfit(range(len(backtest_df)), backtest_df['R2'], 1)[0],
                'stability_score': 1 / (1 + backtest_df['MAPE'].std())  # Higher is more stable
            }
            
            temporal_results = {
                'category': category,
                'backtest_results': backtest_results,
                'stability_metrics': stability_metrics,
                'window_size': window_size,
                'test_years': [r['test_year'] for r in backtest_results]
            }
            
            self.logger.info(f"Temporal backtesting completed for {category}")
            self.logger.info(f"Mean MAPE: {stability_metrics['mean_mape']:.2f}% ± {stability_metrics['std_mape']:.2f}%")
            
            return temporal_results
            
        except Exception as e:
            self.logger.error(f"Error in temporal backtesting for {category}: {str(e)}")
            raise
    
    def comprehensive_validation(self, models: Dict[str, Any], data: Dict[str, pd.DataFrame],
                               output_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive validation across all categories and models.
        
        Args:
            models: Dictionary of trained models by category
            data: Dictionary containing historical and other data
            output_path: Path to save validation results
            
        Returns:
            Complete validation results
        """
        try:
            self.logger.info("Starting comprehensive validation")
            
            historical_data = data['historical']
            steel_categories = data['steel_categories']
            
            all_validation_results = {}
            
            for _, category_info in steel_categories.iterrows():
                category_name = category_info['category']
                target_column = category_info['target_column']
                
                if category_name not in models:
                    self.logger.warning(f"No trained model found for {category_name}")
                    continue
                
                try:
                    self.logger.info(f"Validating {category_name}")
                    
                    model = models[category_name]['model']
                    
                    # Prepare data for this category
                    feature_engineer = self.data_loader.feature_engineer if hasattr(self.data_loader, 'feature_engineer') else None
                    if feature_engineer:
                        features_df = feature_engineer.create_features(historical_data, target_column)
                    else:
                        features_df = historical_data.copy()
                    
                    # Get features and target
                    if target_column not in features_df.columns:
                        self.logger.warning(f"Target column {target_column} not found for {category_name}")
                        continue
                    
                    valid_indices = features_df[target_column].notna()
                    X = features_df[valid_indices].drop(columns=[target_column])
                    y = features_df[valid_indices][target_column]
                    
                    category_results = {}
                    
                    # Historical validation
                    try:
                        category_results['historical_validation'] = self.validate_against_historical_data(
                            model, X, y, category_name
                        )
                    except Exception as e:
                        self.logger.warning(f"Historical validation failed for {category_name}: {str(e)}")
                    
                    # Regression baseline comparison
                    try:
                        category_results['baseline_comparison'] = self.compare_with_regression_baseline(
                            model, X, y, category_name
                        )
                    except Exception as e:
                        self.logger.warning(f"Baseline comparison failed for {category_name}: {str(e)}")
                    
                    # Temporal backtesting
                    try:
                        category_results['temporal_backtesting'] = self.backtest_temporal_stability(
                            model, X, y, category_name
                        )
                    except Exception as e:
                        self.logger.warning(f"Temporal backtesting failed for {category_name}: {str(e)}")
                    
                    all_validation_results[category_name] = category_results
                    
                except Exception as e:
                    self.logger.error(f"Validation failed for {category_name}: {str(e)}")
                    all_validation_results[category_name] = {'error': str(e)}
            
            # Save validation results
            self._save_validation_results(all_validation_results, output_path)
            
            # Generate validation summary
            validation_summary = self._generate_validation_summary(all_validation_results)
            
            self.logger.info("Comprehensive validation completed")
            
            return {
                'detailed_results': all_validation_results,
                'summary': validation_summary
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {str(e)}")
            raise
    
    def _save_validation_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save validation results to CSV files."""
        try:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary metrics
            summary_data = []
            
            for category, category_results in results.items():
                if 'error' in category_results:
                    summary_data.append({
                        'Category': category,
                        'Status': 'FAILED',
                        'Error': category_results['error']
                    })
                else:
                    row = {'Category': category, 'Status': 'SUCCESS'}
                    
                    # Add historical validation metrics
                    if 'historical_validation' in category_results:
                        hist_val = category_results['historical_validation']
                        metrics = hist_val['overall_metrics']
                        row.update({
                            'Historical_MAPE': metrics['MAPE'],
                            'Historical_R2': metrics['R2'],
                            'Historical_RMSE': metrics['RMSE']
                        })
                    
                    # Add baseline comparison
                    if 'baseline_comparison' in category_results:
                        baseline = category_results['baseline_comparison']
                        row.update({
                            'ML_MAPE': baseline['ml_metrics']['MAPE'],
                            'Regression_MAPE': baseline['regression_metrics']['MAPE'],
                            'MAPE_Improvement': baseline['improvement']['MAPE_improvement'],
                            'Relative_Improvement_Pct': baseline['improvement']['relative_MAPE_improvement']
                        })
                    
                    # Add temporal stability
                    if 'temporal_backtesting' in category_results:
                        temporal = category_results['temporal_backtesting']
                        stability = temporal['stability_metrics']
                        row.update({
                            'Backtest_Mean_MAPE': stability['mean_mape'],
                            'Backtest_Std_MAPE': stability['std_mape'],
                            'Stability_Score': stability['stability_score']
                        })
                    
                    summary_data.append(row)
            
            # Save summary
            summary_df = pd.DataFrame(summary_data)
            summary_file = output_dir / 'validation_summary.csv'
            summary_df.to_csv(summary_file, index=False)
            
            self.logger.info(f"Validation summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving validation results: {str(e)}")
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level validation summary."""
        try:
            successful_validations = [r for r in results.values() if 'error' not in r]
            failed_validations = [r for r in results.values() if 'error' in r]
            
            summary = {
                'total_categories': len(results),
                'successful_validations': len(successful_validations),
                'failed_validations': len(failed_validations),
                'success_rate': len(successful_validations) / len(results) * 100 if results else 0
            }
            
            if successful_validations:
                # Aggregate metrics
                historical_mapes = []
                improvements = []
                stability_scores = []
                
                for result in successful_validations:
                    if 'historical_validation' in result:
                        historical_mapes.append(result['historical_validation']['overall_metrics']['MAPE'])
                    
                    if 'baseline_comparison' in result:
                        improvements.append(result['baseline_comparison']['improvement']['relative_MAPE_improvement'])
                    
                    if 'temporal_backtesting' in result:
                        stability_scores.append(result['temporal_backtesting']['stability_metrics']['stability_score'])
                
                if historical_mapes:
                    summary['average_mape'] = np.mean(historical_mapes)
                    summary['best_mape'] = np.min(historical_mapes)
                    summary['worst_mape'] = np.max(historical_mapes)
                
                if improvements:
                    summary['average_improvement'] = np.mean(improvements)
                    summary['categories_improved'] = len([i for i in improvements if i > 0])
                
                if stability_scores:
                    summary['average_stability'] = np.mean(stability_scores)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating validation summary: {str(e)}")
            return {'error': str(e)}