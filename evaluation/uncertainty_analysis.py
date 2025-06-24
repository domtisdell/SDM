"""
Steel Demand ML Model - Uncertainty Quantification and Interpretability
Provides uncertainty estimates and SHAP-based model interpretability.
All parameters loaded from CSV configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for model interpretability
import shap

# Bootstrap and Monte Carlo
from sklearn.utils import resample

warnings.filterwarnings('ignore')

class SteelDemandUncertaintyAnalysis:
    """
    Uncertainty quantification and interpretability analysis for steel demand models.
    Provides confidence intervals, feature importance, and model explanations.
    """
    
    def __init__(self, data_loader, config_path: str = "config/"):
        """
        Initialize uncertainty analysis with data loader and configuration.
        
        Args:
            data_loader: SteelDemandDataLoader instance
            config_path: Path to configuration directory
        """
        self.data_loader = data_loader
        self.config_path = config_path
        self.logger = self._setup_logging()
        self._load_config()
        
        # Initialize SHAP explainers storage
        self.explainers = {}
        self.shap_values = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Load uncertainty analysis configuration from CSV files."""
        self.config = {
            'confidence_intervals': self.data_loader.get_model_config('confidence_intervals'),
            'bootstrap_samples': int(self.data_loader.get_model_config('bootstrap_samples')),
            'monte_carlo_samples': int(self.data_loader.get_model_config('monte_carlo_samples')),
            'random_state': int(self.data_loader.get_model_config('random_state'))
        }
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_state'])
    
    def bootstrap_uncertainty(self, model, X: pd.DataFrame, y: pd.Series,
                            n_bootstrap: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate uncertainty estimates using bootstrap resampling.
        
        Args:
            model: Trained ensemble model
            X: Feature matrix
            y: Target values
            n_bootstrap: Number of bootstrap samples (uses config if None)
            
        Returns:
            Bootstrap uncertainty results
        """
        try:
            if n_bootstrap is None:
                n_bootstrap = self.config['bootstrap_samples']
            
            self.logger.info(f"Computing bootstrap uncertainty with {n_bootstrap} samples")
            
            # Remove Year column for model prediction
            feature_cols = [col for col in X.columns if col != 'Year']
            X_features = X[feature_cols]
            
            bootstrap_predictions = []
            bootstrap_metrics = []
            
            for i in range(n_bootstrap):
                if i % 100 == 0:
                    self.logger.info(f"Bootstrap iteration {i}/{n_bootstrap}")
                
                # Resample with replacement
                X_boot, y_boot = resample(X_features, y, random_state=i)
                
                # Retrain model on bootstrap sample
                try:
                    # Create new model instance for bootstrap
                    from models.ensemble_models import EnsembleSteelModel
                    boot_model = EnsembleSteelModel(self.data_loader)
                    boot_model.fit(X_boot, y_boot)
                    
                    # Make predictions on original test set
                    boot_pred = boot_model.predict(X_features)
                    bootstrap_predictions.append(boot_pred)
                    
                    # Calculate metrics
                    mae = np.mean(np.abs(y - boot_pred))
                    mape = np.mean(np.abs((y - boot_pred) / (y + 1e-6))) * 100
                    bootstrap_metrics.append({'MAE': mae, 'MAPE': mape})
                    
                except Exception as e:
                    self.logger.warning(f"Bootstrap iteration {i} failed: {str(e)}")
                    continue
            
            if not bootstrap_predictions:
                raise ValueError("All bootstrap iterations failed")
            
            # Convert to numpy array
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Calculate uncertainty statistics
            confidence_levels = self.config['confidence_intervals']
            
            uncertainty_results = {
                'mean_prediction': np.mean(bootstrap_predictions, axis=0),
                'std_prediction': np.std(bootstrap_predictions, axis=0),
                'confidence_intervals': {},
                'bootstrap_metrics': bootstrap_metrics,
                'n_successful_bootstrap': len(bootstrap_predictions)
            }
            
            # Calculate confidence intervals
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
                upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
                
                uncertainty_results['confidence_intervals'][f'{conf_level:.0%}'] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            
            # Calculate prediction intervals
            prediction_std = np.std(bootstrap_predictions, axis=0)
            uncertainty_results['prediction_intervals'] = {
                'std': prediction_std,
                'lower_95': uncertainty_results['mean_prediction'] - 1.96 * prediction_std,
                'upper_95': uncertainty_results['mean_prediction'] + 1.96 * prediction_std
            }
            
            self.logger.info("Bootstrap uncertainty calculation completed")
            
            return uncertainty_results
            
        except Exception as e:
            self.logger.error(f"Error in bootstrap uncertainty: {str(e)}")
            raise
    
    def monte_carlo_uncertainty(self, model, X: pd.DataFrame,
                              noise_std: float = 0.1,
                              n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate uncertainty using Monte Carlo sampling.
        
        Args:
            model: Trained ensemble model
            X: Feature matrix
            noise_std: Standard deviation of noise to add to features
            n_samples: Number of Monte Carlo samples (uses config if None)
            
        Returns:
            Monte Carlo uncertainty results
        """
        try:
            if n_samples is None:
                n_samples = self.config['monte_carlo_samples']
            
            self.logger.info(f"Computing Monte Carlo uncertainty with {n_samples} samples")
            
            # Remove Year column for model prediction
            feature_cols = [col for col in X.columns if col != 'Year']
            X_features = X[feature_cols]
            
            mc_predictions = []
            
            for i in range(n_samples):
                if i % 1000 == 0:
                    self.logger.info(f"Monte Carlo iteration {i}/{n_samples}")
                
                # Add noise to features
                noise = np.random.normal(0, noise_std, X_features.shape)
                X_noisy = X_features + noise
                
                # Make prediction
                try:
                    mc_pred = model.predict(X_noisy)
                    mc_predictions.append(mc_pred)
                except Exception as e:
                    self.logger.warning(f"Monte Carlo iteration {i} failed: {str(e)}")
                    continue
            
            if not mc_predictions:
                raise ValueError("All Monte Carlo iterations failed")
            
            # Convert to numpy array
            mc_predictions = np.array(mc_predictions)
            
            # Calculate uncertainty statistics
            confidence_levels = self.config['confidence_intervals']
            
            mc_results = {
                'mean_prediction': np.mean(mc_predictions, axis=0),
                'std_prediction': np.std(mc_predictions, axis=0),
                'confidence_intervals': {},
                'n_successful_samples': len(mc_predictions)
            }
            
            # Calculate confidence intervals
            for conf_level in confidence_levels:
                alpha = 1 - conf_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bound = np.percentile(mc_predictions, lower_percentile, axis=0)
                upper_bound = np.percentile(mc_predictions, upper_percentile, axis=0)
                
                mc_results['confidence_intervals'][f'{conf_level:.0%}'] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            
            self.logger.info("Monte Carlo uncertainty calculation completed")
            
            return mc_results
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo uncertainty: {str(e)}")
            raise
    
    def calculate_shap_values(self, model, X: pd.DataFrame, 
                            category: str = "steel_category") -> Dict[str, Any]:
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            model: Trained ensemble model
            X: Feature matrix
            category: Steel category name
            
        Returns:
            SHAP analysis results
        """
        try:
            self.logger.info(f"Calculating SHAP values for {category}")
            
            # Remove Year column for SHAP analysis
            feature_cols = [col for col in X.columns if col != 'Year']
            X_features = X[feature_cols]
            
            shap_results = {}
            
            # Get individual models from ensemble
            individual_models = model.models
            
            for model_name, individual_model in individual_models.items():
                try:
                    self.logger.info(f"Computing SHAP for {model_name}")
                    
                    if model_name in ['xgboost', 'random_forest']:
                        # Tree-based models
                        if hasattr(individual_model, 'model'):
                            explainer = shap.TreeExplainer(individual_model.model)
                            shap_values = explainer.shap_values(X_features)
                        else:
                            continue
                    
                    elif model_name == 'lstm':
                        # Deep learning model - use permutation explainer
                        def model_predict(x):
                            return individual_model.predict(pd.DataFrame(x, columns=X_features.columns))
                        
                        # Use sample for explanation (LSTM can be slow)
                        sample_size = min(100, len(X_features))
                        X_sample = X_features.sample(n=sample_size, random_state=self.config['random_state'])
                        
                        explainer = shap.Explainer(model_predict, X_sample)
                        shap_values = explainer(X_features[:sample_size])
                        shap_values = shap_values.values
                    
                    else:
                        # Other models - use permutation explainer
                        def model_predict(x):
                            return individual_model.predict(pd.DataFrame(x, columns=X_features.columns))
                        
                        # Use smaller sample for faster computation
                        sample_size = min(50, len(X_features))
                        X_sample = X_features.sample(n=sample_size, random_state=self.config['random_state'])
                        
                        explainer = shap.Explainer(model_predict, X_sample)
                        shap_values = explainer(X_features[:sample_size])
                        shap_values = shap_values.values
                    
                    # Store results
                    shap_results[model_name] = {
                        'shap_values': shap_values,
                        'feature_names': X_features.columns.tolist(),
                        'explainer': explainer
                    }
                    
                    # Calculate feature importance from SHAP values
                    if len(shap_values.shape) == 2:
                        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                    else:
                        mean_abs_shap = np.mean(np.abs(shap_values))
                    
                    feature_importance = pd.DataFrame({
                        'feature': X_features.columns,
                        'shap_importance': mean_abs_shap
                    }).sort_values('shap_importance', ascending=False)
                    
                    shap_results[model_name]['feature_importance'] = feature_importance
                    
                    self.logger.info(f"SHAP calculation completed for {model_name}")
                    
                except Exception as e:
                    self.logger.warning(f"SHAP calculation failed for {model_name}: {str(e)}")
                    continue
            
            # Store SHAP results
            self.shap_values[category] = shap_results
            
            self.logger.info(f"SHAP analysis completed for {category}")
            
            return shap_results
            
        except Exception as e:
            self.logger.error(f"Error calculating SHAP values: {str(e)}")
            raise
    
    def analyze_feature_interactions(self, shap_results: Dict[str, Any],
                                   top_features: int = 10) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP values.
        
        Args:
            shap_results: SHAP analysis results
            top_features: Number of top features to analyze
            
        Returns:
            Feature interaction analysis
        """
        try:
            self.logger.info("Analyzing feature interactions")
            
            interaction_results = {}
            
            for model_name, shap_data in shap_results.items():
                try:
                    shap_values = shap_data['shap_values']
                    feature_names = shap_data['feature_names']
                    
                    if len(shap_values.shape) != 2:
                        self.logger.warning(f"Cannot analyze interactions for {model_name}: unexpected SHAP shape")
                        continue
                    
                    # Get top features by importance
                    feature_importance = shap_data['feature_importance']
                    top_feature_names = feature_importance.head(top_features)['feature'].tolist()
                    top_feature_indices = [feature_names.index(name) for name in top_feature_names]
                    
                    # Calculate feature correlations in SHAP space
                    top_shap_values = shap_values[:, top_feature_indices]
                    shap_correlations = np.corrcoef(top_shap_values.T)
                    
                    # Create correlation DataFrame
                    shap_corr_df = pd.DataFrame(
                        shap_correlations,
                        index=top_feature_names,
                        columns=top_feature_names
                    )
                    
                    interaction_results[model_name] = {
                        'shap_correlations': shap_corr_df,
                        'top_features': top_feature_names,
                        'feature_synergies': self._identify_synergies(shap_corr_df)
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Interaction analysis failed for {model_name}: {str(e)}")
                    continue
            
            return interaction_results
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature interactions: {str(e)}")
            raise
    
    def _identify_synergies(self, correlation_matrix: pd.DataFrame,
                          threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Identify feature synergies from correlation matrix."""
        synergies = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i+1, len(correlation_matrix)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > threshold:
                    synergies.append({
                        'feature_1': correlation_matrix.index[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'type': 'synergy' if corr_value > 0 else 'competition'
                    })
        
        return sorted(synergies, key=lambda x: abs(x['correlation']), reverse=True)
    
    def generate_uncertainty_report(self, uncertainty_results: Dict[str, Any],
                                  category: str, output_path: str) -> pd.DataFrame:
        """
        Generate comprehensive uncertainty report.
        
        Args:
            uncertainty_results: Results from uncertainty analysis
            category: Steel category name
            output_path: Path to save report
            
        Returns:
            Uncertainty report DataFrame
        """
        try:
            self.logger.info(f"Generating uncertainty report for {category}")
            
            report_data = []
            
            # Bootstrap results
            if 'bootstrap_metrics' in uncertainty_results:
                bootstrap_metrics = uncertainty_results['bootstrap_metrics']
                
                if bootstrap_metrics:
                    boot_mapes = [m['MAPE'] for m in bootstrap_metrics]
                    boot_maes = [m['MAE'] for m in bootstrap_metrics]
                    
                    report_data.append({
                        'Metric': 'Bootstrap_MAPE_Mean',
                        'Value': np.mean(boot_mapes),
                        'Std': np.std(boot_mapes),
                        'Min': np.min(boot_mapes),
                        'Max': np.max(boot_mapes)
                    })
                    
                    report_data.append({
                        'Metric': 'Bootstrap_MAE_Mean',
                        'Value': np.mean(boot_maes),
                        'Std': np.std(boot_maes),
                        'Min': np.min(boot_maes),
                        'Max': np.max(boot_maes)
                    })
            
            # Confidence intervals
            if 'confidence_intervals' in uncertainty_results:
                for conf_level, intervals in uncertainty_results['confidence_intervals'].items():
                    mean_width = np.mean(intervals['upper'] - intervals['lower'])
                    
                    report_data.append({
                        'Metric': f'CI_Width_{conf_level}',
                        'Value': mean_width,
                        'Std': np.std(intervals['upper'] - intervals['lower']),
                        'Min': np.min(intervals['upper'] - intervals['lower']),
                        'Max': np.max(intervals['upper'] - intervals['lower'])
                    })
            
            # Prediction uncertainty
            if 'std_prediction' in uncertainty_results:
                pred_std = uncertainty_results['std_prediction']
                
                report_data.append({
                    'Metric': 'Prediction_Uncertainty',
                    'Value': np.mean(pred_std),
                    'Std': np.std(pred_std),
                    'Min': np.min(pred_std),
                    'Max': np.max(pred_std)
                })
            
            report_df = pd.DataFrame(report_data)
            
            # Save report
            report_file = f"{output_path}/{category}_uncertainty_report.csv"
            report_df.to_csv(report_file, index=False)
            
            self.logger.info(f"Uncertainty report saved to {report_file}")
            
            return report_df
            
        except Exception as e:
            self.logger.error(f"Error generating uncertainty report: {str(e)}")
            raise
    
    def export_shap_summary(self, shap_results: Dict[str, Any],
                          category: str, output_path: str) -> Dict[str, str]:
        """
        Export SHAP analysis summary to CSV files.
        
        Args:
            shap_results: SHAP analysis results
            category: Steel category name
            output_path: Path to save files
            
        Returns:
            Dictionary of saved file paths
        """
        try:
            self.logger.info(f"Exporting SHAP summary for {category}")
            
            saved_files = {}
            
            for model_name, shap_data in shap_results.items():
                try:
                    # Export feature importance
                    if 'feature_importance' in shap_data:
                        importance_file = f"{output_path}/{category}_{model_name}_shap_importance.csv"
                        shap_data['feature_importance'].to_csv(importance_file, index=False)
                        saved_files[f"{model_name}_importance"] = importance_file
                    
                    # Export SHAP values summary
                    if 'shap_values' in shap_data:
                        shap_values = shap_data['shap_values']
                        feature_names = shap_data['feature_names']
                        
                        if len(shap_values.shape) == 2:
                            # Create summary statistics
                            shap_summary = pd.DataFrame({
                                'feature': feature_names,
                                'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),
                                'std_shap': np.std(shap_values, axis=0),
                                'mean_shap': np.mean(shap_values, axis=0)
                            })
                            
                            summary_file = f"{output_path}/{category}_{model_name}_shap_summary.csv"
                            shap_summary.to_csv(summary_file, index=False)
                            saved_files[f"{model_name}_summary"] = summary_file
                
                except Exception as e:
                    self.logger.warning(f"Failed to export SHAP for {model_name}: {str(e)}")
                    continue
            
            self.logger.info(f"SHAP summary exported: {len(saved_files)} files")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error exporting SHAP summary: {str(e)}")
            raise
    
    def comprehensive_uncertainty_analysis(self, model, X: pd.DataFrame, y: pd.Series,
                                         category: str, output_path: str) -> Dict[str, Any]:
        """
        Perform comprehensive uncertainty and interpretability analysis.
        
        Args:
            model: Trained ensemble model
            X: Feature matrix
            y: Target values
            category: Steel category name
            output_path: Path to save results
            
        Returns:
            Complete analysis results
        """
        try:
            self.logger.info(f"Starting comprehensive uncertainty analysis for {category}")
            
            results = {
                'category': category,
                'bootstrap_uncertainty': None,
                'monte_carlo_uncertainty': None,
                'shap_analysis': None,
                'interaction_analysis': None
            }
            
            # Bootstrap uncertainty
            try:
                self.logger.info("Computing bootstrap uncertainty...")
                results['bootstrap_uncertainty'] = self.bootstrap_uncertainty(model, X, y)
            except Exception as e:
                self.logger.warning(f"Bootstrap uncertainty failed: {str(e)}")
            
            # Monte Carlo uncertainty
            try:
                self.logger.info("Computing Monte Carlo uncertainty...")
                results['monte_carlo_uncertainty'] = self.monte_carlo_uncertainty(model, X)
            except Exception as e:
                self.logger.warning(f"Monte Carlo uncertainty failed: {str(e)}")
            
            # SHAP analysis
            try:
                self.logger.info("Computing SHAP analysis...")
                results['shap_analysis'] = self.calculate_shap_values(model, X, category)
            except Exception as e:
                self.logger.warning(f"SHAP analysis failed: {str(e)}")
            
            # Feature interaction analysis
            if results['shap_analysis']:
                try:
                    self.logger.info("Analyzing feature interactions...")
                    results['interaction_analysis'] = self.analyze_feature_interactions(
                        results['shap_analysis']
                    )
                except Exception as e:
                    self.logger.warning(f"Interaction analysis failed: {str(e)}")
            
            # Generate reports
            if results['bootstrap_uncertainty']:
                self.generate_uncertainty_report(
                    results['bootstrap_uncertainty'], category, output_path
                )
            
            if results['shap_analysis']:
                self.export_shap_summary(results['shap_analysis'], category, output_path)
            
            self.logger.info(f"Comprehensive uncertainty analysis completed for {category}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive uncertainty analysis: {str(e)}")
            raise