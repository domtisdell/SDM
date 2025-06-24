"""
Steel Demand ML Model - Model Training Framework
Handles training, cross-validation, and model persistence.
All parameters loaded from CSV configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import json
import time
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Import our modules
import sys
sys.path.append('..')
from data.data_loader import SteelDemandDataLoader
from data.feature_engineering import SteelDemandFeatureEngineering
from models.ensemble_models import EnsembleSteelModel

warnings.filterwarnings('ignore')

class SteelDemandModelTrainer:
    """
    Comprehensive training framework for steel demand forecasting models.
    Handles data loading, feature engineering, model training, and evaluation.
    """
    
    def __init__(self, config_path: str = "config/"):
        """
        Initialize model trainer with configuration.
        
        Args:
            config_path: Path to configuration directory
        """
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.data_loader = SteelDemandDataLoader(config_path)
        self.feature_engineer = SteelDemandFeatureEngineering(self.data_loader, config_path)
        
        # Load configuration
        self._load_config()
        
        # Initialize storage
        self.trained_models = {}
        self.training_results = {}
        self.feature_sets = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_config(self):
        """Load training configuration from CSV files."""
        self.config = {
            'train_test_split': float(self.data_loader.get_model_config('train_test_split')),
            'validation_split': float(self.data_loader.get_model_config('validation_split')),
            'cross_validation_folds': int(self.data_loader.get_model_config('cross_validation_folds')),
            'random_state': int(self.data_loader.get_model_config('random_state')),
            'target_mape': float(self.data_loader.get_model_config('target_mape')),
            'acceptable_mape': float(self.data_loader.get_model_config('acceptable_mape')),
            'min_r2_score': float(self.data_loader.get_model_config('min_r2_score')),
            'max_training_time_hours': float(self.data_loader.get_model_config('max_training_time_hours')),
            'output_directory': self.data_loader.get_model_config('output_directory'),
            'save_model_artifacts': self.data_loader.get_model_config('save_model_artifacts'),
            'export_feature_importance': self.data_loader.get_model_config('export_feature_importance'),
            'export_model_metrics': self.data_loader.get_model_config('export_model_metrics')
        }
    
    def load_and_prepare_data(self, start_year: Optional[int] = None, 
                            end_year: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare all data for training.
        
        Args:
            start_year: Start year for training data
            end_year: End year for training data
            
        Returns:
            Dictionary containing prepared datasets
        """
        try:
            self.logger.info("Loading and preparing data")
            
            # Load all data
            self.data_loader.load_all_data()
            
            # Get historical data
            historical_data = self.data_loader.get_historical_data(start_year, end_year)
            
            # Get projection data for future forecasting
            projection_data = self.data_loader.get_projection_data()
            
            # Get steel categories
            steel_categories = self.data_loader.get_steel_categories()
            
            self.logger.info(f"Historical data: {historical_data.shape}")
            self.logger.info(f"Projection data: {projection_data.shape}")
            self.logger.info(f"Steel categories: {len(steel_categories)}")
            
            return {
                'historical': historical_data,
                'projections': projection_data,
                'steel_categories': steel_categories
            }
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features_for_category(self, data: pd.DataFrame, 
                                    target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare features for a specific steel category.
        
        Args:
            data: Historical dataset
            target_column: Target variable column name
            
        Returns:
            Feature matrix and target series
        """
        try:
            self.logger.info(f"Preparing features for {target_column}")
            
            # Create comprehensive feature set
            features_df = self.feature_engineer.create_features(data, target_column)
            
            # Separate features and target
            if target_column not in features_df.columns:
                raise ValueError(f"Target column {target_column} not found in data")
            
            # Get feature columns (exclude target and Year)
            feature_columns = [col for col in features_df.columns 
                             if col not in [target_column, 'Year']]
            
            X = features_df[feature_columns + ['Year']].copy()
            y = features_df[target_column].copy()
            
            # Remove rows with missing target values
            valid_indices = y.notna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            self.logger.info(f"Feature matrix: {X.shape}, Target: {len(y)}")
            self.logger.info(f"Features created: {len(feature_columns)}")
            
            # Store feature set for this category
            self.feature_sets[target_column] = {
                'feature_columns': feature_columns,
                'total_features': len(feature_columns),
                'data_shape': X.shape
            }
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing features for {target_column}: {str(e)}")
            raise
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets using time series split.
        
        Args:
            X: Feature matrix
            y: Target series
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            split_ratio = self.config['train_test_split']
            split_index = int(len(X) * split_ratio)
            
            # Time series split (no shuffling)
            X_train = X.iloc[:split_index].copy()
            X_test = X.iloc[split_index:].copy()
            y_train = y.iloc[:split_index].copy()
            y_test = y.iloc[split_index:].copy()
            
            self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def train_model_for_category(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: Optional[pd.DataFrame] = None,
                               y_val: Optional[pd.Series] = None,
                               category: str = "steel_category") -> EnsembleSteelModel:
        """
        Train ensemble model for a specific steel category.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            category: Steel category name
            
        Returns:
            Trained ensemble model
        """
        try:
            start_time = time.time()
            self.logger.info(f"Training ensemble model for {category}")
            
            # Initialize ensemble model
            ensemble_model = EnsembleSteelModel(self.data_loader)
            
            # Remove Year column for training (keep for Prophet)
            feature_cols = [col for col in X_train.columns if col != 'Year']
            prophet_cols = ['Year'] + feature_cols[:5]  # Prophet needs Year + limited features
            
            # Prepare different feature sets for different models
            X_train_features = X_train[feature_cols]
            X_train_prophet = X_train[prophet_cols]
            
            if X_val is not None:
                X_val_features = X_val[feature_cols]
                X_val_prophet = X_val[prophet_cols]
            else:
                X_val_features = None
                X_val_prophet = None
            
            # Train ensemble with appropriate feature sets
            # Note: This is a simplified approach - in practice, you might want to
            # train different models with different feature sets
            ensemble_model.fit(X_train_features, y_train, X_val_features, y_val)
            
            training_time = time.time() - start_time
            
            # Check training time constraint
            max_time_seconds = self.config['max_training_time_hours'] * 3600
            if training_time > max_time_seconds:
                self.logger.warning(f"Training time ({training_time:.1f}s) exceeded limit ({max_time_seconds:.1f}s)")
            
            self.logger.info(f"Model training completed for {category} in {training_time:.1f} seconds")
            
            return ensemble_model
            
        except Exception as e:
            self.logger.error(f"Error training model for {category}: {str(e)}")
            raise
    
    def evaluate_model(self, model: EnsembleSteelModel, X_test: pd.DataFrame, 
                      y_test: pd.Series, category: str) -> Dict[str, Any]:
        """
        Evaluate trained model performance.
        
        Args:
            model: Trained ensemble model
            X_test: Test features
            y_test: Test target
            category: Steel category name
            
        Returns:
            Evaluation results
        """
        try:
            self.logger.info(f"Evaluating model for {category}")
            
            # Remove Year column for prediction (except for Prophet)
            feature_cols = [col for col in X_test.columns if col != 'Year']
            X_test_features = X_test[feature_cols]
            
            # Get evaluation metrics
            evaluation_results = model.evaluate(X_test_features, y_test)
            
            # Check performance against benchmarks
            benchmarks = self.data_loader.get_validation_benchmarks(category)
            performance_status = self._assess_performance(evaluation_results['ensemble'], benchmarks)
            
            # Add performance assessment
            evaluation_results['performance_status'] = performance_status
            evaluation_results['category'] = category
            evaluation_results['test_samples'] = len(y_test)
            
            # Get feature importance
            try:
                feature_importance = model.get_feature_importance()
                evaluation_results['feature_importance'] = feature_importance
            except:
                self.logger.warning(f"Could not get feature importance for {category}")
            
            self.logger.info(f"Evaluation completed for {category}")
            self.logger.info(f"Performance status: {performance_status}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating model for {category}: {str(e)}")
            raise
    
    def _assess_performance(self, metrics: Dict[str, float], 
                          benchmarks: pd.DataFrame) -> str:
        """Assess model performance against benchmarks."""
        try:
            if benchmarks.empty:
                return "NO_BENCHMARKS"
            
            mape = metrics.get('MAPE', 100)
            r2 = metrics.get('R2', 0)
            
            # Get benchmark values
            target_mape = self.config['target_mape']
            acceptable_mape = self.config['acceptable_mape']
            min_r2 = self.config['min_r2_score']
            
            # Assess performance
            if mape <= target_mape and r2 >= min_r2:
                return "EXCELLENT"
            elif mape <= acceptable_mape and r2 >= (min_r2 * 0.9):
                return "GOOD"
            elif mape <= (acceptable_mape * 1.5):
                return "ACCEPTABLE"
            else:
                return "POOR"
                
        except Exception as e:
            self.logger.warning(f"Error assessing performance: {str(e)}")
            return "UNKNOWN"
    
    def cross_validate_model(self, X: pd.DataFrame, y: pd.Series, 
                           category: str) -> Dict[str, Any]:
        """
        Perform time series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target series
            category: Steel category name
            
        Returns:
            Cross-validation results
        """
        try:
            self.logger.info(f"Performing cross-validation for {category}")
            
            cv_folds = self.config['cross_validation_folds']
            
            # Use TimeSeriesSplit for temporal data
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            
            cv_scores = []
            cv_details = []
            
            feature_cols = [col for col in X.columns if col != 'Year']
            X_features = X[feature_cols]
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_features)):
                self.logger.info(f"CV Fold {fold + 1}/{cv_folds}")
                
                # Split data
                X_train_cv = X_features.iloc[train_idx]
                X_val_cv = X_features.iloc[val_idx]
                y_train_cv = y.iloc[train_idx]
                y_val_cv = y.iloc[val_idx]
                
                # Train model
                model = EnsembleSteelModel(self.data_loader)
                model.fit(X_train_cv, y_train_cv)
                
                # Evaluate
                fold_results = model.evaluate(X_val_cv, y_val_cv)
                cv_scores.append(fold_results['ensemble']['MAPE'])
                cv_details.append(fold_results['ensemble'])
            
            # Calculate CV statistics
            cv_results = {
                'mean_mape': np.mean(cv_scores),
                'std_mape': np.std(cv_scores),
                'min_mape': np.min(cv_scores),
                'max_mape': np.max(cv_scores),
                'fold_details': cv_details,
                'cv_folds': cv_folds
            }
            
            self.logger.info(f"CV completed for {category}: MAPE = {cv_results['mean_mape']:.2f}% ± {cv_results['std_mape']:.2f}%")
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation for {category}: {str(e)}")
            raise
    
    def train_all_categories(self, data: Dict[str, pd.DataFrame],
                           perform_cv: bool = True) -> Dict[str, Any]:
        """
        Train models for all steel categories.
        
        Args:
            data: Prepared datasets
            perform_cv: Whether to perform cross-validation
            
        Returns:
            Complete training results
        """
        try:
            self.logger.info("Training models for all steel categories")
            
            historical_data = data['historical']
            steel_categories = data['steel_categories']
            
            all_results = {}
            
            for _, category_info in steel_categories.iterrows():
                category_name = category_info['category']
                target_column = category_info['target_column']
                
                self.logger.info(f"Processing category: {category_name}")
                
                try:
                    # Prepare features
                    X, y = self.prepare_features_for_category(historical_data, target_column)
                    
                    # Split data
                    X_train, X_test, y_train, y_test = self.split_data(X, y)
                    
                    # Further split training data for validation
                    val_split = self.config['validation_split']
                    val_index = int(len(X_train) * (1 - val_split))
                    
                    X_train_final = X_train.iloc[:val_index]
                    X_val = X_train.iloc[val_index:]
                    y_train_final = y_train.iloc[:val_index]
                    y_val = y_train.iloc[val_index:]
                    
                    # Train model
                    model = self.train_model_for_category(
                        X_train_final, y_train_final, X_val, y_val, category_name
                    )
                    
                    # Evaluate model
                    evaluation_results = self.evaluate_model(model, X_test, y_test, category_name)
                    
                    # Cross-validation if requested
                    cv_results = None
                    if perform_cv:
                        cv_results = self.cross_validate_model(X_train, y_train, category_name)
                    
                    # Store results
                    category_results = {
                        'model': model,
                        'evaluation': evaluation_results,
                        'cross_validation': cv_results,
                        'feature_info': self.feature_sets.get(target_column, {}),
                        'data_splits': {
                            'train_size': len(X_train),
                            'test_size': len(X_test),
                            'val_size': len(X_val)
                        }
                    }
                    
                    all_results[category_name] = category_results
                    self.trained_models[category_name] = model
                    
                    self.logger.info(f"Completed training for {category_name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train model for {category_name}: {str(e)}")
                    all_results[category_name] = {'error': str(e)}
            
            self.training_results = all_results
            self.logger.info(f"Training completed for {len(all_results)} categories")
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Error training all categories: {str(e)}")
            raise
    
    def save_models_and_results(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save trained models and results to files.
        
        Args:
            output_dir: Output directory (uses config if None)
            
        Returns:
            Dictionary of saved file paths
        """
        try:
            if output_dir is None:
                output_dir = self.config['output_directory']
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            saved_files = {}
            
            # Save models if configured
            if self.config['save_model_artifacts']:
                models_dir = output_path / 'models'
                models_dir.mkdir(exist_ok=True)
                
                for category, model in self.trained_models.items():
                    model_file = models_dir / f'{category}_ensemble_model.joblib'
                    joblib.dump(model, model_file)
                    saved_files[f'{category}_model'] = str(model_file)
            
            # Save training results
            if self.config['export_model_metrics']:
                results_file = output_path / 'training_results.json'
                
                # Convert results to JSON-serializable format
                json_results = {}
                for category, results in self.training_results.items():
                    if 'error' in results:
                        json_results[category] = results
                    else:
                        json_results[category] = {
                            'evaluation': results['evaluation'],
                            'cross_validation': results.get('cross_validation'),
                            'feature_info': results['feature_info'],
                            'data_splits': results['data_splits']
                        }
                
                with open(results_file, 'w') as f:
                    json.dump(json_results, f, indent=2, default=str)
                
                saved_files['training_results'] = str(results_file)
            
            # Save feature importance
            if self.config['export_feature_importance']:
                for category, results in self.training_results.items():
                    if 'evaluation' in results and 'feature_importance' in results['evaluation']:
                        importance_data = results['evaluation']['feature_importance']
                        for model_name, importance_df in importance_data.items():
                            importance_file = output_path / f'{category}_{model_name}_feature_importance.csv'
                            importance_df.to_csv(importance_file, index=False)
                            saved_files[f'{category}_{model_name}_importance'] = str(importance_file)
            
            self.logger.info(f"Saved {len(saved_files)} files to {output_path}")
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving models and results: {str(e)}")
            raise
    
    def generate_summary_report(self) -> pd.DataFrame:
        """
        Generate summary report of all model training results.
        
        Returns:
            Summary DataFrame
        """
        try:
            summary_data = []
            
            # Check if training_results exists and is not None
            if not hasattr(self, 'training_results') or self.training_results is None:
                self.logger.warning("No training results available for summary report")
                return pd.DataFrame(columns=['Category', 'Status', 'Error', 'MAPE', 'R2', 'RMSE', 'Performance', 'Features', 'CV_MAPE'])
            
            for category, results in self.training_results.items():
                # Handle None results
                if results is None:
                    summary_data.append({
                        'Category': category,
                        'Status': 'FAILED',
                        'Error': 'No results available',
                        'MAPE': None,
                        'R2': None,
                        'RMSE': None,
                        'Performance': 'FAILED',
                        'Features': None,
                        'CV_MAPE': None
                    })
                elif 'error' in results:
                    summary_data.append({
                        'Category': category,
                        'Status': 'FAILED',
                        'Error': results['error'],
                        'MAPE': None,
                        'R2': None,
                        'RMSE': None,
                        'Performance': 'FAILED',
                        'Features': None,
                        'CV_MAPE': None
                    })
                else:
                    # Safely extract evaluation data with comprehensive null checking
                    evaluation_data = results.get('evaluation', {}) if results else {}
                    ensemble_eval = evaluation_data.get('ensemble', {}) if evaluation_data else {}
                    cv_results = results.get('cross_validation', {}) if results else {}
                    feature_info = results.get('feature_info', {}) if results else {}
                    
                    # Get performance status from multiple possible locations
                    performance_status = 'UNKNOWN'
                    if evaluation_data and evaluation_data.get('performance_status'):
                        performance_status = evaluation_data.get('performance_status')
                    elif results and results.get('performance_status'):
                        performance_status = results.get('performance_status')
                    
                    summary_data.append({
                        'Category': category,
                        'Status': 'SUCCESS',
                        'Error': None,
                        'MAPE': ensemble_eval.get('MAPE') if ensemble_eval else None,
                        'R2': ensemble_eval.get('R2') if ensemble_eval else None,
                        'RMSE': ensemble_eval.get('RMSE') if ensemble_eval else None,
                        'Performance': performance_status,
                        'Features': feature_info.get('total_features') if feature_info else None,
                        'CV_MAPE': cv_results.get('mean_mape') if cv_results else None
                    })
            
            summary_df = pd.DataFrame(summary_data)
            
            self.logger.info("Generated summary report")
            
            return summary_df
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {str(e)}")
            raise