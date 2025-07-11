"""
Optimized Tree-Based Models for Steel Demand Forecasting

This module provides enhanced XGBoost and RandomForest models with:
- Anti-overfitting configurations for small datasets
- Hyperparameter optimization via cross-validation
- Regularization and ensemble diversity
- Performance monitoring and validation

Designed to replace simple tree models in Track A system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, LeaveOneOut, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

class OptimizedTreeModels:
    """
    Enhanced tree-based models with anti-overfitting configurations.
    
    Features:
    - XGBoost with regularization for small datasets
    - RandomForest with optimized diversity settings
    - Hyperparameter optimization via grid/random search
    - Leave-One-Out cross-validation for small samples
    - Feature importance stability analysis
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize optimized tree models.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Model storage
        self.xgboost_model = None
        self.randomforest_model = None
        
        # Performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Anti-overfitting XGBoost parameters
        self.xgb_param_grid = {
            'n_estimators': [50, 75, 100],           # Reduced from default 100
            'max_depth': [2, 3],                     # Reduced from default 6
            'learning_rate': [0.05, 0.1, 0.15],     # Include slower learning rates
            'min_child_weight': [3, 5, 7],          # Higher minimum samples
            'subsample': [0.8, 0.9],                # Row subsampling for regularization
            'colsample_bytree': [0.8, 0.9],         # Feature subsampling
            'reg_alpha': [0.1, 0.5, 1.0],           # L1 regularization
            'reg_lambda': [0.5, 1.0, 2.0],          # L2 regularization
        }
        
        # Optimized RandomForest parameters
        self.rf_param_grid = {
            'n_estimators': [100, 150, 200],         # More trees for stability
            'max_depth': [4, 5, 6, None],            # Allow deeper trees but test None
            'min_samples_split': [3, 5, 7],          # Higher minimum splits
            'min_samples_leaf': [2, 3, 4],           # Higher minimum leaf samples  
            'max_features': ['sqrt', 'log2', 0.8],   # Feature subsampling strategies
            'bootstrap': [True],                      # Enable bootstrapping
        }
        
        # Reduced parameter grids for small datasets
        self.xgb_small_param_grid = {
            'n_estimators': [50, 75],
            'max_depth': [2, 3],
            'learning_rate': [0.05, 0.1],
            'min_child_weight': [5, 7],
            'reg_alpha': [0.5, 1.0],
            'reg_lambda': [1.0, 2.0],
        }
        
        self.rf_small_param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [4, 5],
            'min_samples_split': [3, 5],
            'min_samples_leaf': [2, 3],
            'max_features': ['sqrt', 0.8],
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        return logging.getLogger(__name__)
    
    def train_optimized_xgboost(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """
        Train XGBoost with anti-overfitting configuration.
        
        Args:
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary with model, predictions, and performance metrics
        """
        self.logger.info(f"Training optimized XGBoost for {category}")
        
        # Choose parameter grid based on dataset size
        if len(X) <= 15:
            param_grid = self.xgb_small_param_grid
            self.logger.info("Using reduced parameter grid for small dataset")
        else:
            param_grid = self.xgb_param_grid
        
        # Cross-validation strategy
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        # Base XGBoost model with conservative defaults
        xgb_model = xgb.XGBRegressor(
            random_state=self.random_state,
            verbosity=0,
            n_jobs=1,  # Avoid threading issues with small datasets
            objective='reg:squarederror'
        )
        
        # Use RandomizedSearchCV for efficiency with small datasets
        if len(X) <= 15:
            # Exhaustive grid search for very small datasets
            search = GridSearchCV(
                xgb_model,
                param_grid,
                cv=cv_strategy,
                scoring='neg_mean_absolute_error',
                n_jobs=-1  # Use all available cores
            )
        else:
            # Randomized search for larger datasets
            search = RandomizedSearchCV(
                xgb_model,
                param_grid,
                n_iter=20,  # Limited iterations for efficiency
                cv=cv_strategy,
                scoring='neg_mean_absolute_error',
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
        
        # Fit the model
        search.fit(X, y)
        self.xgboost_model = search.best_estimator_
        
        # Generate predictions
        predictions = self.xgboost_model.predict(X)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.xgboost_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Additional XGBoost-specific metrics
        n_trees_used = self.xgboost_model.n_estimators
        actual_max_depth = self.xgboost_model.max_depth
        
        result = {
            'model': self.xgboost_model,
            'predictions': predictions,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': search.best_params_,
            'cv_score': -search.best_score_,
            'n_trees': n_trees_used,
            'max_depth': actual_max_depth,
            'regularization': {
                'reg_alpha': self.xgboost_model.reg_alpha,
                'reg_lambda': self.xgboost_model.reg_lambda
            }
        }
        
        # Store performance
        self.model_performance[f'{category}_XGBoost'] = metrics
        self.feature_importance[f'{category}_XGBoost'] = feature_importance
        
        self.logger.info(f"XGBoost - Best params: n_est={search.best_params_.get('n_estimators', 'N/A')}, "
                        f"depth={search.best_params_.get('max_depth', 'N/A')}, "
                        f"lr={search.best_params_.get('learning_rate', 'N/A')}, "
                        f"α={search.best_params_.get('reg_alpha', 'N/A')}, "
                        f"λ={search.best_params_.get('reg_lambda', 'N/A')}")
        
        self.logger.info(f"XGBoost - CV MAE: {-search.best_score_:.3f}, "
                        f"MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
        
        # Warning if still overfitting
        if metrics['R2'] > 0.98:
            self.logger.warning(f"XGBoost may still be overfitting (R²={metrics['R2']:.3f}). "
                               "Consider further regularization.")
        
        return result
    
    def train_optimized_randomforest(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """
        Train RandomForest with optimized diversity and performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary with model, predictions, and performance metrics
        """
        self.logger.info(f"Training optimized RandomForest for {category}")
        
        # Choose parameter grid based on dataset size
        if len(X) <= 15:
            param_grid = self.rf_small_param_grid
            self.logger.info("Using reduced parameter grid for small dataset")
        else:
            param_grid = self.rf_param_grid
        
        # Cross-validation strategy
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        # Base RandomForest model
        rf_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=1,  # Avoid threading issues with small datasets
            bootstrap=True,
            oob_score=True  # Out-of-bag score for additional validation
        )
        
        # Grid search for hyperparameter optimization
        if len(X) <= 15:
            # Full grid search for small datasets
            search = GridSearchCV(
                rf_model,
                param_grid,
                cv=cv_strategy,
                scoring='neg_mean_absolute_error',
                n_jobs=-1  # Use all available cores
            )
        else:
            # Randomized search for larger datasets
            search = RandomizedSearchCV(
                rf_model,
                param_grid,
                n_iter=15,  # Limited iterations
                cv=cv_strategy,
                scoring='neg_mean_absolute_error',
                random_state=self.random_state,
                n_jobs=-1  # Use all available cores
            )
        
        # Fit the model
        search.fit(X, y)
        self.randomforest_model = search.best_estimator_
        
        # Generate predictions
        predictions = self.randomforest_model.predict(X)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Feature importance with stability analysis
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.randomforest_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # RandomForest-specific metrics
        oob_score = getattr(self.randomforest_model, 'oob_score_', None)
        n_trees = self.randomforest_model.n_estimators
        max_depth = self.randomforest_model.max_depth
        
        result = {
            'model': self.randomforest_model,
            'predictions': predictions,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': search.best_params_,
            'cv_score': -search.best_score_,
            'oob_score': oob_score,
            'n_trees': n_trees,
            'max_depth': max_depth,
            'diversity_metrics': {
                'max_features': self.randomforest_model.max_features,
                'min_samples_split': self.randomforest_model.min_samples_split,
                'min_samples_leaf': self.randomforest_model.min_samples_leaf
            }
        }
        
        # Store performance
        self.model_performance[f'{category}_RandomForest'] = metrics
        self.feature_importance[f'{category}_RandomForest'] = feature_importance
        
        self.logger.info(f"RandomForest - Best params: n_est={search.best_params_.get('n_estimators', 'N/A')}, "
                        f"depth={search.best_params_.get('max_depth', 'N/A')}, "
                        f"min_split={search.best_params_.get('min_samples_split', 'N/A')}, "
                        f"max_features={search.best_params_.get('max_features', 'N/A')}")
        
        oob_str = f"{oob_score:.3f}" if oob_score is not None else "N/A"
        self.logger.info(f"RandomForest - CV MAE: {-search.best_score_:.3f}, "
                        f"OOB Score: {oob_str}, "
                        f"MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
        
        return result
    
    def predict(self, X: pd.DataFrame, model_type: str = 'XGBoost') -> np.ndarray:
        """
        Generate predictions using trained model.
        
        Args:
            X: Feature matrix for prediction
            model_type: Type of model to use ('XGBoost', 'RandomForest')
            
        Returns:
            Array of predictions
        """
        if model_type == 'XGBoost' and self.xgboost_model:
            predictions = self.xgboost_model.predict(X)
        elif model_type == 'RandomForest' and self.randomforest_model:
            predictions = self.randomforest_model.predict(X)
        else:
            raise ValueError(f"Model {model_type} not trained or invalid model type")
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def analyze_feature_importance_stability(self, X: pd.DataFrame, y: pd.Series, 
                                           model_type: str = 'RandomForest', 
                                           n_runs: int = 10) -> Dict[str, Any]:
        """
        Analyze feature importance stability across multiple training runs.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to analyze
            n_runs: Number of training runs with different random seeds
            
        Returns:
            Dictionary with stability metrics
        """
        self.logger.info(f"Analyzing feature importance stability for {model_type}")
        
        importance_runs = []
        
        for run in range(n_runs):
            # Temporarily change random state
            original_seed = self.random_state
            self.random_state = original_seed + run
            
            try:
                if model_type == 'XGBoost':
                    result = self.train_optimized_xgboost(X, y, f"stability_run_{run}")
                elif model_type == 'RandomForest':
                    result = self.train_optimized_randomforest(X, y, f"stability_run_{run}")
                
                importance_runs.append(result['feature_importance']['importance'].values)
                
            except Exception as e:
                self.logger.warning(f"Stability run {run} failed: {str(e)}")
            
            finally:
                # Restore original random state
                self.random_state = original_seed
        
        if importance_runs:
            importance_matrix = np.array(importance_runs)
            
            # Calculate stability metrics
            mean_importance = np.mean(importance_matrix, axis=0)
            std_importance = np.std(importance_matrix, axis=0)
            cv_importance = std_importance / (mean_importance + 1e-10)  # Coefficient of variation
            
            # Feature ranking stability (Spearman correlation between runs)
            from scipy.stats import spearmanr
            ranking_correlations = []
            for i in range(len(importance_runs)):
                for j in range(i+1, len(importance_runs)):
                    corr, _ = spearmanr(importance_runs[i], importance_runs[j])
                    ranking_correlations.append(corr)
            
            stability_df = pd.DataFrame({
                'feature': X.columns,
                'mean_importance': mean_importance,
                'std_importance': std_importance,
                'cv_importance': cv_importance
            }).sort_values('mean_importance', ascending=False)
            
            return {
                'stability_dataframe': stability_df,
                'mean_ranking_correlation': np.mean(ranking_correlations),
                'ranking_correlation_std': np.std(ranking_correlations),
                'most_stable_features': stability_df.nsmallest(5, 'cv_importance')['feature'].tolist(),
                'least_stable_features': stability_df.nlargest(5, 'cv_importance')['feature'].tolist(),
                'n_successful_runs': len(importance_runs)
            }
        else:
            return {'error': 'All stability runs failed'}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
    
    def get_model_complexity_metrics(self, model_type: str) -> Dict[str, Any]:
        """
        Get complexity metrics for overfitting assessment.
        
        Args:
            model_type: Type of model ('XGBoost', 'RandomForest')
            
        Returns:
            Dictionary with complexity metrics
        """
        if model_type == 'XGBoost' and self.xgboost_model:
            return {
                'n_estimators': self.xgboost_model.n_estimators,
                'max_depth': self.xgboost_model.max_depth,
                'min_child_weight': self.xgboost_model.min_child_weight,
                'reg_alpha': self.xgboost_model.reg_alpha,
                'reg_lambda': self.xgboost_model.reg_lambda,
                'learning_rate': self.xgboost_model.learning_rate,
                'subsample': self.xgboost_model.subsample,
                'complexity_score': (
                    self.xgboost_model.n_estimators * self.xgboost_model.max_depth /
                    (self.xgboost_model.reg_alpha + self.xgboost_model.reg_lambda + 1)
                )
            }
        
        elif model_type == 'RandomForest' and self.randomforest_model:
            return {
                'n_estimators': self.randomforest_model.n_estimators,
                'max_depth': self.randomforest_model.max_depth or 999,  # None means unlimited
                'min_samples_split': self.randomforest_model.min_samples_split,
                'min_samples_leaf': self.randomforest_model.min_samples_leaf,
                'max_features': self.randomforest_model.max_features,
                'complexity_score': (
                    self.randomforest_model.n_estimators * (self.randomforest_model.max_depth or 10) /
                    (self.randomforest_model.min_samples_split * self.randomforest_model.min_samples_leaf)
                )
            }
        
        else:
            return {'error': f'Model {model_type} not trained'}