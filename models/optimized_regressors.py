"""
Optimized Regression Models for Steel Demand Forecasting

This module provides enhanced regression models (Ridge, Lasso, ElasticNet) with:
- Multicollinearity handling via VIF analysis
- Feature preprocessing and scaling
- Hyperparameter optimization via cross-validation
- Robust validation for small datasets

Designed to replace LinearRegression in Track A system.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

class OptimizedRegressors:
    """
    Advanced regression models with preprocessing and optimization for steel demand forecasting.
    
    Features:
    - Ridge Regression with L2 regularization
    - Lasso Regression with automatic feature selection
    - ElasticNet combining Ridge + Lasso benefits
    - VIF-based multicollinearity removal
    - Leave-One-Out cross-validation for small datasets
    - Bootstrap confidence intervals
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize optimized regressors.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.logger = self._setup_logging()
        
        # Model storage
        self.ridge_model = None
        self.lasso_model = None
        self.elastic_model = None
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.vif_features = None
        
        # Hyperparameter grids
        self.param_grids = {
            'Ridge': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
            'Lasso': {'alpha': [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]},
            'ElasticNet': {
                'alpha': [0.1, 0.5, 1.0, 2.0], 
                'l1_ratio': [0.3, 0.5, 0.7, 0.9]
            }
        }
        
        # Performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        return logging.getLogger(__name__)
    
    def remove_multicollinearity(self, X: pd.DataFrame, vif_threshold: float = 50.0) -> pd.DataFrame:
        """
        Remove features with high variance inflation factor (VIF) to handle multicollinearity.
        
        Args:
            X: Feature matrix
            vif_threshold: VIF threshold above which features are removed
            
        Returns:
            Feature matrix with multicollinear features removed
        """
        self.logger.info(f"Checking multicollinearity with VIF threshold {vif_threshold}")
        
        X_clean = X.copy()
        features_to_remove = []
        
        # Calculate VIF for each feature
        for i in range(X_clean.shape[1]):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                feature_name = X_clean.columns[i]
                
                if vif > vif_threshold:
                    features_to_remove.append(feature_name)
                    self.logger.info(f"Removing {feature_name} (VIF: {vif:.2f})")
                else:
                    self.logger.debug(f"Keeping {feature_name} (VIF: {vif:.2f})")
                    
            except Exception as e:
                self.logger.warning(f"Could not calculate VIF for feature {i}: {str(e)}")
        
        # Remove high-VIF features
        X_clean = X_clean.drop(columns=features_to_remove)
        self.vif_features = X_clean.columns.tolist()
        
        self.logger.info(f"Multicollinearity check complete. Features: {len(X.columns)} → {len(X_clean.columns)}")
        return X_clean
    
    def preprocess_features(self, X: pd.DataFrame, y: pd.Series = None, fit_scaler: bool = True) -> pd.DataFrame:
        """
        Preprocess features for regularized regression models.
        
        Args:
            X: Feature matrix
            y: Target variable (used for feature selection)
            fit_scaler: Whether to fit the scaler (True for training, False for prediction)
            
        Returns:
            Preprocessed feature matrix
        """
        self.logger.info("Preprocessing features for regularized models...")
        
        # Step 1: Remove multicollinearity
        if fit_scaler:  # Only during training
            X_vif = self.remove_multicollinearity(X)
        else:  # During prediction, use stored feature list
            if self.vif_features:
                X_vif = X[self.vif_features]
            else:
                X_vif = X.copy()
        
        # Step 2: Handle missing values
        X_clean = X_vif.fillna(X_vif.mean())
        
        # Step 3: Feature scaling
        if fit_scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_clean),
                columns=X_clean.columns,
                index=X_clean.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X_clean),
                columns=X_clean.columns,
                index=X_clean.index
            )
        
        # Step 4: Optional feature selection (if we have target)
        if y is not None and fit_scaler:
            self.logger.info("Performing feature selection...")
            # Use SelectKBest to identify most predictive features
            k = min(len(X_scaled.columns), max(3, len(X_scaled.columns) // 2))  # Select top 50% or at least 3
            selector = SelectKBest(score_func=f_regression, k=k)
            
            X_selected = selector.fit_transform(X_scaled, y)
            self.selected_features = X_scaled.columns[selector.get_support()].tolist()
            self.feature_selector = selector
            
            X_final = pd.DataFrame(X_selected, columns=self.selected_features, index=X_scaled.index)
            self.logger.info(f"Feature selection: {len(X_scaled.columns)} → {len(self.selected_features)}")
        else:
            # Use previously selected features or all features
            if self.selected_features:
                X_final = X_scaled[self.selected_features]
            else:
                X_final = X_scaled
        
        return X_final
    
    def train_ridge_regression(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """
        Train Ridge regression with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary with model, predictions, and performance metrics
        """
        self.logger.info(f"Training Ridge regression for {category}")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, y, fit_scaler=True)
        
        # Cross-validation for hyperparameter tuning
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        ridge = Ridge(random_state=self.random_state)
        grid_search = GridSearchCV(
            ridge, 
            self.param_grids['Ridge'],
            cv=cv_strategy,
            scoring='neg_mean_absolute_error',
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_processed, y)
        self.ridge_model = grid_search.best_estimator_
        
        # Generate predictions
        predictions = self.ridge_model.predict(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Feature importance (absolute coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'coefficient': self.ridge_model.coef_,
            'abs_coefficient': np.abs(self.ridge_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        result = {
            'model': self.ridge_model,
            'predictions': predictions,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,
            'intercept': self.ridge_model.intercept_
        }
        
        # Store performance
        self.model_performance[f'{category}_Ridge'] = metrics
        self.feature_importance[f'{category}_Ridge'] = feature_importance
        
        self.logger.info(f"Ridge - Best alpha: {grid_search.best_params_['alpha']}, "
                        f"CV MAE: {-grid_search.best_score_:.3f}, "
                        f"MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
        
        return result
    
    def train_lasso_regression(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """
        Train Lasso regression with automatic feature selection.
        
        Args:
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary with model, predictions, and performance metrics
        """
        self.logger.info(f"Training Lasso regression for {category}")
        
        # Preprocess features (but don't do additional feature selection since Lasso does it)
        X_processed = self.preprocess_features(X, y=None, fit_scaler=True)  # No y to skip SelectKBest
        
        # Cross-validation for hyperparameter tuning
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        lasso = Lasso(random_state=self.random_state, max_iter=2000)
        grid_search = GridSearchCV(
            lasso,
            self.param_grids['Lasso'],
            cv=cv_strategy,
            scoring='neg_mean_absolute_error',
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_processed, y)
        self.lasso_model = grid_search.best_estimator_
        
        # Generate predictions
        predictions = self.lasso_model.predict(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Feature importance (Lasso coefficients, many will be zero)
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'coefficient': self.lasso_model.coef_,
            'abs_coefficient': np.abs(self.lasso_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Count selected features (non-zero coefficients)
        selected_features = (feature_importance['abs_coefficient'] > 1e-10).sum()
        
        result = {
            'model': self.lasso_model,
            'predictions': predictions,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,
            'selected_features': selected_features,
            'intercept': self.lasso_model.intercept_
        }
        
        # Store performance
        self.model_performance[f'{category}_Lasso'] = metrics
        self.feature_importance[f'{category}_Lasso'] = feature_importance
        
        self.logger.info(f"Lasso - Best alpha: {grid_search.best_params_['alpha']}, "
                        f"Selected features: {selected_features}/{len(X_processed.columns)}, "
                        f"CV MAE: {-grid_search.best_score_:.3f}, "
                        f"MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
        
        return result
    
    def train_elastic_net(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """
        Train ElasticNet regression combining Ridge and Lasso benefits.
        
        Args:
            X: Feature matrix
            y: Target variable
            category: Steel category name
            
        Returns:
            Dictionary with model, predictions, and performance metrics
        """
        self.logger.info(f"Training ElasticNet regression for {category}")
        
        # Preprocess features
        X_processed = self.preprocess_features(X, y, fit_scaler=True)
        
        # Cross-validation for hyperparameter tuning
        cv_strategy = LeaveOneOut() if len(X) <= 20 else 5
        
        elastic = ElasticNet(random_state=self.random_state, max_iter=2000)
        grid_search = GridSearchCV(
            elastic,
            self.param_grids['ElasticNet'],
            cv=cv_strategy,
            scoring='neg_mean_absolute_error',
            n_jobs=-1  # Use all available cores
        )
        
        grid_search.fit(X_processed, y)
        self.elastic_model = grid_search.best_estimator_
        
        # Generate predictions
        predictions = self.elastic_model.predict(X_processed)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, predictions)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_processed.columns,
            'coefficient': self.elastic_model.coef_,
            'abs_coefficient': np.abs(self.elastic_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Count selected features
        selected_features = (feature_importance['abs_coefficient'] > 1e-10).sum()
        
        result = {
            'model': self.elastic_model,
            'predictions': predictions,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_,
            'cv_score': -grid_search.best_score_,
            'selected_features': selected_features,
            'intercept': self.elastic_model.intercept_
        }
        
        # Store performance
        self.model_performance[f'{category}_ElasticNet'] = metrics
        self.feature_importance[f'{category}_ElasticNet'] = feature_importance
        
        self.logger.info(f"ElasticNet - Best params: α={grid_search.best_params_['alpha']}, "
                        f"l1_ratio={grid_search.best_params_['l1_ratio']}, "
                        f"Selected features: {selected_features}/{len(X_processed.columns)}, "
                        f"CV MAE: {-grid_search.best_score_:.3f}, "
                        f"MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
        
        return result
    
    def predict(self, X: pd.DataFrame, model_type: str = 'Ridge') -> np.ndarray:
        """
        Generate predictions using trained model.
        
        Args:
            X: Feature matrix for prediction
            model_type: Type of model to use ('Ridge', 'Lasso', 'ElasticNet')
            
        Returns:
            Array of predictions
        """
        # Get the appropriate model
        if model_type == 'Ridge' and self.ridge_model:
            model = self.ridge_model
        elif model_type == 'Lasso' and self.lasso_model:
            model = self.lasso_model
        elif model_type == 'ElasticNet' and self.elastic_model:
            model = self.elastic_model
        else:
            raise ValueError(f"Model {model_type} not trained or invalid model type")
        
        # Preprocess features (using fitted scaler)
        X_processed = self.preprocess_features(X, fit_scaler=False)
        
        # Generate predictions
        predictions = model.predict(X_processed)
        
        # Ensure positive predictions
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
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
    
    def get_best_model(self, category: str) -> Tuple[str, float]:
        """
        Determine the best performing model for a category.
        
        Args:
            category: Steel category name
            
        Returns:
            Tuple of (best_model_name, best_score)
        """
        model_scores = {}
        
        for model_name in ['Ridge', 'Lasso', 'ElasticNet']:
            key = f'{category}_{model_name}'
            if key in self.model_performance:
                metrics = self.model_performance[key]
                # Combined score: weight R² higher, penalize high MAPE
                score = metrics['R2'] * (1 - metrics['MAPE'] / 100)
                model_scores[model_name] = score
        
        if model_scores:
            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            return best_model, best_score
        else:
            return 'Ridge', 0.0  # Default fallback
    
    def bootstrap_validation(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'Ridge', 
                           n_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Perform bootstrap validation for confidence intervals.
        
        Args:
            X: Feature matrix
            y: Target variable  
            model_type: Type of model to validate
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with bootstrap statistics
        """
        self.logger.info(f"Performing bootstrap validation for {model_type} (n={n_bootstrap})")
        
        bootstrap_scores = []
        bootstrap_coefficients = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X.iloc[indices]
            y_boot = y.iloc[indices]
            
            try:
                # Train model on bootstrap sample
                if model_type == 'Ridge':
                    result = self.train_ridge_regression(X_boot, y_boot, f"bootstrap_{i}")
                elif model_type == 'Lasso':
                    result = self.train_lasso_regression(X_boot, y_boot, f"bootstrap_{i}")
                elif model_type == 'ElasticNet':
                    result = self.train_elastic_net(X_boot, y_boot, f"bootstrap_{i}")
                
                bootstrap_scores.append(result['metrics']['R2'])
                bootstrap_coefficients.append(result['model'].coef_)
                
            except Exception as e:
                self.logger.warning(f"Bootstrap sample {i} failed: {str(e)}")
        
        if bootstrap_scores:
            return {
                'mean_r2': np.mean(bootstrap_scores),
                'std_r2': np.std(bootstrap_scores),
                'ci_lower_r2': np.percentile(bootstrap_scores, 2.5),
                'ci_upper_r2': np.percentile(bootstrap_scores, 97.5),
                'coefficient_stability': np.std(bootstrap_coefficients, axis=0),
                'n_successful_boots': len(bootstrap_scores)
            }
        else:
            return {'error': 'All bootstrap samples failed'}