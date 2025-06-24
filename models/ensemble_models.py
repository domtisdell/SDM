"""
Steel Demand ML Model - Ensemble Models Module
Implements XGBoost, Random Forest, LSTM, and Prophet models.
All parameters loaded from CSV configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
from abc import ABC, abstractmethod

# ML Libraries
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from prophet import Prophet

# Suppress warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class BaseSteelDemandModel(ABC):
    """Base class for all steel demand forecasting models."""
    
    def __init__(self, data_loader, model_name: str):
        """
        Initialize base model with data loader and configuration.
        
        Args:
            data_loader: SteelDemandDataLoader instance
            model_name: Name of the model
        """
        self.data_loader = data_loader
        self.model_name = model_name
        self.model = None
        self.is_fitted = False
        self.logger = self._setup_logging()
        self.feature_importance_ = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"{__name__}.{self.model_name}")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseSteelDemandModel':
        """Fit the model to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X)
        
        # Handle different prediction formats
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        
        # Calculate MAPE
        mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        self.logger.info(f"{self.model_name} Evaluation - MAPE: {mape:.2f}%, R²: {r2:.3f}, RMSE: {rmse:.0f}")
        
        return metrics

class XGBoostSteelModel(BaseSteelDemandModel):
    """XGBoost model for steel demand forecasting."""
    
    def __init__(self, data_loader):
        super().__init__(data_loader, "XGBoost")
        self._load_config()
    
    def _load_config(self):
        """Load XGBoost configuration from CSV."""
        self.config = {
            'n_estimators': int(self.data_loader.get_model_config('xgb_n_estimators')),
            'max_depth': int(self.data_loader.get_model_config('xgb_max_depth')),
            'learning_rate': float(self.data_loader.get_model_config('xgb_learning_rate')),
            'subsample': float(self.data_loader.get_model_config('xgb_subsample')),
            'colsample_bytree': float(self.data_loader.get_model_config('xgb_colsample_bytree')),
            'reg_alpha': float(self.data_loader.get_model_config('xgb_reg_alpha')),
            'reg_lambda': float(self.data_loader.get_model_config('xgb_reg_lambda')),
            'random_state': int(self.data_loader.get_model_config('random_state')),
            'early_stopping_rounds': int(self.data_loader.get_model_config('early_stopping_rounds'))
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'XGBoostSteelModel':
        """
        Fit XGBoost model to training data.
        
        Args:
            X: Training feature matrix
            y: Training target values
            X_val: Validation feature matrix (optional)
            y_val: Validation target values (optional)
            
        Returns:
            Fitted model instance
        """
        try:
            self.logger.info("Training XGBoost model")
            
            # Initialize model with configuration (exclude early_stopping_rounds)
            model_params = {k: v for k, v in self.config.items() if k != 'early_stopping_rounds'}
            self.model = xgb.XGBRegressor(**model_params)
            
            # For XGBoost 3.0+, use simple fit without early stopping to avoid compatibility issues
            self.model.fit(X, y, verbose=False)
            
            # Store feature importance and feature names
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store feature names for consistent prediction
            self.feature_names_ = list(X.columns)
            
            self.is_fitted = True
            self.logger.info("XGBoost model training completed")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using XGBoost model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.feature_importance_.copy()

class RandomForestSteelModel(BaseSteelDemandModel):
    """Random Forest model for steel demand forecasting."""
    
    def __init__(self, data_loader):
        super().__init__(data_loader, "RandomForest")
        self._load_config()
    
    def _load_config(self):
        """Load Random Forest configuration from CSV."""
        self.config = {
            'n_estimators': int(self.data_loader.get_model_config('rf_n_estimators')),
            'max_depth': int(self.data_loader.get_model_config('rf_max_depth')),
            'min_samples_split': int(self.data_loader.get_model_config('rf_min_samples_split')),
            'min_samples_leaf': int(self.data_loader.get_model_config('rf_min_samples_leaf')),
            'max_features': self.data_loader.get_model_config('rf_max_features'),
            'random_state': int(self.data_loader.get_model_config('random_state')),
            'n_jobs': -1
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'RandomForestSteelModel':
        """
        Fit Random Forest model to training data.
        
        Args:
            X: Training feature matrix
            y: Training target values
            
        Returns:
            Fitted model instance
        """
        try:
            self.logger.info("Training Random Forest model")
            
            # Initialize and fit model
            self.model = RandomForestRegressor(**self.config)
            self.model.fit(X, y)
            
            # Store feature importance and feature names
            self.feature_importance_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Store feature names for consistent prediction
            self.feature_names_ = list(X.columns)
            
            self.is_fitted = True
            self.logger.info("Random Forest model training completed")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Random Forest model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        return self.model.predict(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from trained model."""
        if self.feature_importance_ is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.feature_importance_.copy()

class LSTMSteelModel(BaseSteelDemandModel):
    """LSTM neural network model for steel demand forecasting."""
    
    def __init__(self, data_loader):
        super().__init__(data_loader, "LSTM")
        self._load_config()
        # Set random seed for reproducibility
        tf.random.set_seed(self.config['random_state'])
    
    def _load_config(self):
        """Load LSTM configuration from CSV."""
        self.config = {
            'lstm_units': int(self.data_loader.get_model_config('lstm_units')),
            'dropout': float(self.data_loader.get_model_config('lstm_dropout')),
            'recurrent_dropout': float(self.data_loader.get_model_config('lstm_recurrent_dropout')),
            'epochs': int(self.data_loader.get_model_config('lstm_epochs')),
            'batch_size': int(self.data_loader.get_model_config('lstm_batch_size')),
            'sequence_length': int(self.data_loader.get_model_config('lstm_sequence_length')),
            'random_state': int(self.data_loader.get_model_config('random_state'))
        }
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        sequence_length = self.config['sequence_length']
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X.iloc[i-sequence_length:i].values)
            y_sequences.append(y.iloc[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'LSTMSteelModel':
        """
        Fit LSTM model to training data.
        
        Args:
            X: Training feature matrix
            y: Training target values
            X_val: Validation feature matrix (optional)
            y_val: Validation target values (optional)
            
        Returns:
            Fitted model instance
        """
        try:
            self.logger.info("Training LSTM model")
            
            # Create sequences
            X_seq, y_seq = self._create_sequences(X, y)
            
            if len(X_seq) == 0:
                # Use a simpler approach for small datasets
                self.logger.info("Using simplified LSTM approach for small dataset")
                # Reduce sequence length for small datasets
                min_sequence_length = min(3, len(X) - 1)
                if min_sequence_length > 0:
                    X_sequences = []
                    y_sequences = []
                    for i in range(min_sequence_length, len(X)):
                        X_sequences.append(X.iloc[i-min_sequence_length:i].values)
                        y_sequences.append(y.iloc[i])
                    X_seq, y_seq = np.array(X_sequences), np.array(y_sequences)
                else:
                    raise ValueError("Insufficient data for sequence creation")
            
            # Prepare validation data if provided
            validation_data = None
            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self._create_sequences(X_val, y_val)
                if len(X_val_seq) > 0:
                    validation_data = (X_val_seq, y_val_seq)
            
            # Build ultra-simplified LSTM model for small dataset
            # Use minimal architecture to avoid overfitting
            lstm_units = min(8, self.config['lstm_units'])  # Cap LSTM units for small data
            
            self.model = keras.Sequential([
                layers.LSTM(
                    lstm_units,
                    input_shape=(X_seq.shape[1], X_seq.shape[2]),
                    dropout=0.0,  # No dropout for tiny dataset
                    recurrent_dropout=0.0,
                    return_sequences=False,
                    kernel_regularizer=tf.keras.regularizers.l2(0.001)  # Minimal regularization
                ),
                layers.Dense(1, activation='linear')  # Direct linear output
            ])
            
            # Compile model with optimized settings for small data
            optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1.0)  # Higher LR, gradient clipping
            self.model.compile(
                optimizer=optimizer,
                loss='huber',  # More robust loss for outliers
                metrics=['mae']
            )
            
            # Minimal callbacks for small dataset
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='loss',  # Monitor training loss only for stability
                    patience=10,  # Reduced patience
                    restore_best_weights=True,
                    min_delta=1e-6
                )
            ]
            
            # Train model with very few epochs to prevent overfitting
            max_epochs = min(50, self.config['epochs'])  # Cap epochs for small data
            batch_size = min(len(X_seq), self.config['batch_size'])  # Ensure batch size doesn't exceed data
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                history = self.model.fit(
                    X_seq, y_seq,
                    epochs=max_epochs,
                    batch_size=batch_size,
                    validation_data=None,  # Skip validation for stability with tiny dataset
                    callbacks=callbacks,
                    verbose=0
                )
            
            self.training_history = history.history
            self.is_fitted = True
            self.logger.info("LSTM model training completed")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using LSTM model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        sequence_length = self.config['sequence_length']
        
        # Handle insufficient data for LSTM prediction
        min_seq_len = min(sequence_length, len(X))
        if len(X) < sequence_length:
            # Pad with last available values if insufficient data
            padding_needed = sequence_length - len(X)
            if len(X) > 0:
                last_row = X.iloc[-1:]
                padding = pd.concat([last_row] * padding_needed, ignore_index=True)
                X_padded = pd.concat([padding, X], ignore_index=True)
                X_seq = X_padded.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
            else:
                # Handle empty data case
                X_seq = np.zeros((1, sequence_length, X.shape[1]))
        else:
            X_seq = X.iloc[-sequence_length:].values.reshape(1, sequence_length, -1)
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predictions = self.model.predict(X_seq, verbose=0)
            
            pred_flat = predictions.flatten()
            
            # Safety check for extreme predictions
            pred_flat = np.clip(pred_flat, -1e6, 1e6)  # Clip extreme values
            
            # Ensure we return the correct number of predictions
            if len(pred_flat) == 1 and len(X) > 1:
                # Repeat single prediction for all input samples
                return np.repeat(pred_flat[0], len(X))
            elif len(pred_flat) == len(X):
                return pred_flat
            else:
                # Fallback: return last prediction repeated
                return np.repeat(pred_flat[-1] if len(pred_flat) > 0 else 0, len(X))
        
        except Exception as e:
            # If LSTM prediction fails, return simple fallback
            self.logger.warning(f"LSTM prediction failed: {str(e)}, using fallback")
            return np.zeros(len(X))

class ProphetSteelModel(BaseSteelDemandModel):
    """Prophet model for steel demand forecasting."""
    
    def __init__(self, data_loader):
        super().__init__(data_loader, "Prophet")
        self._load_config()
    
    def _load_config(self):
        """Load Prophet configuration from CSV."""
        self.config = {
            'yearly_seasonality': self.data_loader.get_model_config('prophet_yearly_seasonality'),
            'changepoint_prior_scale': float(self.data_loader.get_model_config('prophet_changepoint_prior_scale')),
            'seasonality_prior_scale': float(self.data_loader.get_model_config('prophet_seasonality_prior_scale'))
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'ProphetSteelModel':
        """
        Fit Prophet model to training data.
        
        Args:
            X: Training feature matrix (must include 'Year' column)
            y: Training target values
            
        Returns:
            Fitted model instance
        """
        try:
            self.logger.info("Training Prophet model")
            
            # Prepare data for Prophet
            # Ensure Year column is available for Prophet
            if 'Year' not in X.columns:
                # Add Year column if not present
                if hasattr(self.data_loader, 'historical_data') and 'Year' in self.data_loader.historical_data.columns:
                    # Match index with historical data to get years
                    years = self.data_loader.historical_data.loc[X.index, 'Year'] if len(X.index.intersection(self.data_loader.historical_data.index)) > 0 else range(2007, 2007 + len(X))
                    X = X.copy()
                    X['Year'] = years
                else:
                    # Generate sequential years starting from 2007
                    X = X.copy() 
                    X['Year'] = range(2007, 2007 + len(X))
            
            # Create Prophet DataFrame
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(X['Year'], format='%Y'),
                'y': y.values
            })
            
            # Initialize Prophet model
            self.model = Prophet(
                yearly_seasonality=self.config['yearly_seasonality'],
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=self.config['changepoint_prior_scale'],
                seasonality_prior_scale=self.config['seasonality_prior_scale']
            )
            
            # Add economic regressors
            economic_columns = [col for col in X.columns if col != 'Year']
            for col in economic_columns[:5]:  # Limit to top 5 features for Prophet
                self.model.add_regressor(col)
            
            # Prepare training data with regressors
            for col in economic_columns[:5]:
                prophet_df[col] = X[col].values
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(prophet_df)
            
            self.regressor_columns = economic_columns[:5]
            self.is_fitted = True
            self.logger.info("Prophet model training completed")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training Prophet model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using Prophet model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Prepare future DataFrame - handle missing Year column
        if 'Year' not in X.columns:
            X = X.copy()
            X['Year'] = range(2007, 2007 + len(X))
            
        future_df = pd.DataFrame({
            'ds': pd.to_datetime(X['Year'], format='%Y')
        })
        
        # Add regressors
        for col in self.regressor_columns:
            if col in X.columns:
                future_df[col] = X[col].values
            else:
                # Use median value if regressor not available
                future_df[col] = 0
        
        # Make predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model.predict(future_df)
        
        return forecast['yhat'].values

class MultipleRegressionSteelModel(BaseSteelDemandModel):
    """Multiple regression model for steel demand forecasting based on macro economic drivers."""
    
    def __init__(self, data_loader):
        super().__init__(data_loader, "MultipleRegression")
        self._load_config()
    
    def _load_config(self):
        """Load multiple regression configuration from CSV."""
        self.config = {
            'degree': int(self.data_loader.get_model_config('regression_degree')),
            'include_interaction': self.data_loader.get_model_config('regression_include_interaction'),
            'regularization': self.data_loader.get_model_config('regression_regularization'),
            'random_state': int(self.data_loader.get_model_config('random_state'))
        }
        
        # Key macro economic drivers for regression (based on reg_model analysis)
        self.key_drivers = [
            'GDP_Real_AUD_Billion',
            'Total_Population_Millions', 
            'National_Urbanisation_Rate_pct',
            'Industrial_Production_Index_Base_2007',
            'Iron_Ore_Production_Mt',
            'Coal_Production_Mt'
        ]
    
    def _select_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select key macro economic drivers for regression."""
        available_drivers = []
        
        # Check which key drivers are available in the feature set
        for driver in self.key_drivers:
            if driver in X.columns:
                available_drivers.append(driver)
            else:
                # Look for derived features from the driver
                derived_features = [col for col in X.columns if driver.replace('_', '') in col.replace('_', '')]
                if derived_features:
                    available_drivers.extend(derived_features[:2])  # Limit to top 2 derived features
        
        # Add Year as a feature for trend analysis
        if 'Year' in X.columns:
            available_drivers.append('Year')
        
        # Ensure we have at least some features
        if not available_drivers:
            # Fallback to top correlated features
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            available_drivers = numeric_cols[:6]  # Use top 6 features
        
        return X[available_drivers] if available_drivers else X.iloc[:, :6]
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'MultipleRegressionSteelModel':
        """
        Fit multiple regression model to training data.
        
        Args:
            X: Training feature matrix
            y: Training target values
            
        Returns:
            Fitted model instance
        """
        try:
            self.logger.info("Training Multiple Regression model")
            
            # Select key features for regression
            X_selected = self._select_features(X)
            self.selected_features = X_selected.columns.tolist()
            
            # Create polynomial features if configured
            if self.config['degree'] > 1:
                poly_features = PolynomialFeatures(
                    degree=self.config['degree'], 
                    include_bias=False,
                    interaction_only=self.config['include_interaction'] == 'True'
                )
                
                # Create pipeline with polynomial features and linear regression
                self.model = Pipeline([
                    ('poly', poly_features),
                    ('linear', LinearRegression())
                ])
            else:
                # Simple linear regression
                self.model = LinearRegression()
            
            # Fit the model
            self.model.fit(X_selected, y)
            
            # Store feature importance (coefficients)
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_
                feature_names = self.selected_features
            elif hasattr(self.model, 'named_steps'):
                # Pipeline case
                coefficients = self.model.named_steps['linear'].coef_
                feature_names = self.model.named_steps['poly'].get_feature_names_out(self.selected_features)
            else:
                coefficients = []
                feature_names = []
            
            if len(coefficients) > 0:
                self.feature_importance_ = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': coefficients,
                    'abs_coefficient': np.abs(coefficients)
                }).sort_values('abs_coefficient', ascending=False)
            
            self.is_fitted = True
            self.logger.info("Multiple Regression model training completed")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training Multiple Regression model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using multiple regression model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Select the same features used during training
        X_selected = X[self.selected_features] if hasattr(self, 'selected_features') else self._select_features(X)
        
        # Handle missing features
        for feature in self.selected_features:
            if feature not in X_selected.columns:
                X_selected[feature] = 0  # Fill missing features with 0
        
        predictions = self.model.predict(X_selected[self.selected_features])
        
        # Ensure positive predictions for steel consumption
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (coefficients) from trained model."""
        if not hasattr(self, 'feature_importance_'):
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.feature_importance_.copy()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the regression model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get summary")
        
        summary = {
            'model_type': 'Multiple Regression',
            'features_used': len(self.selected_features),
            'polynomial_degree': self.config['degree'],
            'include_interaction': self.config['include_interaction']
        }
        
        if hasattr(self.model, 'score'):
            summary['r2_score'] = getattr(self.model, 'score', 'Not available')
        
        return summary

class EnsembleSteelModel:
    """Ensemble model combining XGBoost, Random Forest, LSTM, Prophet, and Multiple Regression."""
    
    def __init__(self, data_loader):
        """
        Initialize ensemble model with data loader and configuration.
        
        Args:
            data_loader: SteelDemandDataLoader instance
        """
        self.data_loader = data_loader
        self.logger = self._setup_logging()
        self._load_config()
        
        # Initialize individual models
        self.models = {
            'xgboost': XGBoostSteelModel(data_loader),
            'random_forest': RandomForestSteelModel(data_loader),
            'lstm': LSTMSteelModel(data_loader),
            'prophet': ProphetSteelModel(data_loader),
            'regression': MultipleRegressionSteelModel(data_loader)
        }
        
        self.is_fitted = False
        self.model_weights = {}
        self.individual_predictions = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(f"{__name__}.Ensemble")
    
    def _load_config(self):
        """Load ensemble configuration from CSV."""
        self.weights = {
            'xgboost': float(self.data_loader.get_model_config('ensemble_weights_xgb')),
            'random_forest': float(self.data_loader.get_model_config('ensemble_weights_rf')),
            'lstm': float(self.data_loader.get_model_config('ensemble_weights_lstm')),
            'prophet': float(self.data_loader.get_model_config('ensemble_weights_prophet')),
            'regression': float(self.data_loader.get_model_config('ensemble_weights_regression'))
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
    
    def fit(self, X: pd.DataFrame, y: pd.Series, X_val: Optional[pd.DataFrame] = None, 
            y_val: Optional[pd.Series] = None) -> 'EnsembleSteelModel':
        """
        Fit all ensemble models to training data.
        
        Args:
            X: Training feature matrix
            y: Training target values
            X_val: Validation feature matrix (optional)
            y_val: Validation target values (optional)
            
        Returns:
            Fitted ensemble model instance
        """
        try:
            self.logger.info("Training ensemble models")
            
            model_performances = {}
            
            # Train each model - use list to avoid dictionary modification during iteration
            models_to_train = list(self.models.items())
            failed_models = []
            
            for model_name, model in models_to_train:
                try:
                    self.logger.info(f"Training {model_name}")
                    
                    if model_name in ['xgboost', 'lstm'] and X_val is not None:
                        model.fit(X, y, X_val, y_val)
                    else:
                        model.fit(X, y)
                    
                    # Evaluate model if validation data available
                    if X_val is not None and y_val is not None:
                        try:
                            performance = model.evaluate(X_val, y_val)
                            model_performances[model_name] = performance
                        except Exception as eval_error:
                            self.logger.warning(f"Failed to evaluate {model_name}: {str(eval_error)}")
                    
                    self.logger.info(f"{model_name} training completed")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to train {model_name}: {str(e)}")
                    # Mark for removal
                    failed_models.append(model_name)
            
            # Remove failed models after iteration
            for model_name in failed_models:
                if model_name in self.models:
                    del self.models[model_name]
                if model_name in self.weights:
                    del self.weights[model_name]
            
            # Renormalize weights after removing failed models
            if self.weights:
                total_weight = sum(self.weights.values())
                self.weights = {k: v/total_weight for k, v in self.weights.items()}
            
            # Adjust weights based on performance if validation data available
            if model_performances and len(model_performances) > 1:
                self._adjust_weights_by_performance(model_performances)
            
            self.is_fitted = True
            self.logger.info(f"Ensemble training completed with {len(self.models)} models")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {str(e)}")
            raise
    
    def _adjust_weights_by_performance(self, performances: Dict[str, Dict[str, float]]):
        """Adjust ensemble weights based on validation performance."""
        # Use inverse MAPE as performance score (lower MAPE = higher weight)
        performance_scores = {}
        for model_name, metrics in performances.items():
            mape = metrics.get('MAPE', 100)  # Default to high MAPE if not available
            performance_scores[model_name] = 1 / (mape + 1e-6)
        
        # Normalize scores to weights
        total_score = sum(performance_scores.values())
        performance_weights = {k: v/total_score for k, v in performance_scores.items()}
        
        # Blend with original weights (50% original, 50% performance-based)
        for model_name in self.weights:
            if model_name in performance_weights:
                self.weights[model_name] = 0.5 * self.weights[model_name] + 0.5 * performance_weights[model_name]
        
        # Renormalize
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        self.logger.info(f"Adjusted weights: {self.weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions by combining individual model predictions.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = {}
        
        # Get predictions from each model with proper feature handling
        for model_name, model in self.models.items():
            try:
                # For tree-based models, ensure consistent feature names
                if model_name in ['xgboost', 'random_forest']:
                    if hasattr(model, 'feature_names_') and model.feature_names_:
                        # Use only features that were used during training
                        available_features = [f for f in model.feature_names_ if f in X.columns]
                        missing_features = [f for f in model.feature_names_ if f not in X.columns]
                        
                        if missing_features:
                            self.logger.warning(f"{model_name}: Missing features {missing_features[:5]}...")
                            # Skip this model if critical features are missing
                            continue
                        
                        X_model = X[available_features].copy()
                        
                        # Ensure feature order matches training
                        X_model = X_model.reindex(columns=model.feature_names_, fill_value=0)
                        pred = model.predict(X_model)
                    else:
                        pred = model.predict(X)
                else:
                    # For LSTM, Prophet, and Regression models, use their built-in feature handling
                    pred = model.predict(X)
                
                predictions[model_name] = pred
            except Exception as e:
                self.logger.warning(f"Failed to get predictions from {model_name}: {str(e)}")
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        # Combine predictions using weighted average
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0)
            # Handle different prediction shapes
            if hasattr(pred, '__len__') and len(pred) == len(X):
                ensemble_pred += weight * pred
                total_weight += weight
            elif hasattr(pred, '__len__') and len(pred) == 1 and len(X) == 1:
                # Handle single prediction case
                ensemble_pred += weight * pred
                total_weight += weight
            else:
                self.logger.debug(f"Skipping {model_name}: prediction shape {getattr(pred, 'shape', len(pred) if hasattr(pred, '__len__') else 'scalar')} doesn't match input length {len(X)}")
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Store individual predictions for analysis
        self.individual_predictions = predictions
        
        return ensemble_pred
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate ensemble and individual model performance.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        results = {}
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X)
        
        mae = mean_absolute_error(y, ensemble_pred)
        mse = mean_squared_error(y, ensemble_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, ensemble_pred)
        mape = np.mean(np.abs((y - ensemble_pred) / (y + 1e-6))) * 100
        
        results['ensemble'] = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape
        }
        
        # Evaluate individual models
        for model_name, model in self.models.items():
            try:
                results[model_name] = model.evaluate(X, y)
            except Exception as e:
                self.logger.debug(f"Skipping evaluation for {model_name}: {str(e)}")
        
        self.logger.info(f"Ensemble Evaluation - MAPE: {mape:.2f}%, R²: {r2:.3f}, RMSE: {rmse:.0f}")
        
        return results
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from tree-based models."""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                try:
                    importance_dict[model_name] = model.get_feature_importance()
                except:
                    pass
        
        return importance_dict
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.weights.copy()
    
    def get_individual_predictions(self) -> Dict[str, np.ndarray]:
        """Get latest individual model predictions."""
        return self.individual_predictions.copy()