#!/usr/bin/env python3
"""
ML Algorithms with Regression Model Features

This script applies the 5 ML algorithms (XGBoost, Random Forest, LSTM, Prophet, Multiple Regression)
to the EXACT SAME FEATURES used in the regression model, without any feature engineering.

Goal: Direct comparison of algorithm performance on identical economic drivers.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Time Series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging for the ML comparison."""
    log_file = output_dir / f"ml_regression_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"ML with Regression Features - Log file: {log_file}")
    return logger

class RegressionFeatureMLTrainer:
    """Train ML algorithms using exact regression model features."""
    
    def __init__(self, output_dir: str = "forecasts/ml_regression_features"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.output_dir)
        
        # Load data configuration
        from data.data_loader import SteelDemandDataLoader
        self.data_loader = SteelDemandDataLoader()
        
        # Load steel categories from configuration
        steel_categories = self.data_loader.get_steel_categories()
        econ_indicators = self.data_loader.get_economic_indicators()
        
        # Create category features mapping from configuration
        self.category_features = {}
        feature_columns = econ_indicators['column_name'].tolist()
        
        for _, row in steel_categories.iterrows():
            self.category_features[row['target_column']] = feature_columns
        
        self.results = {}
        self.models = {}
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical and projection data using data loader."""
        self.logger.info("Loading WSA steel and WM macro data...")
        
        # Load all data through data loader
        self.data_loader.load_all_data()
        
        # Get historical data (2004-2023)
        historical_data = self.data_loader.get_historical_data()
        
        # Get projection data (2024-2050) 
        projection_data = self.data_loader.get_projection_data()
        
        self.logger.info(f"Historical data shape: {historical_data.shape}")
        self.logger.info(f"Projection data shape: {projection_data.shape}")
        self.logger.info(f"Steel categories: {list(self.category_features.keys())}")
        
        return historical_data, projection_data
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """Train XGBoost model."""
        self.logger.info(f"Training XGBoost for {category}")
        
        # Simple XGBoost configuration
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
            'feature_importance': feature_importance
        }
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """Train Random Forest model."""
        self.logger.info(f"Training Random Forest for {category}")
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
            'feature_importance': feature_importance
        }
    
    def train_lstm(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """Train LSTM model (if TensorFlow available)."""
        if not TENSORFLOW_AVAILABLE:
            self.logger.warning(f"TensorFlow not available, skipping LSTM for {category}")
            return None
            
        self.logger.info(f"Training LSTM for {category}")
        
        try:
            # Prepare data for LSTM (simple approach - no sequences, just treat as regular regression)
            X_scaled = (X - X.mean()) / X.std()
            y_scaled = (y - y.mean()) / y.std()
            
            # Simple LSTM model
            model = keras.Sequential([
                layers.Dense(8, activation='relu', input_shape=(X.shape[1],)),
                layers.Dense(4, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_scaled.values, y_scaled.values,
                epochs=50,
                batch_size=min(8, len(X)),
                verbose=0
            )
            
            # Make predictions
            predictions_scaled = model.predict(X_scaled.values, verbose=0).flatten()
            predictions = predictions_scaled * y.std() + y.mean()
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
            
            return {
                'model': model,
                'predictions': predictions,
                'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
                'scaler_X_mean': X.mean(),
                'scaler_X_std': X.std(),
                'scaler_y_mean': y.mean(),
                'scaler_y_std': y.std()
            }
            
        except Exception as e:
            self.logger.error(f"LSTM training failed for {category}: {str(e)}")
            return None
    
    def train_prophet(self, X: pd.DataFrame, y: pd.Series, historical_data: pd.DataFrame, 
                     category: str) -> Dict[str, Any]:
        """Train Prophet model (if Prophet available)."""
        if not PROPHET_AVAILABLE:
            self.logger.warning(f"Prophet not available, skipping Prophet for {category}")
            return None
            
        self.logger.info(f"Training Prophet for {category}")
        
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(historical_data['Year'], format='%Y'),
                'y': y.values
            })
            
            # Add regressors
            for feature in X.columns:
                prophet_df[feature] = X[feature].values
            
            # Initialize Prophet model
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Add regressors
            for feature in X.columns:
                model.add_regressor(feature)
            
            # Fit model
            model.fit(prophet_df)
            
            # Make predictions
            forecast = model.predict(prophet_df)
            predictions = forecast['yhat'].values
            
            # Calculate metrics
            mae = mean_absolute_error(y, predictions)
            mse = mean_squared_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)
            mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
            
            return {
                'model': model,
                'predictions': predictions,
                'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
            }
            
        except Exception as e:
            self.logger.error(f"Prophet training failed for {category}: {str(e)}")
            return None
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series, category: str) -> Dict[str, Any]:
        """Train Linear Regression (baseline)."""
        self.logger.info(f"Training Linear Regression for {category}")
        
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, predictions)
        mape = np.mean(np.abs((y - predictions) / (y + 1e-6))) * 100
        
        # Feature coefficients
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'coefficient': model.coef_,
            'abs_coefficient': np.abs(model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return {
            'model': model,
            'predictions': predictions,
            'metrics': {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape},
            'feature_importance': feature_importance,
            'intercept': model.intercept_
        }
    
    def train_all_algorithms(self, historical_data: pd.DataFrame) -> None:
        """Train all algorithms for all steel categories."""
        self.logger.info("Training all ML algorithms with regression features...")
        
        for category, features in self.category_features.items():
            self.logger.info(f"\\n{'='*60}")
            self.logger.info(f"Training models for {category}")
            self.logger.info(f"Features: {features}")
            
            # Prepare data
            X = historical_data[features].copy()
            y = historical_data[category].copy()
            
            # Remove any NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            self.logger.info(f"Training samples: {len(X)}")
            self.logger.info(f"Feature matrix shape: {X.shape}")
            
            # Store results for this category
            self.results[category] = {}
            self.models[category] = {}
            
            # Train each algorithm
            algorithms = [
                ('XGBoost', self.train_xgboost),
                ('RandomForest', self.train_random_forest),
                ('LSTM', self.train_lstm),
                ('Prophet', lambda x, y, cat: self.train_prophet(x, y, historical_data, cat)),
                ('LinearRegression', self.train_linear_regression)
            ]
            
            for algo_name, train_func in algorithms:
                try:
                    if algo_name == 'Prophet':
                        result = train_func(X, y, category)
                    else:
                        result = train_func(X, y, category)
                    
                    if result is not None:
                        self.results[category][algo_name] = result
                        self.models[category][algo_name] = result['model']
                        
                        metrics = result['metrics']
                        self.logger.info(f"{algo_name} - MAPE: {metrics['MAPE']:.2f}%, R²: {metrics['R2']:.3f}")
                    else:
                        self.logger.warning(f"{algo_name} training failed for {category}")
                        
                except Exception as e:
                    self.logger.error(f"Error training {algo_name} for {category}: {str(e)}")
    
    def generate_forecasts(self, projection_data: pd.DataFrame) -> None:
        """Generate forecasts using trained models."""
        self.logger.info("Generating forecasts for 2025-2050...")
        
        all_forecasts = {'Year': projection_data['Year'].values}
        
        for category, features in self.category_features.items():
            self.logger.info(f"Generating forecasts for {category}")
            
            # Prepare projection features
            X_proj = projection_data[features].copy()
            
            category_forecasts = {}
            
            for algo_name, result in self.results[category].items():
                try:
                    model = result['model']
                    
                    if algo_name == 'LSTM' and TENSORFLOW_AVAILABLE:
                        # Apply scaling for LSTM
                        X_scaled = (X_proj - result['scaler_X_mean']) / result['scaler_X_std']
                        predictions_scaled = model.predict(X_scaled.values, verbose=0).flatten()
                        predictions = predictions_scaled * result['scaler_y_std'] + result['scaler_y_mean']
                    
                    elif algo_name == 'Prophet' and PROPHET_AVAILABLE:
                        # Prepare Prophet future dataframe
                        future_df = pd.DataFrame({
                            'ds': pd.to_datetime(projection_data['Year'], format='%Y')
                        })
                        for feature in features:
                            future_df[feature] = X_proj[feature].values
                        
                        forecast = model.predict(future_df)
                        predictions = forecast['yhat'].values
                    
                    else:
                        # Standard scikit-learn models
                        predictions = model.predict(X_proj)
                    
                    # Ensure positive predictions
                    predictions = np.maximum(predictions, 0)
                    
                    category_forecasts[f'{category}_{algo_name}'] = predictions
                    
                except Exception as e:
                    self.logger.error(f"Error generating forecast for {category} with {algo_name}: {str(e)}")
            
            # Add to all forecasts
            all_forecasts.update(category_forecasts)
        
        # Save forecasts
        forecast_df = pd.DataFrame(all_forecasts)
        forecast_file = self.output_dir / 'ML_Algorithm_Forecasts_2025-2050.csv'
        forecast_df.to_csv(forecast_file, index=False)
        self.logger.info(f"Forecasts saved to {forecast_file}")
        
        return forecast_df
    
    def create_ensemble_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble forecast combining XGBoost, Random Forest, Prophet, and Linear Regression."""
        self.logger.info("Creating ensemble forecast...")
        
        # Define ensemble algorithms (excluding LSTM due to poor performance)
        ensemble_algorithms = ['XGBoost', 'RandomForest', 'Prophet', 'LinearRegression']
        
        # Steel categories
        categories = [
            'Hot_Rolled_Structural_Steel_tonnes',
            'Rail_Products_tonnes', 
            'Steel_Billets_tonnes',
            'Steel_Slabs_tonnes'
        ]
        
        ensemble_forecasts = {}
        
        # Add Year column
        ensemble_forecasts['Year'] = forecast_df['Year']
        
        # Create ensemble for each category
        for category in categories:
            category_ensemble = []
            
            # Get predictions from each algorithm
            category_predictions = []
            for algo in ensemble_algorithms:
                col_name = f'{category}_{algo}'
                if col_name in forecast_df.columns:
                    category_predictions.append(forecast_df[col_name])
            
            if category_predictions:
                # Simple average ensemble
                ensemble_prediction = np.mean(category_predictions, axis=0)
                ensemble_forecasts[f'{category}_Ensemble'] = ensemble_prediction
                
                self.logger.info(f"Created ensemble for {category} using {len(category_predictions)} algorithms")
            else:
                self.logger.warning(f"No valid predictions found for {category}")
        
        # Create ensemble dataframe
        ensemble_df = pd.DataFrame(ensemble_forecasts)
        
        # Save ensemble forecasts
        ensemble_file = self.output_dir / 'Ensemble_Forecasts_2025-2050.csv'
        ensemble_df.to_csv(ensemble_file, index=False)
        self.logger.info(f"Ensemble forecasts saved to {ensemble_file}")
        
        # Calculate total ensemble forecast by year
        total_ensemble = []
        for idx, row in ensemble_df.iterrows():
            if idx == 0:  # Skip header row
                continue
            year_total = 0
            for category in categories:
                col_name = f'{category}_Ensemble'
                if col_name in ensemble_df.columns:
                    year_total += row[col_name]
            total_ensemble.append(year_total)
        
        # Add total column
        ensemble_df['Total_Steel_Consumption_Ensemble'] = [0] + total_ensemble
        
        # Log key ensemble results
        if len(total_ensemble) > 0:
            ensemble_2025 = total_ensemble[0] if len(total_ensemble) > 0 else 0
            ensemble_2050 = total_ensemble[-1] if len(total_ensemble) >= 26 else 0
            
            self.logger.info(f"Ensemble Forecast - 2025: {ensemble_2025:,.0f} tonnes")
            self.logger.info(f"Ensemble Forecast - 2050: {ensemble_2050:,.0f} tonnes")
            
            if ensemble_2025 > 0:
                growth_rate = ((ensemble_2050 / ensemble_2025) ** (1/25) - 1) * 100
                self.logger.info(f"Ensemble Average Annual Growth Rate: {growth_rate:.2f}%")
        
        return ensemble_df
    
    def save_results(self) -> None:
        """Save all results and performance comparison."""
        self.logger.info("Saving results and performance comparison...")
        
        # Performance comparison
        performance_data = []
        
        for category, algorithms in self.results.items():
            for algo_name, result in algorithms.items():
                metrics = result['metrics']
                performance_data.append({
                    'Category': category,
                    'Algorithm': algo_name,
                    'MAPE': metrics['MAPE'],
                    'R2': metrics['R2'], 
                    'RMSE': metrics['RMSE'],
                    'MAE': metrics['MAE'],
                    'Features_Used': len(self.category_features[category])
                })
        
        performance_df = pd.DataFrame(performance_data)
        performance_file = self.output_dir / 'Algorithm_Performance_Comparison.csv'
        performance_df.to_csv(performance_file, index=False)
        
        # Feature importance (for tree-based models)
        for category, algorithms in self.results.items():
            for algo_name, result in algorithms.items():
                if 'feature_importance' in result:
                    importance_file = self.output_dir / f'{category}_{algo_name}_feature_importance.csv'
                    result['feature_importance'].to_csv(importance_file, index=False)
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for category, models in self.models.items():
            for algo_name, model in models.items():
                if algo_name not in ['LSTM', 'Prophet']:  # Skip TF and Prophet models for joblib
                    model_file = models_dir / f'{category}_{algo_name}_model.joblib'
                    joblib.dump(model, model_file)
        
        self.logger.info(f"All results saved to {self.output_dir}")

def main():
    """Main execution function."""
    print("🔬 ML Algorithms with Regression Model Features")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RegressionFeatureMLTrainer()
    
    # Load data
    historical_data, projection_data = trainer.load_data()
    
    # Train all algorithms
    trainer.train_all_algorithms(historical_data)
    
    # Generate forecasts
    forecast_df = trainer.generate_forecasts(projection_data)
    
    # Create ensemble forecast
    ensemble_df = trainer.create_ensemble_forecast(forecast_df)
    
    # Save results
    trainer.save_results()
    
    print("\\n" + "=" * 60)
    print("✅ ML ALGORITHM COMPARISON COMPLETED")
    print("=" * 60)
    print(f"📊 Results saved to: {trainer.output_dir}")
    print("\\n📋 Generated files:")
    print("  • Algorithm_Performance_Comparison.csv")
    print("  • ML_Algorithm_Forecasts_2025-2050.csv")
    print("  • Feature importance files by algorithm")
    print("  • Trained models (models/ directory)")
    
    return trainer.results

if __name__ == "__main__":
    results = main()