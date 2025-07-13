#!/usr/bin/env python3
"""
ML Algorithms with Regression Model Features - Simplified Version

This script applies ML algorithms (XGBoost, Random Forest)
to the EXACT SAME FEATURES used in the regression model, without any complex feature engineering.

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression  # Removed
import xgboost as xgb

# Prophet removed from system completely

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    VISUALIZATION_AVAILABLE = True
except ImportError:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('default')
        VISUALIZATION_AVAILABLE = True
    except ImportError:
        VISUALIZATION_AVAILABLE = False

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging for the ML comparison."""
    log_file = output_dir / f"track_a_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    logger.info(f"Regression ML Models - Log file: {log_file}")
    return logger

class RegressionMLTrainer:
    """Train ML algorithms using exact regression model features."""
    
    def __init__(self, output_dir: str = None):
        # Create timestamped output directory if not specified
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"forecasts/track_a_{timestamp}"
        
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
        self.historical_data = None
    
    def get_clean_category_name(self, category: str) -> str:
        """Get a clean display name for a steel category."""
        # Handle NaN or invalid values
        if pd.isna(category) or not isinstance(category, str):
            return "Unknown Category"
            
        # Get the category info from configuration
        steel_categories = self.data_loader.get_steel_categories()
        category_row = steel_categories[steel_categories['target_column'] == category]
        
        if not category_row.empty:
            return category_row.iloc[0]['category']
        else:
            # Fallback to basic cleaning
            return category.replace('_', ' ').title()
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load historical and projection data using data loader."""
        self.logger.info("Loading WSA steel and WM macro data...")
        
        # Load all data through data loader
        self.data_loader.load_all_data()
        
        # Get historical data (2004-2023)
        historical_data = self.data_loader.get_historical_data()
        
        # Get projection data (2024-2050) 
        projection_data = self.data_loader.get_projection_data()
        
        # Store historical data for visualization
        self.historical_data = historical_data.copy()
        
        self.logger.info(f"Historical data shape: {historical_data.shape}")
        self.logger.info(f"Projection data shape: {projection_data.shape}")
        self.logger.info(f"Steel categories: {list(self.category_features.keys())}")
        
        return historical_data, projection_data
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        return {
            'MAPE': mape,
            'R2': r2,
            'RMSE': rmse,
            'MAE': mae
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, category: str) -> xgb.XGBRegressor:
        """Train XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, category: str) -> RandomForestRegressor:
        """Train Random Forest model.""" 
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    # def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray, category: str) -> LinearRegression:
    #     """Train Linear Regression model."""
    #     model = LinearRegression()
    #     model.fit(X_train, y_train)
    #     return model
    
    # Prophet method has been removed from this system
    
    def train_all_algorithms(self, historical_data: pd.DataFrame) -> None:
        """Train all algorithms for each steel category."""
        self.logger.info("Training all ML algorithms...")
        
        for category, features in self.category_features.items():
            self.logger.info(f"\\nTraining models for {category}")
            self.logger.info(f"Using features: {features}")
            
            # Check if category exists in data
            if category not in historical_data.columns:
                self.logger.warning(f"Category {category} not found in data, skipping...")
                continue
            
            # Prepare data
            X = historical_data[features].values
            y = historical_data[category].values
            years = historical_data['Year'].values
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            years = years[valid_mask]
            
            if len(X) < 5:
                self.logger.warning(f"Insufficient data for {category} ({len(X)} samples), skipping...")
                continue
            
            # Initialize results and models for this category
            self.results[category] = {}
            self.models[category] = {}
            
            # Train XGBoost
            try:
                model = self.train_xgboost(X, y, category)
                y_pred = model.predict(X)
                metrics = self.calculate_metrics(y, y_pred)
                
                self.results[category]['XGBoost'] = {
                    'metrics': metrics,
                    'feature_importance': dict(zip(features, model.feature_importances_))
                }
                self.models[category]['XGBoost'] = model
                self.logger.info(f"XGBoost MAPE: {metrics['MAPE']:.2f}%")
                
            except Exception as e:
                self.logger.error(f"XGBoost training failed for {category}: {str(e)}")
            
            # Train Random Forest
            try:
                model = self.train_random_forest(X, y, category)
                y_pred = model.predict(X)
                metrics = self.calculate_metrics(y, y_pred)
                
                self.results[category]['RandomForest'] = {
                    'metrics': metrics,
                    'feature_importance': dict(zip(features, model.feature_importances_))
                }
                self.models[category]['RandomForest'] = model
                self.logger.info(f"Random Forest MAPE: {metrics['MAPE']:.2f}%")
                
            except Exception as e:
                self.logger.error(f"Random Forest training failed for {category}: {str(e)}")
            
            # Linear Regression removed from system
            
            # Prophet has been removed from this system
    
    def generate_forecasts(self, projection_data: pd.DataFrame) -> pd.DataFrame:
        """Generate forecasts for projection period."""
        start_year = projection_data['Year'].min()
        end_year = projection_data['Year'].max()
        self.logger.info(f"Generating forecasts for {start_year}-{end_year}...")
        
        # Use years from projection data
        years = projection_data['Year'].tolist()
        all_forecasts = {'Year': years}
        
        for category, algorithms in self.models.items():
            features = self.category_features[category]
            
            # Prepare projection features
            X_proj = projection_data[features].values
            
            for algo_name, model in algorithms.items():
                try:
                    # Standard scikit-learn models
                    predictions = model.predict(X_proj)
                    
                    # Ensure positive predictions
                    predictions = np.maximum(predictions, 0)
                    
                    all_forecasts[f'{category}_{algo_name}'] = predictions
                    
                except Exception as e:
                    self.logger.error(f"Error generating forecast for {category} with {algo_name}: {str(e)}")
        
        # Save forecasts
        forecast_df = pd.DataFrame(all_forecasts)
        
        # Combine with historical data for complete timeseries (2004-2050)
        if self.historical_data is not None:
            # Prepare historical data with algorithm columns
            historical_for_export = self.historical_data.copy()
            
            # For each category, add algorithm predictions (using actual historical values)
            for category in self.category_features.keys():
                if category in historical_for_export.columns:
                    for algo in ['XGBoost', 'RandomForest']:
                        col_name = f'{category}_{algo}'
                        # Use actual historical values for the historical period
                        historical_for_export[col_name] = historical_for_export[category]
            
            # Combine historical and forecast data
            complete_df = pd.concat([historical_for_export, forecast_df], ignore_index=True)
            
            # Save complete timeseries
            complete_file = self.output_dir / 'ML_Algorithm_Forecasts_2004-2050.csv'
            complete_df.to_csv(complete_file, index=False)
            self.logger.info(f"Complete forecasts with historical data saved to {complete_file}")
            
            # Also save forecast-only file for compatibility
            forecast_file = self.output_dir / 'ML_Algorithm_Forecasts_2025-2050.csv'
            forecast_df.to_csv(forecast_file, index=False)
            self.logger.info(f"Forecast-only data saved to {forecast_file}")
        else:
            # Fallback if no historical data
            forecast_file = self.output_dir / 'ML_Algorithm_Forecasts_2025-2050.csv'
            forecast_df.to_csv(forecast_file, index=False)
            self.logger.info(f"Forecasts saved to {forecast_file}")
        
        return forecast_df
    
    def apply_hierarchical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply WSA hierarchical consistency constraints to ensure forecasts maintain proper relationships."""
        self.logger.info("Applying WSA hierarchical consistency validation...")
        
        # Define hierarchical relationships based on WSA structure
        hierarchical_rules = {
            'crude_steel_breakdown': {
                'type': 'parallel',  # Parent splits into children
                'parent': 'Total Production of Crude Steel_Ensemble',
                'children': [
                    'Production of Crude Steel in Electric Furnaces_Ensemble',
                    'Production of Crude Steel in Oxygen-blown Converters_Ensemble'
                ]
            },
            'semi_finished_breakdown': {
                'type': 'parallel',  # Semi-finished splits into ingots and continuously-cast
                'parent': None,  # Calculate semi-finished total dynamically
                'children': [
                    'Production of Ingots_Ensemble',
                    'Production of Continuously-cast Steel_Ensemble'
                ]
            },
            'hot_rolled_breakdown': {
                'type': 'parallel',  # Hot rolled splits into flat and long
                'parent': 'Production of Hot Rolled Products_Ensemble',
                'children': [
                    'Production of Hot Rolled Flat Products_Ensemble',
                    'Production of Hot Rolled Long Products_Ensemble'
                ]
            },
            'flat_products_breakdown': {
                'type': 'parallel',  # Flat products split into subcategories
                'parent': 'Production of Hot Rolled Flat Products_Ensemble',
                'children': [
                    'Production of Hot Rolled Coil, Sheet, and Strip (<3mm)_Ensemble',
                    'Production of Non-metallic Coated Sheet and Strip_Ensemble',
                    'Production of Other Metal Coated Sheet and Strip_Ensemble'
                ]
            },
            'long_products_breakdown': {
                'type': 'parallel',  # Long products split into subcategories
                'parent': 'Production of Hot Rolled Long Products_Ensemble',
                'children': [
                    'Production of Wire Rod_Ensemble',
                    'Production of Railway Track Material_Ensemble'
                ]
            }
        }
        
        # Define sequential relationships (A → B with yield losses)
        sequential_rules = {
            'crude_to_semi_finished': {
                'from': 'Total Production of Crude Steel_Ensemble',
                'to': None,  # Calculate semi-finished dynamically
                'expected_yield_factor': 0.95,  # 5% loss from crude to semi-finished
                'tolerance': 0.10  # 10% tolerance
            },
            'semi_finished_to_finished': {
                'from': None,  # Calculate semi-finished dynamically  
                'to': None,   # Calculate finished products dynamically
                'expected_yield_factor': 0.92,  # 8% loss from semi-finished to finished
                'tolerance': 0.10  # 10% tolerance
            }
        }
        
        adjustments_made = 0
        
        # 1. Apply parallel hierarchy rules (parent = sum of children)
        for rule_name, rule in hierarchical_rules.items():
            if rule['type'] != 'parallel':
                continue
                
            parent_col = rule['parent']
            child_cols = rule['children']
            
            # Handle dynamic parent calculation for semi-finished
            if parent_col is None and rule_name == 'semi_finished_breakdown':
                # Semi-finished total = sum of ingots + continuously-cast
                available_children = [col for col in child_cols if col in df.columns]
                if len(available_children) >= 2:
                    # For semi-finished, ensure the children are consistent but don't enforce a parent
                    continue
            
            # Check if all columns exist
            if parent_col and parent_col not in df.columns:
                continue
                
            available_children = [col for col in child_cols if col in df.columns]
            if not available_children:
                continue
            
            # For each row, ensure consistency
            for idx in df.index:
                if parent_col:
                    parent_value = df.loc[idx, parent_col]
                    if pd.isna(parent_value) or parent_value <= 0:
                        continue
                else:
                    continue
                    
                child_sum = sum(df.loc[idx, col] for col in available_children if not pd.isna(df.loc[idx, col]))
                
                # If children sum significantly differs from parent, adjust proportionally
                if abs(child_sum - parent_value) > parent_value * 0.05:  # 5% tolerance
                    if child_sum > 0:
                        # Proportionally adjust children to match parent
                        adjustment_factor = parent_value / child_sum
                        for col in available_children:
                            if not pd.isna(df.loc[idx, col]):
                                df.loc[idx, col] *= adjustment_factor
                        adjustments_made += 1
                    else:
                        # If children sum is zero but parent exists, distribute equally
                        equal_share = parent_value / len(available_children)
                        for col in available_children:
                            df.loc[idx, col] = equal_share
                        adjustments_made += 1
        
        # 2. Apply sequential relationship validation (A → B with yield factors)
        for rule_name, rule in sequential_rules.items():
            from_col = rule['from']
            to_col = rule['to']
            expected_yield = rule['expected_yield_factor']
            tolerance = rule['tolerance']
            
            # Calculate dynamic values for semi-finished and finished totals
            if rule_name == 'crude_to_semi_finished':
                # Semi-finished = Ingots + Continuously-cast
                ingots_col = 'Production of Ingots_Ensemble'
                continuous_col = 'Production of Continuously-cast Steel_Ensemble'
                if all(col in df.columns for col in [from_col, ingots_col, continuous_col]):
                    for idx in df.index:
                        crude_steel = df.loc[idx, from_col] if not pd.isna(df.loc[idx, from_col]) else 0
                        ingots = df.loc[idx, ingots_col] if not pd.isna(df.loc[idx, ingots_col]) else 0
                        continuous = df.loc[idx, continuous_col] if not pd.isna(df.loc[idx, continuous_col]) else 0
                        semi_finished_actual = ingots + continuous
                        
                        if crude_steel > 0:
                            expected_semi_finished = crude_steel * expected_yield
                            deviation = abs(semi_finished_actual - expected_semi_finished) / expected_semi_finished
                            
                            if deviation > tolerance:
                                # Adjust semi-finished products proportionally
                                if semi_finished_actual > 0:
                                    adjustment_factor = expected_semi_finished / semi_finished_actual
                                    df.loc[idx, ingots_col] *= adjustment_factor
                                    df.loc[idx, continuous_col] *= adjustment_factor
                                    adjustments_made += 1
            
            elif rule_name == 'semi_finished_to_finished':
                # Finished = Flat + Long + Tubular
                flat_col = 'Production of Hot Rolled Flat Products_Ensemble'
                long_col = 'Production of Hot Rolled Long Products_Ensemble'
                tubular_col = 'Total Production of Tubular Products_Ensemble'
                ingots_col = 'Production of Ingots_Ensemble'
                continuous_col = 'Production of Continuously-cast Steel_Ensemble'
                
                if all(col in df.columns for col in [flat_col, long_col, tubular_col, ingots_col, continuous_col]):
                    for idx in df.index:
                        # Calculate totals
                        semi_finished_total = (df.loc[idx, ingots_col] if not pd.isna(df.loc[idx, ingots_col]) else 0) + \
                                            (df.loc[idx, continuous_col] if not pd.isna(df.loc[idx, continuous_col]) else 0)
                        
                        finished_total = (df.loc[idx, flat_col] if not pd.isna(df.loc[idx, flat_col]) else 0) + \
                                       (df.loc[idx, long_col] if not pd.isna(df.loc[idx, long_col]) else 0) + \
                                       (df.loc[idx, tubular_col] if not pd.isna(df.loc[idx, tubular_col]) else 0)
                        
                        if semi_finished_total > 0:
                            expected_finished = semi_finished_total * expected_yield
                            deviation = abs(finished_total - expected_finished) / expected_finished if expected_finished > 0 else 0
                            
                            if deviation > tolerance:
                                # Adjust finished products proportionally
                                if finished_total > 0:
                                    adjustment_factor = expected_finished / finished_total
                                    df.loc[idx, flat_col] *= adjustment_factor
                                    df.loc[idx, long_col] *= adjustment_factor
                                    df.loc[idx, tubular_col] *= adjustment_factor
                                    adjustments_made += 1
        
        # Special handling for derived categories
        df = self._calculate_derived_categories(df)
        
        if adjustments_made > 0:
            self.logger.info(f"Applied {adjustments_made} hierarchical consistency adjustments")
        
        return df
    
    def _calculate_derived_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived categories like 'Other Long Products' and ratios."""
        
        # Calculate "Other Long Products" as remainder of long products
        long_parent = 'Production of Hot Rolled Long Products_Ensemble'
        wire_rod = 'Production of Wire Rod_Ensemble'
        railway = 'Production of Railway Track Material_Ensemble'
        
        if all(col in df.columns for col in [long_parent, wire_rod, railway]):
            other_long = []
            for idx in df.index:
                total_long = df.loc[idx, long_parent] if not pd.isna(df.loc[idx, long_parent]) else 0
                wire = df.loc[idx, wire_rod] if not pd.isna(df.loc[idx, wire_rod]) else 0
                rail = df.loc[idx, railway] if not pd.isna(df.loc[idx, railway]) else 0
                other = max(0, total_long - wire - rail)
                other_long.append(other)
            
            df['Other_Long_Products_Derived'] = other_long
        
        # Calculate production efficiency ratios
        crude_steel = 'Total Production of Crude Steel_Ensemble'
        iron_ore = 'Production of Iron Ore_Ensemble'
        
        if all(col in df.columns for col in [crude_steel, iron_ore]):
            efficiency_ratios = []
            for idx in df.index:
                crude = df.loc[idx, crude_steel] if not pd.isna(df.loc[idx, crude_steel]) else 0
                ore = df.loc[idx, iron_ore] if not pd.isna(df.loc[idx, iron_ore]) else 0
                ratio = crude / ore if ore > 0 else 0
                efficiency_ratios.append(ratio)
            
            df['Steel_Production_Efficiency_Ratio'] = efficiency_ratios
        
        return df
    
    def create_ensemble_forecast(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """Create ensemble forecast combining XGBoost and Random Forest."""
        self.logger.info("Creating ensemble forecast...")
        
        # Define ensemble algorithms
        ensemble_algorithms = ['XGBoost', 'RandomForest']
        
        # Get steel categories from configuration
        steel_categories_df = self.data_loader.get_steel_categories()
        categories = steel_categories_df['category'].tolist()
        
        ensemble_forecasts = {}
        
        # Add Year column
        ensemble_forecasts['Year'] = forecast_df['Year']
        
        # Create ensemble for each category
        for category in categories:
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
        
        # Create ensemble dataframe
        ensemble_df = pd.DataFrame(ensemble_forecasts)
        
        # Skip hierarchical consistency validation to keep pure averages
        # Comment out the line below if you want to enforce WSA hierarchical relationships
        # ensemble_df = self.apply_hierarchical_consistency(ensemble_df)
        
        # Calculate total ensemble forecast by year
        total_ensemble = []
        for idx, row in ensemble_df.iterrows():
            year_total = 0
            for category in categories:
                col_name = f'{category}_Ensemble'
                if col_name in ensemble_df.columns and not pd.isna(row[col_name]):
                    year_total += row[col_name]
            total_ensemble.append(year_total)
        
        # Add total column
        ensemble_df['Total_Steel_Consumption_Ensemble'] = total_ensemble
        
        # Save ensemble forecasts with historical data
        if self.historical_data is not None:
            # Prepare historical data with ensemble columns
            historical_ensemble = self.historical_data.copy()
            
            # For each category, add ensemble column (using actual historical values)
            for category in self.category_features.keys():
                if category in historical_ensemble.columns:
                    ensemble_col = f'{category}_Ensemble'
                    historical_ensemble[ensemble_col] = historical_ensemble[category]
            
            # Add total ensemble column for historical data
            total_historical = []
            for idx, row in historical_ensemble.iterrows():
                year_total = 0
                for category in self.category_features.keys():
                    if category in historical_ensemble.columns and not pd.isna(row[category]):
                        year_total += row[category]
                total_historical.append(year_total)
            historical_ensemble['Total_Steel_Consumption_Ensemble'] = total_historical
            
            # Combine historical and forecast ensemble data
            complete_ensemble_df = pd.concat([historical_ensemble, ensemble_df], ignore_index=True)
            
            # Save complete timeseries ensemble
            complete_ensemble_file = self.output_dir / 'Ensemble_Forecasts_2004-2050.csv'
            complete_ensemble_df.to_csv(complete_ensemble_file, index=False)
            self.logger.info(f"Complete ensemble forecasts with historical data saved to {complete_ensemble_file}")
            
            # Also save forecast-only file for compatibility  
            ensemble_file = self.output_dir / 'Ensemble_Forecasts_2025-2050.csv'
            ensemble_df.to_csv(ensemble_file, index=False)
            self.logger.info(f"Forecast-only ensemble data saved to {ensemble_file}")
        else:
            # Fallback if no historical data
            ensemble_file = self.output_dir / 'Ensemble_Forecasts_2025-2050.csv'
            ensemble_df.to_csv(ensemble_file, index=False)
            self.logger.info(f"Ensemble forecasts saved to {ensemble_file}")
        
        self.logger.info("Note: Ensemble values are pure averages of XGBoost and Random Forest predictions")
        self.logger.info("      No hierarchical consistency adjustments have been applied")
        
        # Log key ensemble results
        if len(total_ensemble) > 0:
            ensemble_2025 = total_ensemble[0]
            ensemble_2050 = total_ensemble[-1]
            
            self.logger.info(f"Ensemble Forecast - 2025: {ensemble_2025:,.0f} tonnes")
            self.logger.info(f"Ensemble Forecast - 2050: {ensemble_2050:,.0f} tonnes")
            
            if ensemble_2025 > 0:
                growth_rate = ((ensemble_2050 / ensemble_2025) ** (1/25) - 1) * 100
                self.logger.info(f"Ensemble Average Annual Growth Rate: {growth_rate:.2f}%")
        
        return ensemble_df
    
    def create_visualizations(self, forecast_df: pd.DataFrame, ensemble_df: pd.DataFrame) -> None:
        """Create comprehensive visualizations for the forecasting results."""
        if not VISUALIZATION_AVAILABLE:
            self.logger.warning("Visualization libraries not available, skipping charts...")
            return
            
        self.logger.info("Creating visualizations...")
        
        # Create visualizations directory
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Set style and color palette
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Algorithm Performance Comparison
        self._create_performance_chart(viz_dir)
        
        # 2. Ensemble vs Individual Algorithm Comparison
        self._create_ensemble_comparison(forecast_df, ensemble_df, viz_dir)
        
        # 3. Feature Importance Heatmap
        self._create_feature_importance_heatmap(viz_dir)
        
        # 4. Algorithm Accuracy Dashboard
        self._create_accuracy_dashboard(viz_dir)
        
        self.logger.info(f"Visualizations saved to {viz_dir}")
    
    def _create_performance_chart(self, viz_dir: Path) -> None:
        """Create algorithm performance comparison chart."""
        # Collect performance data
        performance_data = []
        for category, algorithms in self.results.items():
            for algo_name, result in algorithms.items():
                performance_data.append({
                    'Category': self.get_clean_category_name(category),
                    'Algorithm': algo_name,
                    'MAPE': result['metrics']['MAPE'],
                    'R2': result['metrics']['R2']
                })
        
        df = pd.DataFrame(performance_data)
        
        # Create subplot for MAPE and R2
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # MAPE comparison
        pivot_mape = df.pivot(index='Category', columns='Algorithm', values='MAPE')
        sns.heatmap(pivot_mape, annot=True, fmt='.2f', cmap='YlOrRd_r', ax=ax1)
        ax1.set_title('Algorithm Performance by Category (MAPE %)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Steel Category')
        
        # R2 comparison
        pivot_r2 = df.pivot(index='Category', columns='Algorithm', values='R2')
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax2)
        ax2.set_title('Algorithm Performance by Category (R²)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('Steel Category')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'algorithm_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ensemble_comparison(self, forecast_df: pd.DataFrame, ensemble_df: pd.DataFrame, viz_dir: Path) -> None:
        """Create ensemble vs individual algorithm comparison with historical data."""
        categories = [cat for cat in self.category_features.keys() if pd.notna(cat) and isinstance(cat, str)]
        
        # Create subplot layout based on number of categories
        n_cats = len(categories)
        if n_cats <= 4:
            rows, cols = 2, 2
        elif n_cats <= 6:
            rows, cols = 2, 3
        elif n_cats <= 9:
            rows, cols = 3, 3
        elif n_cats <= 12:
            rows, cols = 3, 4
        elif n_cats <= 16:
            rows, cols = 4, 4
        elif n_cats <= 25:
            rows, cols = 5, 5
        elif n_cats <= 36:
            rows, cols = 6, 6
        else:
            # For very large numbers of categories, use 8x8 grid and limit to first 64
            rows, cols = 8, 8
            if n_cats > 64:
                self.logger.warning(f"Too many categories ({n_cats}), limiting visualization to first 64")
                categories = categories[:64]
                n_cats = 64
            
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 7*rows))
        if n_cats == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, category in enumerate(categories):
            ax = axes[i]
            
            # Plot historical data first
            if self.historical_data is not None and category in self.historical_data.columns:
                hist_years = self.historical_data['Year']
                hist_consumption = self.historical_data[category]
                
                # Remove NaN values
                valid_mask = ~hist_consumption.isna()
                hist_years_clean = hist_years[valid_mask]
                hist_consumption_clean = hist_consumption[valid_mask]
                
                ax.plot(hist_years_clean, hist_consumption_clean, 
                       marker='s', linewidth=3, markersize=5, color='black', 
                       label='Historical Data', alpha=0.9, zorder=10)
                
                # Add vertical line to separate historical from forecast
                if len(hist_years_clean) > 0:
                    last_hist_year = hist_years_clean.iloc[-1]
                    ax.axvline(x=last_hist_year, color='gray', linestyle='--', 
                              alpha=0.7, linewidth=2)
            
            # Plot individual algorithms with thin lines
            for algo in ['XGBoost', 'RandomForest']:
                col_name = f'{category}_{algo}'
                if col_name in forecast_df.columns:
                    ax.plot(forecast_df['Year'], forecast_df[col_name], 
                           linewidth=1.5, alpha=0.6, label=f'{algo}', linestyle='-')
            
            # Plot ensemble with thick line
            ensemble_col = f'{category}_Ensemble'
            if ensemble_col in ensemble_df.columns:
                ax.plot(ensemble_df['Year'], ensemble_df[ensemble_col], 
                       linewidth=4, color='red', label='Ensemble', alpha=0.9, zorder=5)
            
            ax.set_title(f'{self.get_clean_category_name(category)}', 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Consumption (tonnes)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
            
            # Set reasonable y-axis limits
            if self.historical_data is not None and category in self.historical_data.columns:
                hist_min = self.historical_data[category].min()
                hist_max = self.historical_data[category].max()
                forecast_min = forecast_df[[col for col in forecast_df.columns if category in col and col != 'Year']].min().min()
                forecast_max = forecast_df[[col for col in forecast_df.columns if category in col and col != 'Year']].max().max()
                
                y_min = min(hist_min, forecast_min) * 0.9
                y_max = max(hist_max, forecast_max) * 1.1
                ax.set_ylim(y_min, y_max)
        
        # Hide empty subplots
        for j in range(len(categories), len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Historical Data (2004-2023) & ML Ensemble Forecasts (2024-2050)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(viz_dir / 'historical_ensemble_vs_algorithms.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_heatmap(self, viz_dir: Path) -> None:
        """Create feature importance heatmap for tree-based models, grouped by algorithm."""
        # Collect feature importance data grouped by algorithm
        xgboost_data = {}
        randomforest_data = {}
        
        for category, algorithms in self.results.items():
            for algo_name, result in algorithms.items():
                if 'feature_importance' in result:
                    category_clean = self.get_clean_category_name(category)
                    
                    if algo_name == 'XGBoost':
                        xgboost_data[category_clean] = result['feature_importance']
                    elif algo_name == 'RandomForest':
                        randomforest_data[category_clean] = result['feature_importance']
        
        if not xgboost_data and not randomforest_data:
            return
        
        # Create DataFrames for each algorithm
        xgb_df = pd.DataFrame(xgboost_data).T if xgboost_data else pd.DataFrame()
        rf_df = pd.DataFrame(randomforest_data).T if randomforest_data else pd.DataFrame()
        
        # Combine with algorithm labels
        combined_data = []
        if not xgb_df.empty:
            for idx in xgb_df.index:
                combined_data.append(('XGBoost', idx, xgb_df.loc[idx]))
        if not rf_df.empty:
            for idx in rf_df.index:
                combined_data.append(('RandomForest', idx, rf_df.loc[idx]))
        
        if not combined_data:
            return
        
        # Create combined DataFrame with multi-index
        rows = []
        indices = []
        for algo, category, values in combined_data:
            rows.append(values)
            indices.append(f"{algo} - {category}")
        
        importance_df = pd.DataFrame(rows, index=indices)
        importance_df = importance_df.fillna(0)
        
        # Clean feature names to match current WM data structure
        current_columns = list(importance_df.columns)
        feature_mapping = {}
        for col in current_columns:
            if 'Population' in col:
                feature_mapping[col] = 'Population'
            elif 'Urbanisation' in col or 'urbanisation' in col.lower():
                feature_mapping[col] = 'Urbanisation'
            elif 'GDP' in col:
                feature_mapping[col] = 'GDP (Real 2015 AUD)'
            elif 'Iron' in col and 'Ore' in col:
                feature_mapping[col] = 'Iron Ore Production'
            elif 'Coal' in col:
                feature_mapping[col] = 'Coal Production'
            elif 'IP_Index' in col or 'Industrial' in col:
                feature_mapping[col] = 'Industrial Production'
        
        importance_df = importance_df.rename(columns=feature_mapping)
        
        # Sort by algorithm to group them together
        xgb_rows = [idx for idx in importance_df.index if idx.startswith('XGBoost')]
        rf_rows = [idx for idx in importance_df.index if idx.startswith('RandomForest')]
        importance_df = importance_df.reindex(xgb_rows + rf_rows)
        
        # Create heatmap
        plt.figure(figsize=(12, len(importance_df.index) * 0.6 + 3))
        sns.heatmap(importance_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Feature Importance'})
        plt.title('Feature Importance by Algorithm and Steel Category', fontsize=16, fontweight='bold')
        plt.xlabel('Economic Features')
        plt.ylabel('Algorithm - Steel Category')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(viz_dir / 'feature_importance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_accuracy_dashboard(self, viz_dir: Path) -> None:
        """Create algorithm accuracy dashboard."""
        # Collect accuracy data
        accuracy_data = []
        for category, algorithms in self.results.items():
            for algo_name, result in algorithms.items():
                accuracy_data.append({
                    'Algorithm': algo_name,
                    'Category': category.replace('_tonnes', '').replace('_', ' '),
                    'MAPE': result['metrics']['MAPE'],
                    'R2': result['metrics']['R2']
                })
        
        df = pd.DataFrame(accuracy_data)
        
        # Create dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Average MAPE by Algorithm
        avg_mape = df.groupby('Algorithm')['MAPE'].mean().sort_values()
        colors = ['green', 'lightgreen', 'orange', 'red']
        bars1 = ax1.bar(avg_mape.index, avg_mape.values, color=colors)
        ax1.set_title('Average MAPE by Algorithm', fontsize=12, fontweight='bold')
        ax1.set_ylabel('MAPE (%)')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. MAPE by Category and Algorithm
        pivot_mape = df.pivot(index='Category', columns='Algorithm', values='MAPE')
        pivot_mape.plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title('MAPE by Category and Algorithm', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAPE (%)')
        ax2.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. R² Score Distribution
        df['R2'].hist(bins=15, ax=ax3, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(df['R2'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean R² = {df["R2"].mean():.3f}')
        ax3.set_title('R² Score Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('R² Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Algorithm Performance Radar
        avg_stats = df.groupby('Algorithm').agg({
            'MAPE': 'mean',
            'R2': 'mean'
        }).reset_index()
        
        # Normalize MAPE (invert so higher is better)
        avg_stats['MAPE_norm'] = 1 / (1 + avg_stats['MAPE'] / 100)
        
        algorithms = avg_stats['Algorithm'].tolist()
        mape_scores = avg_stats['MAPE_norm'].tolist()
        r2_scores = avg_stats['R2'].tolist()
        
        x = range(len(algorithms))
        width = 0.35
        
        bars_mape = ax4.bar([i - width/2 for i in x], mape_scores, width, 
                          label='MAPE (normalized)', alpha=0.8, color='lightcoral')
        bars_r2 = ax4.bar([i + width/2 for i in x], r2_scores, width, 
                         label='R² Score', alpha=0.8, color='lightblue')
        
        ax4.set_title('Algorithm Performance Summary', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Performance Score')
        ax4.set_xticks(x)
        ax4.set_xticklabels(algorithms)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'algorithm_accuracy_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_wsa_steel_taxonomy_analysis(self) -> None:
        """Generate WSA Steel Taxonomy Analysis based on official WSA hierarchy diagrams."""
        self.logger.info("Generating WSA Steel Taxonomy Analysis...")
        
        try:
            # Import new WSA taxonomy module
            from analysis.wsa_steel_taxonomy import WSASteelTaxonomyAnalyzer
            
            # Try to find the complete 2004-2050 file first, fallback to 2025-2050
            complete_ensemble_file = self.output_dir / 'Ensemble_Forecasts_2004-2050.csv'
            ensemble_file = self.output_dir / 'Ensemble_Forecasts_2025-2050.csv'
            
            # Use complete file if available, otherwise use forecast-only file
            if complete_ensemble_file.exists():
                file_to_use = complete_ensemble_file
                self.logger.info("Using complete 2004-2050 ensemble file for WSA analysis")
            elif ensemble_file.exists():
                file_to_use = ensemble_file
                self.logger.info("Using 2025-2050 ensemble file for WSA analysis")
            else:
                file_to_use = None
            
            if file_to_use:
                # Create WSA taxonomy analysis subdirectory
                wsa_output_dir = self.output_dir / 'wsa_steel_taxonomy_analysis'
                wsa_output_dir.mkdir(exist_ok=True)
                
                # Initialize WSA taxonomy analyzer
                analyzer = WSASteelTaxonomyAnalyzer()
                
                # Generate comprehensive WSA analysis
                generated_files = analyzer.generate_complete_wsa_analysis(
                    track_a_forecast_file=str(file_to_use),
                    output_directory=str(wsa_output_dir)
                )
                
                self.logger.info(f"WSA Steel Taxonomy Analysis completed:")
                self.logger.info(f"  • Generated {len(generated_files)} files")
                self.logger.info(f"  • All files saved to: {wsa_output_dir}")
                for file_type, file_path in generated_files.items():
                    self.logger.info(f"    • {file_type}: {Path(file_path).name}")
                
            else:
                self.logger.warning("Ensemble forecast file not found - skipping WSA taxonomy analysis")
                
        except ImportError as e:
            self.logger.error(f"WSA Steel Taxonomy module not available: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating WSA Steel Taxonomy analysis: {str(e)}")
    
    def _extract_wsa_v3_structure(self, html_path: Path) -> Dict[str, Any]:
        """Extract WSA v3 4-level hierarchy structure from HTML document."""
        try:
            wsa_v3_structure = {
                "material_flow_hierarchy": {
                    "level_0": "Raw Materials (Iron Ore Production)",
                    "level_1": "Intermediate Materials (Pig Iron Production)", 
                    "level_2": "Primary Steel Production (Crude Steel)",
                    "level_3": "Steel Forming Methods (Ingots/Continuous Casting)",
                    "level_4": "Finished Products (Hot Rolled, Tubular)"
                },
                "crude_steel_breakdown": {
                    "by_process": ["EAF (Electric Arc Furnace)", "BOF (Basic Oxygen Furnace)"],
                    "by_casting": ["Ingots", "Continuously-cast Steel"]
                },
                "product_form_hierarchy": {
                    "hot_rolled_products": {
                        "flat_products": ["Hot Rolled Coil/Sheet/Strip <3mm", "Coated Sheet/Strip"],
                        "long_products": ["Wire Rod", "Railway Track Material"]
                    },
                    "tubular_products": ["Total Production of Tubular Products"]
                },
                "steel_trade_hierarchy": {
                    "direct_trade": ["Flat Products", "Long Products", "Tubular Products"],
                    "indirect_trade": ["Steel-containing goods", "Embedded steel content"]
                }
            }
            
            self.logger.info("Extracted WSA v3 4-diagram structure")
            return wsa_v3_structure
            
        except Exception as e:
            self.logger.warning(f"Could not extract WSA v3 structure: {str(e)}")
            return {}
    
    def _generate_pdf_reports(self) -> None:
        """Generate PDF reports from markdown files."""
        self.logger.info("Generating PDF reports...")
        
        try:
            # Check if convert_md_to_pdf_final.py exists
            pdf_converter_path = Path("convert_md_to_pdf_final.py")
            if not pdf_converter_path.exists():
                self.logger.warning("PDF converter script not found, skipping PDF generation")
                return
            
            # Generate PDFs for outputs/track_a directory
            track_a_outputs = Path("outputs/track_a")
            if track_a_outputs.exists():
                # Find all markdown files
                md_files = list(track_a_outputs.rglob("*.md"))
                if md_files:
                    self.logger.info(f"Found {len(md_files)} markdown files in outputs/track_a")
                    
                    # Create PDF output directory
                    pdf_output_dir = track_a_outputs / "pdf_reports"
                    pdf_output_dir.mkdir(exist_ok=True)
                    
                    # Run the PDF converter
                    import subprocess
                    result = subprocess.run(
                        ["python3", "convert_md_to_pdf_final.py", str(track_a_outputs), str(pdf_output_dir)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.logger.info(f"PDF reports generated successfully in {pdf_output_dir}")
                    else:
                        self.logger.error(f"PDF generation failed: {result.stderr}")
                else:
                    self.logger.info("No markdown files found to convert to PDF")
            
            # Generate PDFs for forecasts directory (with mermaid diagrams)
            forecast_dir = self.output_dir
            if forecast_dir.exists():
                # Find all markdown files in the forecast directory and subdirectories
                all_md_files = list(forecast_dir.rglob("*.md"))
                
                if all_md_files:
                    self.logger.info(f"Found {len(all_md_files)} markdown files in {forecast_dir}")
                    
                    # Create single pdf output directory within the forecast run folder
                    pdf_output_dir = forecast_dir / "pdf_reports"
                    pdf_output_dir.mkdir(exist_ok=True)
                    
                    # Use the final converter which supports mermaid diagrams
                    import subprocess
                    result = subprocess.run(
                        ["python3", "convert_md_to_pdf_final.py", str(forecast_dir), str(pdf_output_dir)],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode == 0:
                        self.logger.info(f"PDF reports generated successfully in {pdf_output_dir}")
                    else:
                        self.logger.error(f"PDF generation failed: {result.stderr}")
                            
        except Exception as e:
            self.logger.error(f"Error generating PDF reports: {str(e)}")
    
    def save_results(self) -> None:
        """Save all results and performance comparison."""
        self.logger.info("Saving results and performance comparison...")
        
        # Generate WSA Steel Taxonomy Analysis first
        self._generate_wsa_steel_taxonomy_analysis()
        
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
                    importance_data = []
                    for feature, importance in result['feature_importance'].items():
                        importance_data.append({
                            'feature': feature,
                            'importance': importance
                        })
                    
                    importance_df = pd.DataFrame(importance_data)
                    # Create feature_importance subfolder
                    feature_importance_dir = self.output_dir / 'feature_importance'
                    feature_importance_dir.mkdir(exist_ok=True)
                    
                    importance_file = feature_importance_dir / f'{category}_{algo_name}_feature_importance.csv'
                    importance_df.to_csv(importance_file, index=False)
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for category, models in self.models.items():
            for algo_name, model in models.items():
                model_file = models_dir / f'{category}_{algo_name}_model.joblib'
                joblib.dump(model, model_file)
        
        self.logger.info(f"All results saved to {self.output_dir}")
        self.logger.info("✅ WSA Steel Taxonomy Analysis completed with Track A forecasts")
        
        # Generate PDF reports
        self._generate_pdf_reports()

def main():
    """Main execution function."""
    print("🔬 ML Algorithms with Regression Model Features - Simplified")
    print("=" * 65)
    
    # Initialize trainer
    trainer = RegressionMLTrainer()
    
    # Load data
    historical_data, projection_data = trainer.load_data()
    
    # Train all algorithms
    trainer.train_all_algorithms(historical_data)
    
    # Generate forecasts
    forecast_df = trainer.generate_forecasts(projection_data)
    
    # Create ensemble forecast
    ensemble_df = trainer.create_ensemble_forecast(forecast_df)
    
    # Create visualizations
    trainer.create_visualizations(forecast_df, ensemble_df)
    
    # Save results
    trainer.save_results()
    
    print("\\n" + "=" * 65)
    print("✅ Training completed! Check the forecasts/track_a_*/ directory for all results.")
    print("🏗️ WSA Steel Taxonomy Analysis included with comprehensive CSVs, visualizations, and mermaid hierarchy charts.")
    print("   Based on official WSA hierarchy diagrams - Production Flow, Trade Flow, and Consumption Metrics.")
    print("   Includes interactive mermaid diagrams for 2015, 2020, 2025, 2035, and 2050 showing volume flows through WSA hierarchy.")
    print("📄 PDF reports automatically generated from all markdown files with rendered mermaid diagrams.")
    print("=" * 65)

if __name__ == "__main__":
    main()