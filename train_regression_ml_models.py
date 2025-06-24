#!/usr/bin/env python3
"""
ML Algorithms with Regression Model Features - Simplified Version

This script applies ML algorithms (XGBoost, Random Forest, Prophet, Linear Regression)
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
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Note: Prophet has been removed from this system

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
    log_file = output_dir / f"regression_ml_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    
    def __init__(self, output_dir: str = "forecasts/regression_ml"):
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
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray, category: str) -> LinearRegression:
        """Train Linear Regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
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
            
            # Train Linear Regression
            try:
                model = self.train_linear_regression(X, y, category)
                y_pred = model.predict(X)
                metrics = self.calculate_metrics(y, y_pred)
                
                self.results[category]['LinearRegression'] = {
                    'metrics': metrics,
                    'coefficients': dict(zip(features, model.coef_))
                }
                self.models[category]['LinearRegression'] = model
                self.logger.info(f"Linear Regression MAPE: {metrics['MAPE']:.2f}%")
                
            except Exception as e:
                self.logger.error(f"Linear Regression training failed for {category}: {str(e)}")
            
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
        forecast_file = self.output_dir / 'ML_Algorithm_Forecasts_2025-2050.csv'
        forecast_df.to_csv(forecast_file, index=False)
        self.logger.info(f"Forecasts saved to {forecast_file}")
        
        return forecast_df
    
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
        
        # Save ensemble forecasts
        ensemble_file = self.output_dir / 'Ensemble_Forecasts_2025-2050.csv'
        ensemble_df.to_csv(ensemble_file, index=False)
        self.logger.info(f"Ensemble forecasts saved to {ensemble_file}")
        
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
        categories = list(self.category_features.keys())
        
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
        else:
            rows, cols = 4, 4
            
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
            for algo in ['XGBoost', 'RandomForest', 'LinearRegression']:
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
        
        plt.suptitle('Historical Data, Individual Algorithms & Ensemble Forecasts (2004-2050)', 
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
                    importance_data = []
                    for feature, importance in result['feature_importance'].items():
                        importance_data.append({
                            'feature': feature,
                            'importance': importance
                        })
                    
                    importance_df = pd.DataFrame(importance_data)
                    importance_file = self.output_dir / f'{category}_{algo_name}_feature_importance.csv'
                    importance_df.to_csv(importance_file, index=False)
        
        # Save models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        for category, models in self.models.items():
            for algo_name, model in models.items():
                model_file = models_dir / f'{category}_{algo_name}_model.joblib'
                joblib.dump(model, model_file)
        
        self.logger.info(f"All results saved to {self.output_dir}")

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
    print("✅ Training completed! Check the forecasts/regression_ml/ directory for results.")
    print("=" * 65)

if __name__ == "__main__":
    main()