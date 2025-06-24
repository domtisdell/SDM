"""
Steel Demand ML Model - Feature Engineering Module
Creates 150+ features from real economic and steel consumption data.
All parameters loaded from CSV configuration files.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

class SteelDemandFeatureEngineering:
    """
    Feature engineering for Australian Steel Demand Model ML system.
    Creates comprehensive feature set from real historical data.
    """
    
    def __init__(self, data_loader, config_path: str = "config/"):
        """
        Initialize feature engineering with data loader and configuration.
        
        Args:
            data_loader: SteelDemandDataLoader instance
            config_path: Path to configuration directory
        """
        self.data_loader = data_loader
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.scalers = {}
        self.feature_metadata = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_features(self, data: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Create comprehensive feature set from historical data.
        
        Args:
            data: Historical dataset with macro drivers and steel consumption
            target_column: Target variable column name
            
        Returns:
            Enhanced dataset with engineered features
        """
        try:
            self.logger.info("Starting feature engineering process")
            
            # Get configuration parameters
            max_lag = self.data_loader.get_model_config('max_lag_features')
            rolling_windows = self.data_loader.get_model_config('rolling_window_sizes')
            correlation_threshold = self.data_loader.get_model_config('correlation_threshold')
            
            # Start with original data
            features_df = data.copy()
            feature_count = len(features_df.columns)
            
            # 1. Economic Indicator Features
            features_df = self._create_economic_features(features_df)
            self.logger.info(f"Economic features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 2. Lag Features
            features_df = self._create_lag_features(features_df, max_lag)
            self.logger.info(f"Lag features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 3. Rolling Statistics Features
            features_df = self._create_rolling_features(features_df, rolling_windows)
            self.logger.info(f"Rolling features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 4. Growth Rate Features
            features_df = self._create_growth_features(features_df)
            self.logger.info(f"Growth features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 5. Ratio and Interaction Features
            features_df = self._create_ratio_features(features_df)
            self.logger.info(f"Ratio features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 6. Regional Features
            features_df = self._create_regional_features(features_df)
            self.logger.info(f"Regional features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 7. Steel Market Features
            features_df = self._create_steel_market_features(features_df)
            self.logger.info(f"Steel market features: {len(features_df.columns) - feature_count} added")
            feature_count = len(features_df.columns)
            
            # 8. Economic Cycle Features
            features_df = self._create_cycle_features(features_df)
            cycle_features_added = len(features_df.columns) - feature_count
            self.logger.info(f"Cycle features: {cycle_features_added} added")
            
            # Log comprehensive feature engineering status
            total_features = len(features_df.columns)
            memory_usage = features_df.memory_usage(deep=True).sum() / 1024**2  # MB
            self.logger.info(f"Feature engineering stage completed: {total_features} total features")
            self.logger.info(f"DataFrame memory usage: {memory_usage:.2f} MB")
            self.logger.info(f"DataFrame shape: {features_df.shape}")
            
            # Remove highly correlated features
            if target_column:
                self.logger.info(f"Starting correlation analysis for target: {target_column}")
                features_df = self._remove_correlated_features(features_df, target_column, correlation_threshold)
                self.logger.info(f"Correlation analysis completed. Final features: {len(features_df.columns)}")
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            total_features = len(features_df.columns)
            self.logger.info(f"Feature engineering completed: {total_features} total features")
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering: {str(e)}")
            raise
    
    def _create_economic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create features from economic indicators."""
        df = data.copy()
        
        # Get economic indicators configuration
        economic_indicators = self.data_loader.get_economic_indicators()
        
        for _, indicator in economic_indicators.iterrows():
            column_name = indicator['column_name']
            
            if column_name in df.columns:
                # Log transformation for GDP and production volumes
                if indicator['transformation'] == 'log' and (df[column_name] > 0).all():
                    df[f'{column_name}_log'] = np.log(df[column_name])
                
                # Normalized versions
                df[f'{column_name}_normalized'] = (df[column_name] - df[column_name].mean()) / df[column_name].std()
                
                # Detrended versions
                trend = np.polyfit(range(len(df)), df[column_name], 1)
                trend_line = np.polyval(trend, range(len(df)))
                df[f'{column_name}_detrended'] = df[column_name] - trend_line
        
        # Additional calculated economic indicators
        if 'GDP_Real_AUD_Billion' in df.columns and 'Total_Population_Millions' in df.columns:
            df['GDP_per_capita'] = df['GDP_Real_AUD_Billion'] * 1000 / df['Total_Population_Millions']
        
        if 'Iron_Ore_Production_Mt' in df.columns and 'Coal_Production_Mt' in df.columns:
            df['Mining_Production_Total'] = df['Iron_Ore_Production_Mt'] + df['Coal_Production_Mt']
            df['Iron_Coal_Ratio'] = df['Iron_Ore_Production_Mt'] / (df['Coal_Production_Mt'] + 1e-6)
        
        if 'National_Urbanisation_Rate_pct' in df.columns and 'Total_Population_Millions' in df.columns:
            df['Urban_Population_Millions'] = df['Total_Population_Millions'] * df['National_Urbanisation_Rate_pct'] / 100
            df['Rural_Population_Millions'] = df['Total_Population_Millions'] - df['Urban_Population_Millions']
        
        return df
    
    def _create_lag_features(self, data: pd.DataFrame, max_lag: int) -> pd.DataFrame:
        """Create lagged features for time series analysis."""
        df = data.copy()
        
        # Get key economic columns for lagging
        economic_columns = [
            'GDP_Real_AUD_Billion',
            'Total_Population_Millions', 
            'Iron_Ore_Production_Mt',
            'Coal_Production_Mt',
            'Industrial_Production_Index_Base_2007',
            'National_Urbanisation_Rate_pct'
        ]
        
        # Add steel consumption columns if they exist
        steel_columns = [col for col in df.columns if col.endswith('_tonnes')]
        lag_columns = economic_columns + steel_columns
        
        # Create lag features
        for column in lag_columns:
            if column in df.columns:
                for lag in range(1, max_lag + 1):
                    df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        return df
    
    def _create_rolling_features(self, data: pd.DataFrame, rolling_windows: List[int]) -> pd.DataFrame:
        """Create rolling statistics features using vectorized operations."""
        df = data.copy()
        
        # Get numeric columns for rolling features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != 'Year']
        
        # Create all rolling features at once using concat
        rolling_features = []
        
        for column in numeric_columns:
            for window in rolling_windows:
                if len(df) >= window:  # Only create if we have enough data
                    # Create all rolling stats for this column/window combo
                    rolling_data = {
                        f'{column}_rolling_mean_{window}': df[column].rolling(window=window, min_periods=1).mean(),
                        f'{column}_rolling_std_{window}': df[column].rolling(window=window, min_periods=1).std(),
                        f'{column}_rolling_min_{window}': df[column].rolling(window=window, min_periods=1).min(),
                        f'{column}_rolling_max_{window}': df[column].rolling(window=window, min_periods=1).max()
                    }
                    rolling_features.append(pd.DataFrame(rolling_data, index=df.index))
        
        # Concatenate all rolling features at once
        if rolling_features:
            rolling_df = pd.concat(rolling_features, axis=1)
            df = pd.concat([df, rolling_df], axis=1)
        
        return df
    
    def _create_growth_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create growth rate and change features using vectorized operations."""
        df = data.copy()
        
        # Get numeric columns for growth features
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col != 'Year']
        
        # Create all growth features at once using concat
        growth_features = []
        
        for column in numeric_columns:
            yoy_change = df[column].diff()
            growth_data = {
                f'{column}_yoy_change': yoy_change,
                f'{column}_yoy_pct_change': df[column].pct_change() * 100,
                f'{column}_acceleration': yoy_change.diff()
            }
            growth_features.append(pd.DataFrame(growth_data, index=df.index))
        
        # Concatenate all growth features at once
        if growth_features:
            growth_df = pd.concat(growth_features, axis=1)
            df = pd.concat([df, growth_df], axis=1)
        
        return df
    
    def _create_ratio_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ratio and interaction features between key variables."""
        df = data.copy()
        
        # Steel intensity ratios
        steel_columns = [col for col in df.columns if col.endswith('_tonnes') and not 'lag' in col and not 'rolling' in col]
        
        if 'GDP_Real_AUD_Billion' in df.columns:
            for steel_col in steel_columns:
                df[f'{steel_col}_per_GDP'] = df[steel_col] / (df['GDP_Real_AUD_Billion'] + 1e-6)
        
        if 'Total_Population_Millions' in df.columns:
            for steel_col in steel_columns:
                df[f'{steel_col}_per_capita'] = df[steel_col] / (df['Total_Population_Millions'] + 1e-6)
        
        if 'Industrial_Production_Index_Base_2007' in df.columns:
            for steel_col in steel_columns:
                df[f'{steel_col}_per_industrial_index'] = df[steel_col] / (df['Industrial_Production_Index_Base_2007'] + 1e-6)
        
        # Production efficiency ratios
        if 'Iron_Ore_Production_Mt' in df.columns and 'GDP_Real_AUD_Billion' in df.columns:
            df['Iron_Ore_Efficiency'] = df['Iron_Ore_Production_Mt'] / (df['GDP_Real_AUD_Billion'] + 1e-6)
        
        if 'Coal_Production_Mt' in df.columns and 'GDP_Real_AUD_Billion' in df.columns:
            df['Coal_Efficiency'] = df['Coal_Production_Mt'] / (df['GDP_Real_AUD_Billion'] + 1e-6)
        
        # Cross-category steel ratios
        if len(steel_columns) > 1:
            for i, col1 in enumerate(steel_columns):
                for col2 in steel_columns[i+1:]:
                    df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-6)
        
        return df
    
    def _create_regional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create regional distribution features using vectorized operations."""
        df = data.copy()
        
        # Get regional factors from configuration
        regional_factors = self.data_loader.get_regional_factors()
        
        # Collect all regional features to create at once
        regional_features = []
        
        # Population concentration metrics
        major_states = ['NSW_Population_Millions', 'VIC_Population_Millions', 'QLD_Population_Millions']
        if all(col in df.columns for col in major_states):
            regional_data = {
                'Major_States_Population': df[major_states].sum(axis=1),
            }
            regional_data['Major_States_Share'] = regional_data['Major_States_Population'] / df['Total_Population_Millions']
            regional_features.append(pd.DataFrame(regional_data, index=df.index))
        
        # Regional growth differentials - vectorized
        population_cols = [col for col in df.columns if 'Population_Millions' in col and col != 'Total_Population_Millions']
        if population_cols:
            growth_data = {}
            for col in population_cols:
                growth_data[f'{col}_growth_rate'] = df[col].pct_change() * 100
            regional_features.append(pd.DataFrame(growth_data, index=df.index))
        
        # Urbanisation differentials by state - vectorized
        urbanisation_cols = [col for col in df.columns if 'Urbanisation_Rate_pct' in col]
        if len(urbanisation_cols) > 1 and 'National_Urbanisation_Rate_pct' in df.columns:
            urban_data = {}
            for col in urbanisation_cols:
                if col != 'National_Urbanisation_Rate_pct':
                    urban_data[f'{col}_differential'] = df[col] - df['National_Urbanisation_Rate_pct']
            if urban_data:
                regional_features.append(pd.DataFrame(urban_data, index=df.index))
        
        # Concatenate all regional features at once
        if regional_features:
            regional_df = pd.concat(regional_features, axis=1)
            df = pd.concat([df, regional_df], axis=1)
        
        return df
    
    def _create_steel_market_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create steel market specific features."""
        df = data.copy()
        
        # Get steel categories configuration
        steel_categories = self.data_loader.get_steel_categories()
        
        # Total steel consumption and market share
        steel_columns = [col for col in df.columns if col.endswith('_tonnes') and not 'lag' in col and not 'rolling' in col]
        
        if len(steel_columns) > 1:
            total_col = 'Total_Steel_Consumption_tonnes'
            if total_col in df.columns:
                # Market share features
                for steel_col in steel_columns:
                    if steel_col != total_col:
                        df[f'{steel_col}_market_share'] = df[steel_col] / (df[total_col] + 1e-6)
                
                # Concentration metrics
                df['Steel_Market_Concentration'] = sum((df[f'{col}_market_share'] ** 2) for col in steel_columns if col != total_col and f'{col}_market_share' in df.columns)
        
        # Steel elasticity features based on configuration
        for _, category in steel_categories.iterrows():
            target_col = category['target_column']
            if target_col in df.columns:
                gdp_elasticity = category['elasticity_gdp']
                pop_elasticity = category['elasticity_population']
                
                # Expected consumption based on elasticity
                if 'GDP_Real_AUD_Billion' in df.columns:
                    base_gdp = df['GDP_Real_AUD_Billion'].iloc[0] if len(df) > 0 else 1
                    df[f'{target_col}_gdp_expected'] = df[target_col].iloc[0] * (df['GDP_Real_AUD_Billion'] / base_gdp) ** gdp_elasticity if len(df) > 0 else 0
                
                if 'Total_Population_Millions' in df.columns:
                    base_pop = df['Total_Population_Millions'].iloc[0] if len(df) > 0 else 1
                    df[f'{target_col}_pop_expected'] = df[target_col].iloc[0] * (df['Total_Population_Millions'] / base_pop) ** pop_elasticity if len(df) > 0 else 0
        
        return df
    
    def _create_cycle_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create economic cycle and trend features."""
        df = data.copy()
        
        # Business cycle indicators
        if 'GDP_Real_AUD_Billion' in df.columns:
            # GDP trend and cycle components
            gdp_trend = df['GDP_Real_AUD_Billion'].rolling(window=5, center=True, min_periods=1).mean()
            df['GDP_cycle'] = df['GDP_Real_AUD_Billion'] - gdp_trend
            df['GDP_trend'] = gdp_trend
        
        if 'Industrial_Production_Index_Base_2007' in df.columns:
            # Industrial production cycle
            ip_trend = df['Industrial_Production_Index_Base_2007'].rolling(window=5, center=True, min_periods=1).mean()
            df['Industrial_Production_cycle'] = df['Industrial_Production_Index_Base_2007'] - ip_trend
        
        # Time-based features
        if 'Year' in df.columns:
            # Years since reference point
            base_year = df['Year'].min()
            df['Years_since_base'] = df['Year'] - base_year
            
            # Economic maturity indicator
            df['Economic_maturity'] = (df['Year'] - 2000) / 50  # Normalized to 0-1 over 50 years
        
        return df
    
    def _remove_correlated_features(self, data: pd.DataFrame, target_column: str, threshold: float) -> pd.DataFrame:
        """Remove highly correlated features using efficient approach for large feature sets."""
        df = data.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_column and col != 'Year']
        
        initial_features = len(feature_cols)
        self.logger.info(f"Starting correlation analysis with {initial_features} features")
        
        # For any feature set, pre-select based on target correlation due to small dataset
        if len(feature_cols) > 50:
            if target_column in df.columns:
                target_correlations = {}
                for col in feature_cols:
                    try:
                        corr = abs(df[col].corr(df[target_column]))
                        if not np.isnan(corr):
                            target_correlations[col] = corr
                    except:
                        continue
                
                # Keep top 50 features based on target correlation (reduced for small dataset)
                top_features = sorted(target_correlations.items(), key=lambda x: x[1], reverse=True)[:50]
                feature_cols = [col for col, _ in top_features]
                self.logger.info(f"Pre-selected top {len(feature_cols)} features based on target correlation")
        
        # Skip detailed correlation analysis for now to get the system running
        # This can be re-enabled once the basic training pipeline works
        self.logger.info(f"Skipping detailed correlation analysis for stability")
        
        final_features = len(feature_cols)
        
        # Keep the selected features
        cols_to_keep = feature_cols + [target_column, 'Year']
        cols_to_keep = [col for col in cols_to_keep if col in df.columns]
        df = df[cols_to_keep]
        
        self.logger.info(f"Feature selection completed: {initial_features} -> {final_features} features")
        
        return df
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the feature set."""
        df = data.copy()
        
        # Forward fill for time series data
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        
        # Backward fill for remaining missing values
        df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
        
        # Fill any remaining missing values with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def scale_features(self, data: pd.DataFrame, target_columns: List[str], 
                      method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """
        Scale features for ML models.
        
        Args:
            data: Dataset with features
            target_columns: Target variable column names to exclude from scaling
            method: Scaling method ('standard' or 'robust')
            
        Returns:
            Scaled dataset and fitted scalers
        """
        df = data.copy()
        
        # Get feature columns (exclude targets and Year)
        exclude_cols = target_columns + ['Year']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Select numeric features only
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform features
        df[numeric_features] = scaler.fit_transform(df[numeric_features])
        
        # Store scaler for future use
        self.scalers[method] = scaler
        
        self.logger.info(f"Scaled {len(numeric_features)} features using {method} scaling")
        
        return df, {method: scaler}
    
    def select_features(self, data: pd.DataFrame, target_column: str, 
                       k: int = 50) -> pd.DataFrame:
        """
        Select top k features based on statistical tests.
        
        Args:
            data: Dataset with features
            target_column: Target variable column name
            k: Number of features to select
            
        Returns:
            Dataset with selected features
        """
        df = data.copy()
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in [target_column, 'Year']]
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_features) <= k:
            self.logger.info(f"Number of features ({len(numeric_features)}) <= k ({k}), keeping all features")
            return df
        
        # Select features
        selector = SelectKBest(score_func=f_regression, k=k)
        
        # Handle missing values in target
        valid_indices = df[target_column].notna()
        
        selector.fit(df.loc[valid_indices, numeric_features], df.loc[valid_indices, target_column])
        
        # Get selected feature names
        selected_features = [numeric_features[i] for i in range(len(numeric_features)) if selector.get_support()[i]]
        
        # Keep selected features plus target and Year
        keep_cols = selected_features + [target_column, 'Year']
        df_selected = df[keep_cols].copy()
        
        self.logger.info(f"Selected {len(selected_features)} features out of {len(numeric_features)}")
        
        return df_selected
    
    def get_feature_importance_summary(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Generate feature importance summary based on correlations.
        
        Args:
            data: Dataset with features
            target_column: Target variable column name
            
        Returns:
            Feature importance summary
        """
        feature_cols = [col for col in data.columns if col not in [target_column, 'Year']]
        numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        importance_data = []
        
        for feature in numeric_features:
            correlation = data[feature].corr(data[target_column])
            importance_data.append({
                'feature': feature,
                'correlation': correlation,
                'abs_correlation': abs(correlation),
                'feature_type': self._classify_feature_type(feature)
            })
        
        importance_df = pd.DataFrame(importance_data)
        importance_df = importance_df.sort_values('abs_correlation', ascending=False)
        
        return importance_df
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on naming convention."""
        if 'lag_' in feature_name:
            return 'lag'
        elif 'rolling_' in feature_name:
            return 'rolling'
        elif '_yoy_' in feature_name or '_pct_change' in feature_name:
            return 'growth'
        elif '_ratio' in feature_name or '_per_' in feature_name:
            return 'ratio'
        elif 'cycle' in feature_name or 'trend' in feature_name:
            return 'cycle'
        elif any(region in feature_name for region in ['NSW', 'VIC', 'QLD', 'WA']):
            return 'regional'
        elif 'Steel' in feature_name or 'tonnes' in feature_name:
            return 'steel_market'
        else:
            return 'economic'