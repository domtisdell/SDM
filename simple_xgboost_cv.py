#!/usr/bin/env python3
"""
Simple XGBoost training with cross-validation to assess true performance.
This script uses only basic macro features and implements proper CV for honest evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score
import xgboost as xgb
from data.data_loader import SteelDemandDataLoader
import logging

def main():
    """Main function to train XGBoost with honest cross-validation."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info("Loading data...")
    data_loader = SteelDemandDataLoader("config/")
    data_loader.load_all_data()
    
    # Get historical data (20 samples)
    historical_data = data_loader.get_historical_data()
    
    # Use only the most important macro features (5 features for 20 samples)
    features = [
        'GDP_AUD_Real2015', 'Population', 'Iron_Ore_Production', 
        'Coal_Production', 'Wind_Offshore'
    ]
    
    # Test one steel category
    target = 'Total Production of Crude Steel'
    
    # Prepare data
    X = historical_data[features].fillna(0)
    y = historical_data[target].fillna(0)
    
    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Features: {features}")
    logger.info(f"Target: {target}")
    
    # Ultra-conservative XGBoost parameters
    xgb_params = {
        'n_estimators': 5,
        'max_depth': 1,
        'learning_rate': 0.5,
        'subsample': 0.4,
        'colsample_bytree': 0.4,
        'reg_alpha': 50.0,
        'reg_lambda': 100.0,
        'min_child_weight': 10,
        'random_state': 42
    }
    
    model = xgb.XGBRegressor(**xgb_params)
    
    # Cross-validation with small dataset
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    
    logger.info("Performing cross-validation...")
    
    # Use manual CV to get both MAPE and R2
    mape_scores = []
    r2_scores = []
    
    for train_idx, val_idx in cv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        fold_model = xgb.XGBRegressor(**xgb_params)
        fold_model.fit(X_train, y_train)
        
        # Predict
        y_pred = fold_model.predict(X_val)
        
        # Calculate metrics
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        r2 = r2_score(y_val, y_pred)
        
        mape_scores.append(mape)
        r2_scores.append(r2)
    
    mape_scores = np.array(mape_scores)
    r2_scores = np.array(r2_scores)
    
    logger.info("=" * 50)
    logger.info("HONEST CROSS-VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Features used: {len(features)}")
    logger.info(f"Data points: {len(X)}")
    logger.info(f"XGBoost parameters: {xgb_params}")
    logger.info("")
    logger.info(f"MAPE scores: {mape_scores}")
    logger.info(f"Mean MAPE: {np.mean(mape_scores):.2f}% ± {np.std(mape_scores):.2f}%")
    logger.info("")
    logger.info(f"R² scores: {r2_scores}")
    logger.info(f"Mean R²: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    logger.info("=" * 50)
    
    # Train on full dataset and show training error for comparison
    model.fit(X, y)
    train_pred = model.predict(X)
    train_mape = mean_absolute_percentage_error(y, train_pred) * 100
    train_r2 = r2_score(y, train_pred)
    
    logger.info(f"Training error (overfitted): MAPE={train_mape:.3f}%, R²={train_r2:.3f}")
    logger.info(f"CV error (honest): MAPE={np.mean(mape_scores):.2f}%, R²={np.mean(r2_scores):.3f}")
    
    overfitting_ratio = train_mape / np.mean(mape_scores)
    logger.info(f"Overfitting ratio: {overfitting_ratio:.2f}x")
    
    if overfitting_ratio < 0.5:
        logger.info("✅ Model is severely overfitting!")
    elif overfitting_ratio < 0.8:
        logger.info("⚠️ Model is moderately overfitting")
    else:
        logger.info("✅ Model generalization is reasonable")

if __name__ == "__main__":
    main()