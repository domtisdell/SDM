#!/usr/bin/env python3
"""
Test script to validate the ML system works with real data.
Quick validation without full training.
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_data_loader():
    """Test data loading functionality."""
    print("Testing data loader...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        
        # Initialize data loader
        loader = SteelDemandDataLoader("config/")
        
        # Load all data
        data = loader.load_all_data()
        print(f"Loaded {len(data)} datasets")
        
        # Test historical data consolidation
        historical = loader.get_historical_data()
        print(f"Historical data shape: {historical.shape}")
        print(f"Year range: {historical['Year'].min()}-{historical['Year'].max()}")
        
        # Test projection data
        projections = loader.get_projection_data()
        print(f"Projection data shape: {projections.shape}")
        print(f"Projection years: {projections['Year'].min()}-{projections['Year'].max()}")
        
        # Validate data quality
        quality_report = loader.validate_data_quality()
        print(f"Data quality: {quality_report['overall_quality']}")
        
        return True
        
    except Exception as e:
        print(f"Data loader test failed: {str(e)}")
        return False

def test_feature_engineering():
    """Test feature engineering functionality."""
    print("\nTesting feature engineering...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        from data.feature_engineering import SteelDemandFeatureEngineering
        
        # Initialize components
        loader = SteelDemandDataLoader("config/")
        loader.load_all_data()
        feature_engineer = SteelDemandFeatureEngineering(loader)
        
        # Get historical data
        historical = loader.get_historical_data()
        
        # Test feature creation for one category
        target_column = 'Hot_Rolled_Structural_Steel_tonnes'
        features_df = feature_engineer.create_features(historical, target_column)
        
        print(f"Original features: {historical.shape[1]}")
        print(f"Engineered features: {features_df.shape[1]}")
        print(f"Feature engineering created {features_df.shape[1] - historical.shape[1]} new features")
        
        # Test feature importance
        importance_summary = feature_engineer.get_feature_importance_summary(features_df, target_column)
        print(f"Top 5 features by correlation:")
        print(importance_summary.head()[['feature', 'correlation']].to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"Feature engineering test failed: {str(e)}")
        return False

def test_model_components():
    """Test individual model components."""
    print("\nTesting model components...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        from models.ensemble_models import XGBoostSteelModel, RandomForestSteelModel
        
        # Initialize data loader
        loader = SteelDemandDataLoader("config/")
        loader.load_all_data()
        
        # Get small sample of data for testing
        historical = loader.get_historical_data()
        
        # Create simple features for testing
        feature_cols = ['GDP_Real_AUD_Billion', 'Total_Population_Millions', 'Iron_Ore_Production_Mt']
        available_features = [col for col in feature_cols if col in historical.columns]
        
        if not available_features:
            print("No suitable features found for model testing")
            return False
        
        X = historical[available_features].fillna(method='ffill').fillna(0)
        y = historical['Hot_Rolled_Structural_Steel_tonnes'].fillna(method='ffill')
        
        # Remove rows with missing target
        valid_indices = y.notna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 5:
            print("Insufficient data for model testing")
            return False
        
        # Test XGBoost model
        print("Testing XGBoost model...")
        xgb_model = XGBoostSteelModel(loader)
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)
        print(f"XGBoost predictions range: {xgb_pred.min():.0f} - {xgb_pred.max():.0f}")
        
        # Test Random Forest model
        print("Testing Random Forest model...")
        rf_model = RandomForestSteelModel(loader)
        rf_model.fit(X, y)
        rf_pred = rf_model.predict(X)
        print(f"Random Forest predictions range: {rf_pred.min():.0f} - {rf_pred.max():.0f}")
        
        return True
        
    except Exception as e:
        print(f"Model components test failed: {str(e)}")
        return False

def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        
        loader = SteelDemandDataLoader("config/")
        
        # Test model config
        train_split = loader.get_model_config('train_test_split')
        print(f"Train/test split: {train_split}")
        
        xgb_estimators = loader.get_model_config('xgb_n_estimators')
        print(f"XGBoost estimators: {xgb_estimators}")
        
        # Test steel categories
        categories = loader.get_steel_categories()
        print(f"Steel categories: {len(categories)}")
        print(categories[['category', 'target_column']].to_string(index=False))
        
        # Test economic indicators
        indicators = loader.get_economic_indicators()
        print(f"Economic indicators: {len(indicators)}")
        
        # Test validation benchmarks
        benchmarks = loader.get_validation_benchmarks()
        print(f"Validation benchmarks: {len(benchmarks)}")
        
        return True
        
    except Exception as e:
        print(f"Configuration test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Australian Steel Demand ML System - Quick Test")
    print("="*60)
    
    # Setup basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    tests = [
        ("Configuration Loading", test_configuration),
        ("Data Loading", test_data_loader),
        ("Feature Engineering", test_feature_engineering),
        ("Model Components", test_model_components)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System ready for training.")
        return 0
    else:
        print("✗ Some tests failed. Check configuration and data files.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)