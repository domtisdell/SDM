#!/usr/bin/env python3
"""
Analyze feature sets used by Track A and available for Track B
"""

import sys
sys.path.append('.')
from data.data_loader import SteelDemandDataLoader

def main():
    # Load data loader
    loader = SteelDemandDataLoader()
    loader.load_all_data()

    # Get Track A features (historical data)
    historical_data = loader.get_historical_data()
    print('=== TRACK A FEATURE SET ===')
    print(f'Historical data shape: {historical_data.shape}')
    print('Features used by Track A:')
    feature_cols = [col for col in historical_data.columns if col != 'Year']
    for i, col in enumerate(feature_cols, 1):
        print(f'{i:2d}. {col}')

    print()
    print('=== AVAILABLE MACRO FEATURES IN WM_MACROS.CSV ===')
    all_data = loader.load_all_data()
    macro_data = all_data['macro_drivers_wm']
    print(f'Full macro data shape: {macro_data.shape}')
    print('All available columns:')
    for i, col in enumerate(macro_data.columns, 1):
        print(f'{i:2d}. {col}')

    print()
    print('=== PROJECTION DATA FEATURES ===')
    projection_data = loader.get_projection_data()
    print(f'Projection data shape: {projection_data.shape}')
    print('Features available for forecasting:')
    proj_cols = [col for col in projection_data.columns if col != 'Year']
    for i, col in enumerate(proj_cols, 1):
        print(f'{i:2d}. {col}')

if __name__ == "__main__":
    main()