#!/usr/bin/env python3
"""Debug script to check railway track values."""

import pandas as pd

# Load ensemble forecasts with different methods
print("Loading ensemble forecasts...")
ensemble_df = pd.read_csv('outputs/track_a/Ensemble_Forecasts_2025-2050.csv')

# Check data types
print("\nData types:")
print(ensemble_df.dtypes)

# Show first few rows for relevant columns
print("\nFirst 5 rows of relevant columns:")
cols = ['Year', 'Production of Hot Rolled Long Products_Ensemble', 
        'Production of Wire Rod_Ensemble', 'Production of Railway Track Material_Ensemble']
print(ensemble_df[cols].head())

# Check for any NaN values
print("\nNaN values in these columns:")
print(ensemble_df[cols].isna().sum())

# Check raw values using iloc
print("\nRaw values using iloc for row 0 (2024):")
print(f"Year: {ensemble_df.iloc[0, 0]}")
print(f"Hot Rolled Long: {ensemble_df.iloc[0, 10]}")  
print(f"Wire Rod: {ensemble_df.iloc[0, 16]}")
print(f"Railway Track: {ensemble_df.iloc[0, 17]}")

# Check column names
print("\nColumn names at positions:")
print(f"Column 10: {ensemble_df.columns[10]}")
print(f"Column 16: {ensemble_df.columns[16]}")
print(f"Column 17: {ensemble_df.columns[17]}")

# Let's also check the ML forecasts
print("\n" + "="*80)
print("Loading ML forecasts...")
ml_df = pd.read_csv('outputs/track_a/ML_Algorithm_Forecasts_2025-2050.csv')

# Find railway track columns
rail_cols = [col for col in ml_df.columns if 'Railway Track' in col]
print(f"\nRailway Track columns found: {rail_cols}")

# Show values
print("\nFirst 5 rows of Railway Track predictions:")
print(ml_df[['Year'] + rail_cols].head())