#!/usr/bin/env python3
"""Test script to understand railway track ensemble issue."""

import pandas as pd
import numpy as np

# Load the forecast files
ml_forecasts = pd.read_csv('outputs/track_a/ML_Algorithm_Forecasts_2025-2050.csv')
ensemble_forecasts = pd.read_csv('outputs/track_a/Ensemble_Forecasts_2025-2050.csv')

# Extract relevant columns for 2025
year_2025_idx = 1  # Index for year 2025

# Hot Rolled Long Products
hrlong_xgb = ml_forecasts.loc[year_2025_idx, 'Production of Hot Rolled Long Products_XGBoost']
hrlong_rf = ml_forecasts.loc[year_2025_idx, 'Production of Hot Rolled Long Products_RandomForest']
hrlong_ensemble_before = np.mean([hrlong_xgb, hrlong_rf])

# Wire Rod
wire_xgb = ml_forecasts.loc[year_2025_idx, 'Production of Wire Rod_XGBoost'] 
wire_rf = ml_forecasts.loc[year_2025_idx, 'Production of Wire Rod_RandomForest']
wire_ensemble_before = np.mean([wire_xgb, wire_rf])

# Railway Track
rail_xgb = ml_forecasts.loc[year_2025_idx, 'Production of Railway Track Material_XGBoost']
rail_rf = ml_forecasts.loc[year_2025_idx, 'Production of Railway Track Material_RandomForest']
rail_ensemble_before = np.mean([rail_xgb, rail_rf])

# Final ensemble values
hrlong_ensemble_after = ensemble_forecasts.loc[year_2025_idx, 'Production of Hot Rolled Long Products_Ensemble']
wire_ensemble_after = ensemble_forecasts.loc[year_2025_idx, 'Production of Wire Rod_Ensemble']
rail_ensemble_after = ensemble_forecasts.loc[year_2025_idx, 'Production of Railway Track Material_Ensemble']

print("2025 Forecast Analysis:")
print("=" * 80)
print("\nIndividual Model Predictions:")
print(f"Hot Rolled Long Products - XGBoost: {hrlong_xgb:,.2f}, RF: {hrlong_rf:,.2f}")
print(f"Wire Rod - XGBoost: {wire_xgb:,.2f}, RF: {wire_rf:,.2f}")
print(f"Railway Track - XGBoost: {rail_xgb:,.2f}, RF: {rail_rf:,.2f}")

print("\nSimple Average (before hierarchical adjustment):")
print(f"Hot Rolled Long Products: {hrlong_ensemble_before:,.2f}")
print(f"Wire Rod: {wire_ensemble_before:,.2f}")
print(f"Railway Track: {rail_ensemble_before:,.2f}")
print(f"Sum of children: {wire_ensemble_before + rail_ensemble_before:,.2f}")
print(f"Ratio (children/parent): {(wire_ensemble_before + rail_ensemble_before) / hrlong_ensemble_before:.2f}")

print("\nFinal Ensemble (after hierarchical adjustment):")
print(f"Hot Rolled Long Products: {hrlong_ensemble_after:,.2f}")
print(f"Wire Rod: {wire_ensemble_after:,.2f}")
print(f"Railway Track: {rail_ensemble_after:,.2f}")
print(f"Sum of children: {wire_ensemble_after + rail_ensemble_after:,.2f}")
print(f"Ratio (children/parent): {(wire_ensemble_after + rail_ensemble_after) / hrlong_ensemble_after:.2f}")

print("\nExpected Adjustment:")
if abs((wire_ensemble_before + rail_ensemble_before) - hrlong_ensemble_before) > hrlong_ensemble_before * 0.05:
    print("Children sum differs from parent by >5%, should have been adjusted")
    adjustment_factor = hrlong_ensemble_before / (wire_ensemble_before + rail_ensemble_before)
    print(f"Expected adjustment factor: {adjustment_factor:.4f}")
    print(f"Expected adjusted Wire Rod: {wire_ensemble_before * adjustment_factor:,.2f}")
    print(f"Expected adjusted Railway Track: {rail_ensemble_before * adjustment_factor:,.2f}")
else:
    print("Children sum within 5% tolerance, no adjustment needed")

print("\nActual vs Expected:")
print(f"Railway Track - Actual: {rail_ensemble_after:,.2f}, Expected: {rail_ensemble_before * (hrlong_ensemble_before / (wire_ensemble_before + rail_ensemble_before)):,.2f}")