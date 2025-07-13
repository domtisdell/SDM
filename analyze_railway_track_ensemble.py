#!/usr/bin/env python3
"""Analyze the railway track ensemble forecast issue."""

import pandas as pd
import numpy as np

# Load the data
ml_df = pd.read_csv('outputs/track_a/ML_Algorithm_Forecasts_2025-2050.csv')
ensemble_df = pd.read_csv('outputs/track_a/Ensemble_Forecasts_2025-2050.csv')

print("Railway Track Ensemble Forecast Analysis")
print("=" * 80)

# Get the correct column names
hrlong_col = 'Production of Hot Rolled Long Products_Ensemble'
wire_col = 'Production of Wire Rod_Ensemble'  
rail_col = 'Production of Railway Track Material_Ensemble'

# Analyze year 2025 (index 1)
year_idx = 1

print(f"\nYear 2025 Analysis:")
print("-" * 40)

# ML model predictions
rail_xgb = ml_df.loc[year_idx, 'Production of Railway Track Material_XGBoost']
rail_rf = ml_df.loc[year_idx, 'Production of Railway Track Material_RandomForest']
hrlong_xgb = ml_df.loc[year_idx, 'Production of Hot Rolled Long Products_XGBoost']
hrlong_rf = ml_df.loc[year_idx, 'Production of Hot Rolled Long Products_RandomForest']
wire_xgb = ml_df.loc[year_idx, 'Production of Wire Rod_XGBoost']
wire_rf = ml_df.loc[year_idx, 'Production of Wire Rod_RandomForest']

print(f"\nIndividual Model Predictions:")
print(f"  Hot Rolled Long - XGBoost: {hrlong_xgb:,.2f}, RF: {hrlong_rf:,.2f}")
print(f"  Wire Rod - XGBoost: {wire_xgb:,.2f}, RF: {wire_rf:,.2f}")  
print(f"  Railway Track - XGBoost: {rail_xgb:,.2f}, RF: {rail_rf:,.2f}")

# Simple averages (before hierarchical adjustment)
hrlong_avg = (hrlong_xgb + hrlong_rf) / 2
wire_avg = (wire_xgb + wire_rf) / 2
rail_avg = (rail_xgb + rail_rf) / 2

print(f"\nSimple Averages (before adjustment):")
print(f"  Hot Rolled Long: {hrlong_avg:,.2f}")
print(f"  Wire Rod: {wire_avg:,.2f}")
print(f"  Railway Track: {rail_avg:,.2f}")
print(f"  Sum of children: {wire_avg + rail_avg:,.2f}")

# Final ensemble values
hrlong_final = ensemble_df.loc[year_idx, hrlong_col]
wire_final = ensemble_df.loc[year_idx, wire_col]
rail_final = ensemble_df.loc[year_idx, rail_col]

print(f"\nFinal Ensemble Values:")
print(f"  Hot Rolled Long: {hrlong_final:,.2f}")
print(f"  Wire Rod: {wire_final:,.2f}")
print(f"  Railway Track: {rail_final:,.2f}")
print(f"  Sum of children: {wire_final + rail_final:,.2f}")

# Analysis
print(f"\nHierarchical Consistency Check:")
child_sum = wire_final + rail_final
diff_pct = abs(child_sum - hrlong_final) / hrlong_final * 100
print(f"  Parent-Child difference: {diff_pct:.2f}%")
if diff_pct > 5:
    print(f"  ❌ ISSUE: Children sum exceeds parent by {diff_pct:.2f}%")
    print(f"  Expected adjustment factor: {hrlong_final / (wire_avg + rail_avg):.4f}")
    print(f"  But railway track value suggests no adjustment was applied")
else:
    print(f"  ✓ Children sum matches parent (within 5% tolerance)")

# Check if values look like thousands separator was misread
print(f"\nPossible Data Format Issue:")
print(f"  Railway Track final value: {rail_final:,.2f}")
print(f"  If this was meant to be {rail_final/1000:.2f}, it would make more sense")
print(f"  Original average would be: {rail_avg:.2f}")

# Look at historical context
print(f"\nChecking all years for Railway Track pattern:")
print(f"{'Year':<6} {'XGBoost':<12} {'RF':<12} {'Ensemble':<12}")
print("-" * 45)
for i in range(min(5, len(ml_df))):
    year = ml_df.loc[i, 'Year']
    xgb = ml_df.loc[i, 'Production of Railway Track Material_XGBoost']
    rf = ml_df.loc[i, 'Production of Railway Track Material_RandomForest']
    ens = ensemble_df.loc[i, rail_col]
    print(f"{year:<6} {xgb:<12.2f} {rf:<12.2f} {ens:<12.2f}")