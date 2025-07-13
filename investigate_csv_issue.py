#!/usr/bin/env python3
"""Investigate the CSV loading discrepancy."""

import pandas as pd
import csv

csv_file = 'outputs/track_a/Ensemble_Forecasts_2025-2050.csv'

print("Method 1: Using pandas read_csv")
df = pd.read_csv(csv_file)
rail_col = 'Production of Railway Track Material_Ensemble'
print(f"Railway Track values (first 3): {df[rail_col].head(3).tolist()}")

print("\nMethod 2: Using csv module")
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i < 3:
            print(f"Year {row['Year']}: Railway Track = {row[rail_col]}")
        else:
            break

print("\nMethod 3: Manual parsing")
with open(csv_file, 'r') as f:
    lines = f.readlines()
    header = lines[0].strip().split(',')
    rail_idx = header.index(rail_col)
    print(f"Railway Track column index: {rail_idx}")
    for i in range(1, 4):
        values = lines[i].strip().split(',')
        print(f"Row {i}: Railway Track = {values[rail_idx]}")

print("\nColumn analysis:")
print(f"Total columns in header: {len(header)}")
print(f"Columns 15-20:")
for i in range(15, min(21, len(header))):
    print(f"  {i}: {header[i]}")

# Check if there are duplicate column names
print(f"\nChecking for duplicate column names...")
rail_cols = [i for i, col in enumerate(header) if 'Railway Track' in col]
print(f"Columns containing 'Railway Track': {rail_cols}")
for idx in rail_cols:
    print(f"  Column {idx}: {header[idx]}")