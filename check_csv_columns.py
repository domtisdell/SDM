#!/usr/bin/env python3
"""Check CSV file integrity."""

import subprocess
import pandas as pd

csv_file = 'outputs/track_a/Ensemble_Forecasts_2025-2050.csv'

# Get header using bash
bash_cmd = f"head -n 1 {csv_file}"
result = subprocess.run(bash_cmd, shell=True, capture_output=True, text=True)
bash_header = result.stdout.strip().split(',')

# Get header using pandas
df = pd.read_csv(csv_file)
pandas_header = df.columns.tolist()

print(f"Bash header count: {len(bash_header)}")
print(f"Pandas header count: {len(pandas_header)}")

# Find Railway Track column
bash_rail_idx = [i for i, col in enumerate(bash_header) if 'Railway Track' in col]
pandas_rail_idx = [i for i, col in enumerate(pandas_header) if 'Railway Track' in col]

print(f"\nBash Railway Track index: {bash_rail_idx}")
print(f"Pandas Railway Track index: {pandas_rail_idx}")

# Compare specific columns
print("\nColumn comparison (15-20):")
print(f"{'Idx':<4} {'Bash':<50} {'Pandas':<50}")
print("-" * 105)
for i in range(15, 21):
    bash_col = bash_header[i] if i < len(bash_header) else "N/A"
    pandas_col = pandas_header[i] if i < len(pandas_header) else "N/A"
    print(f"{i:<4} {bash_col:<50} {pandas_col:<50}")

# Get specific values using bash
print("\n\nGetting row 2 (year 2025) values using bash:")
bash_cmd = f"sed -n '3p' {csv_file}"
result = subprocess.run(bash_cmd, shell=True, capture_output=True, text=True)
bash_values = result.stdout.strip().split(',')

if bash_rail_idx:
    idx = bash_rail_idx[0]
    print(f"Railway Track (column {idx}): {bash_values[idx]}")
    
# Compare with pandas
print("\nGetting same row using pandas:")
rail_col = 'Production of Railway Track Material_Ensemble'
print(f"Railway Track: {df.loc[1, rail_col]}")

# Check file size and modification time
import os
stat = os.stat(csv_file)
print(f"\nFile info:")
print(f"Size: {stat.st_size} bytes")
print(f"Modified: {stat.st_mtime}")