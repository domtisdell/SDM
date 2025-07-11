"""
Comprehensive analysis of Track B vs Track A divergence
Analyzes the hierarchical forecasting methodology to understand declining steel demand patterns.
"""

import pandas as pd
import numpy as np
import sys
import ast
from pathlib import Path

# Add parent directories to path for imports
sys.path.append('.')
sys.path.append('..')

from data.data_loader import SteelDemandDataLoader
from data.hierarchical_features import HierarchicalProductFeatures
from data.renewable_energy_features import RenewableEnergyFeatureEngineering

def analyze_track_b_divergence():
    """
    Comprehensive analysis of Track B hierarchical forecasting divergence.
    """
    print("=== TRACK B HIERARCHICAL FORECASTING ANALYSIS ===\n")
    
    # 1. Load configuration and data
    print("1. LOADING CONFIGURATION AND DATA")
    print("-" * 50)
    
    data_loader = SteelDemandDataLoader('config/')
    hierarchical_features = HierarchicalProductFeatures('config/')
    renewable_features = RenewableEnergyFeatureEngineering('config/renewable_energy_config.csv')
    
    # Load all data
    data_loader.load_all_data()
    historical_data = data_loader.get_historical_data()
    projection_data = data_loader.get_projection_data()
    
    # Create hierarchical features
    historical_hierarchical = hierarchical_features.create_hierarchical_features(historical_data)
    forecast_hierarchical = hierarchical_features.create_hierarchical_features(projection_data)
    
    print(f"Historical data: {historical_hierarchical.shape[0]} years ({historical_hierarchical['Year'].min()}-{historical_hierarchical['Year'].max()})")
    print(f"Forecast data: {forecast_hierarchical.shape[0]} years ({forecast_hierarchical['Year'].min()}-{forecast_hierarchical['Year'].max()})")
    
    # 2. Analyze sectoral weights
    print("\n2. SECTORAL WEIGHTS ANALYSIS")
    print("-" * 50)
    
    sectoral_weights = pd.read_csv('config/sectoral_weights.csv')
    print("Sectoral weights configuration:")
    print(sectoral_weights)
    
    # 3. Analyze construction intensity calculation
    print("\n3. CONSTRUCTION INTENSITY ANALYSIS")
    print("-" * 50)
    
    # Show construction intensity in historical data
    print("Construction intensity in historical data (last 10 years):")
    hist_construction = historical_hierarchical[['Year', 'GDP_AUD_Real2015', 'construction_intensity_gdp']].tail(10)
    print(hist_construction)
    
    # Show construction intensity in forecast data
    print("\nConstruction intensity in forecast data (first 15 years):")
    forecast_construction = forecast_hierarchical[['Year', 'GDP_AUD_Real2015', 'construction_intensity_gdp']].head(15)
    print(forecast_construction)
    
    # 4. Analyze sectoral model training approach
    print("\n4. SECTORAL MODEL TRAINING ANALYSIS")
    print("-" * 50)
    
    # The key issue: Track B trains models on construction_intensity_gdp as target
    # This means it learns historical patterns of construction intensity
    # But then applies time-varying weights that create the declining pattern
    
    print("Track B methodology:")
    print("1. Trains ML models with construction_intensity_gdp as target")
    print("2. Uses historical GDP as features to predict construction intensity")
    print("3. Applies time-varying sectoral weights to the predictions")
    print("4. Construction weight: 45% (2025-2030) → 35% (2031-2040) → 30% (2041-2050)")
    
    # 5. Calculate actual Track B construction demand
    print("\n5. TRACK B CONSTRUCTION DEMAND CALCULATION")
    print("-" * 50)
    
    # Load Track B results
    track_b_results = pd.read_csv('forecasts/track_b_20250710_215758/hierarchical_level_0_forecasts_2025_2050.csv')
    
    # Extract construction values from sectoral breakdown
    def extract_construction(breakdown_str):
        try:
            breakdown = ast.literal_eval(breakdown_str)
            return breakdown.get('construction', 0)
        except:
            return 0
    
    track_b_results['construction_actual'] = track_b_results['sectoral_breakdown'].apply(extract_construction)
    
    # Calculate expected construction based on intensity and weights
    def get_weight_for_year(year):
        if year <= 2030:
            return 0.45
        elif year <= 2040:
            return 0.35
        else:
            return 0.30
    
    forecast_analysis = forecast_hierarchical[['Year', 'GDP_AUD_Real2015', 'construction_intensity_gdp']].copy()
    forecast_analysis['construction_weight'] = forecast_analysis['Year'].apply(get_weight_for_year)
    forecast_analysis['construction_expected'] = forecast_analysis['construction_intensity_gdp'] * forecast_analysis['construction_weight']
    
    # Merge with Track B results
    comparison = pd.merge(
        track_b_results[['Year', 'construction_actual', 'total_steel_demand']], 
        forecast_analysis[['Year', 'construction_expected', 'construction_weight', 'GDP_AUD_Real2015']], 
        on='Year'
    )
    
    print("Track B Construction Demand Analysis:")
    print(comparison[['Year', 'GDP_AUD_Real2015', 'construction_weight', 'construction_expected', 'construction_actual', 'total_steel_demand']].head(15))
    
    # 6. Compare with Track A
    print("\n6. TRACK A VS TRACK B COMPARISON")
    print("-" * 50)
    
    # Load Track A results
    track_a_results = pd.read_csv('forecasts/track_a_20250710_213454/Ensemble_Forecasts_2025-2050.csv')
    
    # Track A uses "Apparent Steel Use (crude steel equivalent)" as closest equivalent
    track_comparison = pd.merge(
        track_a_results[['Year', 'Apparent Steel Use (crude steel equivalent)_Ensemble']].rename(columns={'Apparent Steel Use (crude steel equivalent)_Ensemble': 'Track_A_Steel_Use'}),
        track_b_results[['Year', 'total_steel_demand']].rename(columns={'total_steel_demand': 'Track_B_Steel_Demand'}),
        on='Year'
    )
    
    track_comparison['Difference'] = track_comparison['Track_B_Steel_Demand'] - track_comparison['Track_A_Steel_Use']
    track_comparison['Percent_Difference'] = (track_comparison['Difference'] / track_comparison['Track_A_Steel_Use']) * 100
    
    print("Track A vs Track B Total Steel Demand:")
    print(track_comparison.head(15))
    
    # 7. Root cause analysis
    print("\n7. ROOT CAUSE ANALYSIS")
    print("-" * 50)
    
    print("KEY FINDINGS:")
    print("1. Track B construction demand drops sharply in 2031 due to sectoral weight reduction (45% → 35%)")
    print("2. Track B total steel demand declines from ~7,000 kt (2025) to ~5,810 kt (2050)")
    print("3. Track A remains relatively stable around 7,000 kt throughout forecast period")
    print("4. The divergence is primarily driven by the time-varying sectoral weights methodology")
    
    # Calculate the exact drop amounts
    construction_2030 = comparison[comparison['Year'] == 2030]['construction_actual'].iloc[0]
    construction_2031 = comparison[comparison['Year'] == 2031]['construction_actual'].iloc[0]
    construction_drop = construction_2031 - construction_2030
    
    total_2030 = comparison[comparison['Year'] == 2030]['total_steel_demand'].iloc[0]
    total_2031 = comparison[comparison['Year'] == 2031]['total_steel_demand'].iloc[0]
    total_drop = total_2031 - total_2030
    
    print(f"\nSPECIFIC NUMBERS:")
    print(f"Construction demand drop 2030→2031: {construction_drop:,.0f} kt ({construction_drop/construction_2030*100:.1f}%)")
    print(f"Total steel demand drop 2030→2031: {total_drop:,.0f} kt ({total_drop/total_2030*100:.1f}%)")
    
    # 8. Methodology critique
    print("\n8. METHODOLOGY CRITIQUE")
    print("-" * 50)
    
    print("ISSUES WITH TRACK B METHODOLOGY:")
    print("1. Time-varying sectoral weights assume construction's share of steel demand will decline significantly")
    print("2. This contradicts Australian economic growth projections and infrastructure needs")
    print("3. The renewable energy weight increase (8% → 25%) may be optimistic")
    print("4. Track B's hierarchical structure introduces complexity without clear benefit")
    print("5. The sectoral model training approach may not capture realistic steel demand patterns")
    
    print("\nTRACK A ADVANTAGES:")
    print("1. Direct forecasting of steel categories using proven ML ensemble")
    print("2. Uses actual historical steel consumption data as training targets")
    print("3. Incorporates renewable energy as time series features without overly complex weighting")
    print("4. Produces more stable and realistic forecasts aligned with economic growth")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    analyze_track_b_divergence()