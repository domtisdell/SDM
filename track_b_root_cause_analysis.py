"""
Track B Root Cause Analysis - Detailed Technical Analysis
Identifies the exact code sections and calculations that create the divergence.
"""

import pandas as pd
import numpy as np
import sys

# Add parent directories to path for imports
sys.path.append('.')
sys.path.append('..')

from data.hierarchical_features import HierarchicalProductFeatures

def demonstrate_root_cause():
    """
    Demonstrate the exact root cause of Track B divergence with specific code references.
    """
    print("=== TRACK B ROOT CAUSE ANALYSIS ===")
    print("Technical analysis of the declining steel demand pattern\n")
    
    # Initialize hierarchical features
    hierarchical_features = HierarchicalProductFeatures('config/')
    
    # 1. Show the sectoral weights configuration
    print("1. SECTORAL WEIGHTS CONFIGURATION")
    print("=" * 60)
    print("File: config/sectoral_weights.csv")
    sectoral_weights = pd.read_csv('config/sectoral_weights.csv')
    print(sectoral_weights.to_string(index=False))
    
    # 2. Show the weight application logic
    print("\n2. WEIGHT APPLICATION LOGIC")
    print("=" * 60)
    print("File: data/hierarchical_features.py, lines 43-67")
    print("Function: get_sectoral_weights_for_year()")
    print("""
    def get_sectoral_weights_for_year(self, year: int) -> Dict[str, float]:
        if year <= 2030:
            period = "2025-2030"
        elif year <= 2040:
            period = "2031-2040" 
        else:
            period = "2041-2050"
            
        period_weights = self.sectoral_weights[self.sectoral_weights['period'] == period].iloc[0]
        
        return {
            'gdp_construction': period_weights['gdp_construction'],
            'infrastructure_traditional': period_weights['infrastructure_traditional'],
            'manufacturing_ip': period_weights['manufacturing_ip'],
            'wm_renewable_energy': period_weights['wm_renewable_energy']
        }
    """)
    
    # 3. Demonstrate the weight changes
    print("\n3. WEIGHT CHANGES DEMONSTRATION")
    print("=" * 60)
    
    test_years = [2030, 2031, 2040, 2041, 2050]
    for year in test_years:
        weights = hierarchical_features.get_sectoral_weights_for_year(year)
        print(f"Year {year}: Construction weight = {weights['gdp_construction']:.2f}")
    
    # 4. Show the calculation in forecasting
    print("\n4. FORECASTING CALCULATION")
    print("=" * 60)
    print("File: forecasting/hierarchical_forecasting.py, lines 177-194")
    print("Function: forecast_level_0_total_demand()")
    print("""
    # Apply time-varying sectoral weights and combine
    for idx, (i, row) in enumerate(level_0_forecasts.iterrows()):
        year = row['Year']
        sectoral_weights = self.hierarchical_features.get_sectoral_weights_for_year(year)
        
        # Calculate total demand as weighted sum of sectors
        total_demand = 0.0
        sector_breakdown = {}
        
        for sector, weight in sectoral_weights.items():
            sector_key = sector.replace('gdp_', '').replace('wm_', '').replace('_ip', '')
            if sector_key in sectoral_forecasts:
                sector_demand = sectoral_forecasts[sector_key][idx] * weight  # ← KEY LINE
                total_demand += sector_demand
                sector_breakdown[sector_key] = sector_demand
    """)
    
    # 5. Show the construction intensity calculation
    print("\n5. CONSTRUCTION INTENSITY CALCULATION")
    print("=" * 60)
    print("File: data/hierarchical_features.py, line 170")
    print("Function: create_hierarchical_features()")
    print("""
    features['construction_intensity_gdp'] = features.get('GDP_AUD_Real2015', 0) * 4.07
    """)
    
    # 6. Demonstrate the actual calculation
    print("\n6. ACTUAL CALCULATION DEMONSTRATION")
    print("=" * 60)
    
    # Load WM macro data
    wm_data = pd.read_csv('data/macro_drivers/WM_macros.csv')
    
    # Calculate construction intensity for 2030 and 2031
    gdp_2030 = wm_data[wm_data['Year'] == 2030]['GDP_AUD_Real2015'].iloc[0]
    gdp_2031 = wm_data[wm_data['Year'] == 2031]['GDP_AUD_Real2015'].iloc[0]
    
    construction_intensity_2030 = gdp_2030 * 4.07
    construction_intensity_2031 = gdp_2031 * 4.07
    
    weight_2030 = 0.45  # 2025-2030 period
    weight_2031 = 0.35  # 2031-2040 period
    
    construction_demand_2030 = construction_intensity_2030 * weight_2030
    construction_demand_2031 = construction_intensity_2031 * weight_2031
    
    print(f"2030 Calculation:")
    print(f"  GDP: {gdp_2030:,.1f} AUD billion")
    print(f"  Construction Intensity: {gdp_2030:.1f} * 4.07 = {construction_intensity_2030:,.1f} kt")
    print(f"  Construction Weight: {weight_2030:.2f}")
    print(f"  Construction Demand: {construction_intensity_2030:,.1f} * {weight_2030:.2f} = {construction_demand_2030:,.1f} kt")
    
    print(f"\n2031 Calculation:")
    print(f"  GDP: {gdp_2031:,.1f} AUD billion")
    print(f"  Construction Intensity: {gdp_2031:.1f} * 4.07 = {construction_intensity_2031:,.1f} kt")
    print(f"  Construction Weight: {weight_2031:.2f}")
    print(f"  Construction Demand: {construction_intensity_2031:,.1f} * {weight_2031:.2f} = {construction_demand_2031:,.1f} kt")
    
    drop_absolute = construction_demand_2031 - construction_demand_2030
    drop_percentage = (drop_absolute / construction_demand_2030) * 100
    
    print(f"\nDrop in Construction Demand: {drop_absolute:,.1f} kt ({drop_percentage:.1f}%)")
    
    # 7. Show the fundamental problem
    print("\n7. FUNDAMENTAL PROBLEM ANALYSIS")
    print("=" * 60)
    
    print("ROOT CAUSE IDENTIFIED:")
    print("1. Track B assumes construction's share of steel demand will decline dramatically")
    print("2. From 45% (2025-2030) to 35% (2031-2040) to 30% (2041-2050)")
    print("3. This is applied as a multiplicative factor to construction intensity")
    print("4. Even though GDP continues to grow, construction steel demand is artificially reduced")
    print("5. The weight reduction causes the sharp drop in 2031 and continued decline")
    
    print("\nSPECIFIC CODE LOCATIONS:")
    print("- Sectoral weights defined in: config/sectoral_weights.csv")
    print("- Weight application logic: data/hierarchical_features.py:43-67")
    print("- Construction intensity calculation: data/hierarchical_features.py:170")
    print("- Forecasting calculation: forecasting/hierarchical_forecasting.py:177-194")
    
    print("\nCONTRAST WITH TRACK A:")
    print("- Track A directly forecasts steel categories using historical steel consumption")
    print("- Track A doesn't apply artificial sectoral weight reductions")
    print("- Track A produces stable forecasts aligned with economic growth")
    print("- Track A's renewable energy integration is additive, not substitutive")
    
    print("\nRECOMMENDATION:")
    print("- Track A methodology is more reliable for Australian steel demand forecasting")
    print("- Track B's hierarchical approach introduces unnecessary complexity")
    print("- The declining construction weight assumption is not well-justified")
    print("- For production planning, Track A forecasts should be preferred")
    
    print("\n=== ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    demonstrate_root_cause()