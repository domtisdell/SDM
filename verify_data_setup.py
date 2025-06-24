#!/usr/bin/env python3
"""
Verify that all required data files are accessible in the self-contained ML_model directory.
"""

import sys
from pathlib import Path
import pandas as pd

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def verify_data_files():
    """Verify all data files exist and are readable."""
    print("Verifying data files in self-contained ML_model directory...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        
        # Initialize data loader
        loader = SteelDemandDataLoader("config/")
        
        # Load configuration
        data_sources = loader.config.get('data_sources')
        if data_sources is None:
            print("ERROR: Could not load data sources configuration")
            return False
        
        print(f"Found {len(data_sources)} data source definitions")
        
        # Check each data source
        missing_files = []
        accessible_files = []
        
        for _, source in data_sources.iterrows():
            source_name = source['source_name']
            file_path = source['file_path']
            
            # Check if file exists
            full_path = current_dir / file_path
            
            if full_path.exists():
                try:
                    # Try to read the file
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(full_path)
                        print(f"✓ {source_name}: {df.shape} - {file_path}")
                        accessible_files.append(source_name)
                    else:
                        print(f"✓ {source_name}: Config file - {file_path}")
                        accessible_files.append(source_name)
                except Exception as e:
                    print(f"✗ {source_name}: File exists but cannot read - {str(e)}")
                    missing_files.append(source_name)
            else:
                print(f"✗ {source_name}: File not found - {file_path}")
                missing_files.append(source_name)
        
        print(f"\nSummary:")
        print(f"Accessible files: {len(accessible_files)}")
        print(f"Missing/problematic files: {len(missing_files)}")
        
        if missing_files:
            print(f"Issues with: {missing_files}")
            return False
        else:
            print("✓ All data files are accessible!")
            return True
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

def test_data_loading():
    """Test the data loading functionality."""
    print("\nTesting data loading functionality...")
    
    try:
        from data.data_loader import SteelDemandDataLoader
        
        # Initialize and load data
        loader = SteelDemandDataLoader("config/")
        data = loader.load_all_data()
        
        print(f"Successfully loaded {len(data)} datasets:")
        for dataset_name, df in data.items():
            if hasattr(df, 'shape'):
                print(f"  - {dataset_name}: {df.shape}")
            else:
                print(f"  - {dataset_name}: Configuration data")
        
        # Test historical data consolidation
        historical = loader.get_historical_data()
        print(f"\nConsolidated historical data: {historical.shape}")
        
        if 'Year' in historical.columns:
            print(f"Year range: {historical['Year'].min()}-{historical['Year'].max()}")
        
        # Check for steel consumption columns
        steel_cols = [col for col in historical.columns if col.endswith('_tonnes')]
        print(f"Steel consumption categories: {len(steel_cols)}")
        for col in steel_cols:
            print(f"  - {col}")
        
        # Test projection data
        projections = loader.get_projection_data()
        print(f"\nProjection data: {projections.shape}")
        if 'Year' in projections.columns:
            print(f"Projection years: {projections['Year'].min()}-{projections['Year'].max()}")
        
        return True
        
    except Exception as e:
        print(f"ERROR in data loading test: {str(e)}")
        return False

def main():
    """Run verification tests."""
    print("="*60)
    print("ML_model Data Setup Verification")
    print("="*60)
    
    # Check working directory
    print(f"Working directory: {current_dir}")
    print(f"Data directory exists: {(current_dir / 'data').exists()}")
    print(f"Config directory exists: {(current_dir / 'config').exists()}")
    
    # Verify files
    files_ok = verify_data_files()
    
    # Test loading
    loading_ok = test_data_loading() if files_ok else False
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if files_ok and loading_ok:
        print("✓ SUCCESS: ML_model is fully self-contained and ready to use!")
        print("\nTo train models, run:")
        print("  python train_steel_demand_models.py")
        return 0
    else:
        print("✗ FAILED: Some issues found with data setup")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)