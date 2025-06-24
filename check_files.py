#!/usr/bin/env python3
"""
Simple file check without external dependencies.
Verifies all required CSV files are present in the self-contained ML_model directory.
"""

from pathlib import Path

def check_files():
    """Check that all required files exist."""
    print("Checking self-contained ML_model data files...")
    
    base_dir = Path(__file__).parent
    
    required_files = [
        # WSA Steel data
        "data/worldsteelassoc/WSA_Au_2004-2023.csv",
        
        # WM Macro drivers
        "data/macro_drivers/WM_macros.csv",
        
        # Reference forecasts
        "data/reference_forecasts/Steel_Consumption_Forecasts_2025-2050.csv",
        
        # Configuration files
        "config/model_config.csv",
        "config/data_sources.csv",
        "config/steel_categories.csv", 
        "config/economic_indicators.csv",
        "config/regional_adjustment_factors.csv",
        "config/validation_benchmarks.csv",
        
        # Core modules
        "data/data_loader.py",
        "data/feature_engineering.py",
        "models/ensemble_models.py",
        "training/model_trainer.py",
        "train_steel_demand_models.py"
    ]
    
    missing_files = []
    present_files = []
    file_sizes = {}
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            present_files.append(file_path)
            if full_path.is_file():
                size_kb = full_path.stat().st_size / 1024
                file_sizes[file_path] = f"{size_kb:.1f} KB"
            else:
                file_sizes[file_path] = "Directory"
        else:
            missing_files.append(file_path)
    
    print(f"\nFile Check Results:")
    print(f"Total files checked: {len(required_files)}")
    print(f"Present files: {len(present_files)}")
    print(f"Missing files: {len(missing_files)}")
    
    if present_files:
        print(f"\n✓ Present files:")
        for file_path in present_files:
            size_info = file_sizes.get(file_path, "")
            print(f"  {file_path} ({size_info})")
    
    if missing_files:
        print(f"\n✗ Missing files:")
        for file_path in missing_files:
            print(f"  {file_path}")
        return False
    else:
        print(f"\n✓ SUCCESS: All required files are present!")
        print(f"\nThe ML_model directory is fully self-contained and ready to use.")
        print(f"To train models (after installing dependencies):")
        print(f"  cd {base_dir}")
        print(f"  pip install -r requirements.txt")
        print(f"  python train_steel_demand_models.py")
        return True

if __name__ == "__main__":
    success = check_files()
    exit(0 if success else 1)