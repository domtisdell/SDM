#!/usr/bin/env python3
"""
Verify that all required packages can be imported successfully.
Run this script to check if the requirements are properly installed.
"""

import sys
from typing import List, Tuple

def test_imports() -> List[Tuple[str, bool, str]]:
    """Test importing all critical packages."""
    test_results = []
    
    # Core data science
    packages = [
        ("pandas", "import pandas"),
        ("numpy", "import numpy"),
        ("scipy", "import scipy"),
        ("sklearn", "import sklearn"),
        ("joblib", "import joblib"),
        
        # ML frameworks
        ("xgboost", "import xgboost"),
        ("lightgbm", "import lightgbm"),
        ("catboost", "import catboost"),
        
        # Deep learning
        ("tensorflow", "import tensorflow"),
        ("keras", "import keras"),
        
        # Time series
        ("prophet", "import prophet"),
        ("statsmodels", "import statsmodels"),
        
        # Feature engineering
        ("category_encoders", "import category_encoders"),
        ("feature_engine", "import feature_engine"),
        
        # Model interpretation
        ("shap", "import shap"),
        ("lime", "import lime"),
        
        # Visualization
        ("matplotlib", "import matplotlib"),
        ("seaborn", "import seaborn"),
        ("plotly", "import plotly"),
        
        # API
        ("fastapi", "import fastapi"),
        ("pydantic", "import pydantic"),
        
        # Testing
        ("pytest", "import pytest"),
    ]
    
    for package_name, import_statement in packages:
        try:
            exec(import_statement)
            test_results.append((package_name, True, "OK"))
        except ImportError as e:
            test_results.append((package_name, False, str(e)))
        except Exception as e:
            test_results.append((package_name, False, f"Unexpected error: {str(e)}"))
    
    return test_results

def main():
    """Main verification function."""
    print("🔍 Verifying SDM Requirements Installation")
    print("=" * 50)
    
    results = test_imports()
    
    success_count = 0
    failure_count = 0
    
    for package, success, message in results:
        status = "✅" if success else "❌"
        print(f"{status} {package:20} - {message}")
        
        if success:
            success_count += 1
        else:
            failure_count += 1
    
    print("=" * 50)
    print(f"📊 Results: {success_count} successful, {failure_count} failed")
    
    if failure_count == 0:
        print("🎉 All packages imported successfully!")
        print("✅ Requirements installation verified")
        return 0
    else:
        print("⚠️  Some packages failed to import")
        print("💡 Try reinstalling with: pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())