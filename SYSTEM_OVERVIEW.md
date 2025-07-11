# Australian Steel Demand Model - System Overview

## Executive Summary

The Australian Steel Demand Model (SDM) is a comprehensive forecasting platform that combines machine learning, econometric modeling, and hierarchical forecasting to predict Australian steel consumption through 2050. The system implements a dual-track approach with renewable energy integration and automatic timestamped output management.

## 🎯 **Key Capabilities**

### **Dual-Track Forecasting**
- **Track A**: Traditional approach with renewable energy time series integration
- **Track B**: Hierarchical product taxonomy with consistency constraints
- **Unified Results**: Combined forecasts with performance comparison and validation

### **Advanced ML Ensemble**
- XGBoost (45%), Random Forest (40%), Linear Regression (15%)
- LSTM and Prophet models for time series analysis
- Cross-validation with uncertainty quantification
- Bootstrap sampling for confidence intervals

### **Renewable Energy Integration**
- Direct integration of wind and solar capacity time series
- Steel intensity mapping for infrastructure demand
- No complex feature engineering - raw data used as model inputs
- Alignment with Australia's renewable energy targets

### **Robust Output Management**
- Automatic timestamped directories prevent overwrites
- Comprehensive logging and audit trails
- Structured CSV outputs with performance metrics
- Option to disable timestamping for development

## 🏗️ **System Architecture**

### **Data Layer**
```
data/
├── WSA_Au_2004-2023.csv              # Steel consumption data (2004-2023)
├── WM_macros.csv                     # Economic + renewable time series (2004-2050)
└── Steel_Consumption_Forecasts_2025-2050.csv  # Reference forecasts
```

### **Configuration Layer (13 CSV Files)**
```
config/
├── Core Configuration
│   ├── model_config.csv              # ML hyperparameters
│   ├── data_sources.csv              # Data file locations
│   ├── steel_categories.csv          # Product definitions
│   ├── economic_indicators.csv       # Economic driver definitions
│   └── validation_benchmarks.csv     # Performance targets
├── Renewable Energy
│   ├── renewable_steel_intensity.csv # Steel intensities by technology
│   └── renewable_technology_mapping.csv # Technology definitions
├── Hierarchical Structure  
│   ├── hierarchical_steel_products_L1.csv # Level 1 categories
│   ├── hierarchical_steel_products_L2.csv # Level 2 families
│   ├── hierarchical_steel_products_L3.csv # Level 3 products
│   └── sectoral_steel_mapping.csv    # Sector-to-product mapping
└── Regional & Validation
    ├── regional_adjustment_factors.csv # State-level adjustments
    └── end_use_sector_mapping.csv    # End-use definitions
```

### **Processing Layer**
```
Core Components:
├── data/data_loader.py               # Data loading and validation
├── data/feature_engineering.py      # Feature creation and preprocessing
├── models/ensemble_models.py         # ML model implementations
├── training/model_trainer.py        # Training and cross-validation
├── forecasting/
│   ├── dual_track_forecasting.py    # Track A implementation
│   └── hierarchical_forecasting.py  # Track B implementation
└── evaluation/                      # Uncertainty quantification
```

### **Output Layer**
```
forecasts/
├── hierarchical_run_YYYYMMDD_HHMMSS/ # Dual-track results
├── training_YYYYMMDD_HHMMSS/         # ML ensemble training
├── regression_ml_YYYYMMDD_HHMMSS/    # Simplified ML algorithms
└── [historical timestamped directories] # Previous runs
```

## 📊 **Data Flow Architecture**

### **Track A (Traditional + Renewable)**
```
WSA Steel Data (2004-2023) 
    ↓
WM Macro Data (Economic + Renewable 2004-2050)
    ↓
Direct Integration (Wind_Onshore, Wind_Offshore, Solar_Grid, Solar_Distributed)
    ↓
ML Ensemble Training (XGBoost, RF, Linear Regression)
    ↓
Forecasts 2025-2050 with Uncertainty Quantification
```

### **Track B (Hierarchical)**
```
All Data Sources + Configuration
    ↓
Enhanced Feature Engineering with Renewable Integration
    ↓
Multi-Level Model Training (Level 0-3)
    ↓
Consistency Constraints and Aggregation
    ↓
Hierarchical Forecasts with Cross-Level Validation
```

### **Unified Output**
```
Track A Results + Track B Results
    ↓
Performance Comparison and Validation
    ↓
Combined Forecasts with Uncertainty Bounds
    ↓
Timestamped Output Directory with Comprehensive Logging
```

## 🎛️ **Key Configuration Parameters**

### **Model Configuration (model_config.csv)**
| Parameter | Value | Description |
|-----------|-------|-------------|
| train_test_split | 0.8 | Training data percentage |
| ensemble_weights_xgb | 0.45 | XGBoost weight in ensemble |
| ensemble_weights_rf | 0.40 | Random Forest weight |
| ensemble_weights_regression | 0.15 | Linear Regression weight |
| bootstrap_samples | 500 | Bootstrap iterations for uncertainty |
| target_mape | 4.0 | Target MAPE percentage |
| min_r2_score | 0.85 | Minimum R² threshold |

### **Renewable Energy Integration**
| Technology | Steel Intensity | Data Column |
|------------|----------------|-------------|
| Wind Onshore | 164 tonnes/MW | Wind_Onshore |
| Wind Offshore | 220 tonnes/MW | Wind_Offshore |
| Solar Grid | 35 tonnes/MW | Solar_Grid |
| Solar Distributed | 8 tonnes/MW | Solar_Distributed |

## 🚀 **Execution Commands**

### **Primary Commands**
```bash
# Dual-track forecasting (RECOMMENDED)
python run_hierarchical_forecasting.py

# Traditional ML ensemble only
python train_steel_demand_models.py

# Simplified ML comparison
python train_regression_ml_models.py

# Data verification
python verify_data_setup.py
```

### **Advanced Options**
```bash
# Custom timestamped output
python run_hierarchical_forecasting.py --output-dir production_run

# Disable timestamping for development
python run_hierarchical_forecasting.py --output-dir test --no-timestamp

# Specific time periods
python train_steel_demand_models.py --start-year 2010 --end-year 2020
```

## 📈 **Performance Metrics**

### **Target Performance**
- **MAPE < 4%**: Excellent performance threshold
- **R² > 0.85**: Minimum acceptable accuracy
- **Cross-validation consistency**: Time series split validation
- **Uncertainty quantification**: 90% confidence intervals

### **Expected Results by Category**
| Steel Category | Target MAPE | Expected R² |
|----------------|-------------|-------------|
| Hot Rolled Structural Steel | ~3.5% | ~0.92 |
| Rail Products | ~4.5% | ~0.88 |
| Steel Billets | ~3.0% | ~0.90 |
| Steel Slabs | ~4.0% | ~0.87 |

## 🔧 **System Requirements**

### **Environment**
- Python 3.12+ with virtual environment
- 8GB+ RAM (16GB recommended for full ensemble)
- 2-4 CPU cores (configurable)
- ~1GB storage for data and outputs

### **Dependencies**
```bash
# Production (RECOMMENDED)
pip install -r requirements_fixed.txt

# Minimal development
pip install -r requirements_minimal.txt

# Development with extras
pip install -r requirements-dev.txt
```

### **Data Requirements**
- Complete WSA steel consumption data (2004-2023)
- WM macro-economic projections through 2050
- All 13 CSV configuration files
- Renewable energy capacity time series

## 🎯 **Key Features**

### **✅ Implemented Features**
- ✅ Dual-track forecasting architecture
- ✅ Renewable energy time series integration
- ✅ Timestamped output management
- ✅ ML ensemble with 5 algorithms
- ✅ CSV-driven configuration system
- ✅ Uncertainty quantification
- ✅ Cross-validation framework
- ✅ Hierarchical product taxonomy
- ✅ Performance benchmarking
- ✅ Comprehensive logging

### **🔧 Configurable Elements**
- Model hyperparameters and ensemble weights
- Renewable energy steel intensities
- Hierarchical product structure
- Performance thresholds and validation criteria
- Output directory and timestamping behavior
- Feature engineering parameters

### **📊 Output Deliverables**
- Unified forecasts combining both tracks
- Individual track results for comparison
- Uncertainty bounds and confidence intervals
- Feature importance analysis
- Performance metrics and validation reports
- Comprehensive execution logs

## 📚 **Documentation Structure**

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Main development guide and system overview |
| `README_HIERARCHICAL.md` | Hierarchical forecasting methodology |
| `SYSTEM_OVERVIEW.md` | This comprehensive system documentation |
| `TIMESTAMPED_OUTPUTS_IMPLEMENTATION.md` | Output management details |
| `config/*.csv` | Individual configuration parameter documentation |

## 🎉 **Summary**

The Australian Steel Demand Model provides a robust, flexible, and comprehensive platform for steel demand forecasting with:

1. **Dual methodological approaches** for robust predictions
2. **Renewable energy integration** for infrastructure demand modeling  
3. **Advanced ML ensemble** with uncertainty quantification
4. **Timestamped output management** for audit trails
5. **CSV-driven configuration** for maximum flexibility
6. **Comprehensive validation** framework for quality assurance

The system is production-ready with clear documentation, automated testing, and structured outputs suitable for both research and commercial applications.