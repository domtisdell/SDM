# Australian Steel Demand Model - Project Structure

## Directory Overview

```
wm_wsa_model/                              # Main project directory
├── 📁 data/                               # Data files and sources
├── 📁 config/                             # CSV-driven configuration system
├── 📁 forecasting/                        # Dual-track forecasting implementations
├── 📁 training/                           # Model training and cross-validation
├── 📁 models/                             # ML model implementations
├── 📁 data/                               # Data loading and feature engineering
├── 📁 evaluation/                         # Uncertainty quantification and validation
├── 📁 forecasts/                          # Timestamped output directories
├── 📁 venv/                               # Virtual environment (created by user)
├── 📄 Main execution scripts              # Entry points for system operation
├── 📄 Requirements files                  # Dependency specifications
└── 📄 Documentation files                 # System documentation
```

## 📁 **Core Directories**

### **`data/` - Data Sources**
```
data/
├── WSA_Au_2004-2023.csv                   # 🔴 CRITICAL: Steel consumption data
├── WM_macros.csv                          # 🔴 CRITICAL: Economic + renewable time series
├── Steel_Consumption_Forecasts_2025-2050.csv # Reference forecasts for validation
└── [cleaned up - unused files removed]    # Previous cleanup removed 6 unused files
```

**Data File Descriptions:**
| File | Years | Description | Size | Status |
|------|-------|-------------|------|--------|
| `WSA_Au_2004-2023.csv` | 2004-2023 | Steel consumption by product category | ~20 rows | 🔴 Critical |
| `WM_macros.csv` | 2004-2050 | Economic drivers + renewable energy capacity | ~47 rows | 🔴 Critical |
| `Steel_Consumption_Forecasts_2025-2050.csv` | 2025-2050 | Reference benchmarks | ~26 rows | ✅ Active |

### **`config/` - Configuration System (13 Files)**
```
config/
├── 🔧 Core Configuration
│   ├── model_config.csv                   # ML hyperparameters and training settings
│   ├── data_sources.csv                   # Data file locations and descriptions
│   ├── steel_categories.csv               # Steel product definitions
│   ├── economic_indicators.csv            # Economic variable definitions
│   └── validation_benchmarks.csv          # Performance targets and thresholds
├── 🌱 Renewable Energy
│   ├── renewable_steel_intensity.csv      # Steel intensities by technology type
│   └── renewable_technology_mapping.csv   # Technology definitions and mapping
├── 🏗️ Hierarchical Structure
│   ├── hierarchical_steel_products_L1.csv # Level 1 category definitions
│   ├── hierarchical_steel_products_L2.csv # Level 2 product families
│   ├── hierarchical_steel_products_L3.csv # Level 3 specific products
│   └── sectoral_steel_mapping.csv         # Sector-to-product mapping matrices
└── 🌏 Regional & Validation
    ├── regional_adjustment_factors.csv    # State-level population adjustments
    └── end_use_sector_mapping.csv         # End-use sector definitions
```

**Configuration Status:**
- ✅ **12/13 files actively used** in current system
- ⚠️ **1 file potentially unused** (`end_use_sector_mapping.csv` - kept for future use)
- 🔧 **All parameters configurable** - no hardcoded values in system

### **`forecasting/` - Dual-Track Implementation**
```
forecasting/
├── dual_track_forecasting.py              # 🚀 Track A: Traditional + renewable integration
├── hierarchical_forecasting.py            # 🏗️ Track B: Hierarchical product taxonomy
└── __init__.py                            # Module initialization
```

**Implementation Details:**
| File | Purpose | Track | Features |
|------|---------|-------|----------|
| `dual_track_forecasting.py` | Traditional approach with renewables | Track A | Direct time series integration, no feature engineering |
| `hierarchical_forecasting.py` | Multi-level taxonomy forecasting | Track B | Consistency constraints, aggregation logic |

### **`training/` - Model Training Framework**
```
training/
├── model_trainer.py                       # 🧠 Core training and cross-validation framework
└── __init__.py                            # Module initialization
```

### **`models/` - ML Model Implementations**
```
models/
├── ensemble_models.py                     # 🤖 5-algorithm ensemble implementation
└── __init__.py                            # Module initialization
```

**Ensemble Composition:**
- **XGBoost** (45% weight): Gradient boosting for non-linear relationships
- **Random Forest** (40% weight): Ensemble decision trees for robustness
- **Linear Regression** (15% weight): Baseline linear relationships
- **LSTM** (available): Deep learning for time series patterns
- **Prophet** (available): Time series decomposition and trends

### **`data/` - Data Processing**
```
data/
├── data_loader.py                         # 📊 CSV data loading and validation
├── feature_engineering.py                 # ⚙️ Feature creation and preprocessing
└── __init__.py                            # Module initialization
```

### **`evaluation/` - Validation Framework**
```
evaluation/
├── [uncertainty quantification modules]   # 📊 Bootstrap sampling and confidence intervals
├── [validation frameworks]                # ✅ Performance assessment and benchmarking
└── __init__.py                            # Module initialization
```

### **`forecasts/` - Timestamped Outputs**
```
forecasts/                                 # 📈 Automatically managed output directory
├── hierarchical_run_20250710_181517/      # Latest dual-track forecasting run
├── training_20250710_120156/              # ML ensemble training results
├── regression_ml_20250710_143829/         # Simplified ML algorithm comparison
├── ml_regression_features_20250710_*/     # ML with regression features
└── [previous timestamped directories]     # Historical runs preserved
```

**Output Management:**
- ✅ **Automatic timestamping** prevents overwrites
- 📝 **Comprehensive logging** in each directory
- 🔍 **Audit trail** of all system runs
- ⚙️ **Configurable** via `--no-timestamp` flag

## 📄 **Main Execution Scripts**

### **Primary Entry Points**
```
wm_wsa_model/
├── run_hierarchical_forecasting.py        # 🚀 MAIN: Dual-track forecasting system
├── train_steel_demand_models.py           # 🧠 Traditional ML ensemble training
├── train_regression_ml_models.py          # 📊 Simplified ML algorithm comparison
├── train_ml_with_regression_features.py   # ⚙️ ML with regression features
├── verify_data_setup.py                   # ✅ Data and configuration validation
└── test_system.py                         # 🧪 Quick system functionality test
```

**Usage Priority:**
1. **`run_hierarchical_forecasting.py`** - 🥇 **RECOMMENDED** for production forecasting
2. **`train_steel_demand_models.py`** - 🥈 Traditional ML ensemble approach
3. **`train_regression_ml_models.py`** - 🥉 Simplified algorithm comparison
4. **`verify_data_setup.py`** - ✅ Always run first for new installations

### **Development and Testing Scripts**
```
wm_wsa_model/
├── verify_data_setup.py                   # 🔍 Comprehensive data validation
├── test_system.py                         # ⚡ Quick functionality test
└── [various utility scripts]              # 🛠️ Development helpers
```

## 📄 **Requirements and Dependencies**

### **Requirements Files**
```
wm_wsa_model/
├── requirements_fixed.txt                 # 🎯 RECOMMENDED: Complete ML stack (tested)
├── requirements_minimal.txt               # ⚡ Essential packages only
├── requirements_current.txt               # 📊 Current environment state
├── requirements.txt                       # 🔗 Standard requirements (full stack)
└── requirements-dev.txt                   # 🛠️ Development extras and optional packages
```

**Installation Recommendation:**
```bash
# Production use (RECOMMENDED)
pip install -r requirements_fixed.txt

# Lightweight development
pip install -r requirements_minimal.txt

# Development with extras
pip install -r requirements-dev.txt
```

## 📄 **Documentation Files**

### **Core Documentation**
```
wm_wsa_model/
├── CLAUDE.md                              # 🎯 MAIN: Development guide and system overview
├── README_HIERARCHICAL.md                 # 🏗️ Hierarchical forecasting methodology
├── SYSTEM_OVERVIEW.md                     # 📋 Comprehensive system documentation
├── PROJECT_STRUCTURE.md                   # 📁 This file - project organization
└── TIMESTAMPED_OUTPUTS_IMPLEMENTATION.md  # ⏰ Output management details
```

**Documentation Hierarchy:**
1. **`CLAUDE.md`** - 🥇 **START HERE** - Main development guide
2. **`SYSTEM_OVERVIEW.md`** - 🥈 Comprehensive technical overview
3. **`README_HIERARCHICAL.md`** - 🥉 Hierarchical methodology details
4. **`PROJECT_STRUCTURE.md`** - 📁 This file - organization guide

## 🔧 **Virtual Environment**

### **Environment Structure**
```
wm_wsa_model/
└── venv/                                  # 🐍 Python virtual environment (user-created)
    ├── bin/                               # Executables and activation scripts
    ├── lib/                               # Installed packages
    ├── include/                           # Header files
    └── pyvenv.cfg                         # Environment configuration
```

**Setup Commands:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements_fixed.txt
```

## 📊 **File Status Summary**

### **Critical Files (🔴 Required)**
- ✅ `data/WSA_Au_2004-2023.csv` - Steel consumption data
- ✅ `data/WM_macros.csv` - Economic + renewable time series
- ✅ All 13 configuration files in `config/`
- ✅ Main execution scripts

### **Active Development Files (✅ Used)**
- ✅ All forecasting, training, and model implementation files
- ✅ Data processing and evaluation frameworks
- ✅ Documentation and requirements files

### **Cleaned Up (🗑️ Removed)**
- 🗑️ 6 unused files from `data/` directory (historical cleanup)
- 🗑️ Redundant configuration files
- 🗑️ Legacy code and deprecated scripts

### **Generated Files (📈 Automatic)**
- 📈 Timestamped output directories in `forecasts/`
- 📝 Log files with execution details
- 📊 Performance metrics and validation reports
- 🤖 Trained model artifacts

## 🎯 **Quick Navigation**

### **New Users Start Here:**
1. 📖 Read `CLAUDE.md` for development overview
2. ✅ Run `python verify_data_setup.py` to check installation
3. 🧪 Run `python test_system.py` for quick validation
4. 🚀 Execute `python run_hierarchical_forecasting.py` for forecasting

### **Configuration Changes:**
1. 🔧 Edit relevant CSV files in `config/` directory
2. ✅ Validate with `python verify_data_setup.py`
3. 🚀 Re-run forecasting with updated configuration

### **Troubleshooting:**
1. 📝 Check latest log files in timestamped output directories
2. ✅ Verify data integrity with `python verify_data_setup.py`
3. 📖 Refer to `CLAUDE.md` for common issues and solutions

## 🎉 **Summary**

The project structure is designed for:
- **🔧 Maintainability**: Clear separation of concerns and modular design
- **📊 Transparency**: CSV-driven configuration with no hardcoded values
- **🔍 Auditability**: Timestamped outputs and comprehensive logging
- **⚡ Usability**: Simple entry points and clear documentation
- **🛡️ Robustness**: Comprehensive validation and error handling

The system supports both research exploration and production deployment with minimal configuration requirements and maximum flexibility.