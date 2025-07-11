# Australian Steel Demand Model (SDM)

A comprehensive forecasting platform for Australian steel consumption through 2050, featuring dual-track modeling, renewable energy integration, and advanced ML ensemble methods.

## 🚀 Quick Start

```bash
# 1. Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements_fixed.txt

# 3. Verify installation
python verify_data_setup.py

# 4A. Run Track A (Production Forecasting)
python track_a_production_forecasting.py

# 4B. Run Track B (Consumption/Demand Forecasting)
python track_b_consumption_forecasting.py
```

## 🎯 Key Features

- **Dual-Track Forecasting**: Traditional + hierarchical approaches
- **Renewable Energy Integration**: Wind and solar capacity time series
- **Optimized ML Ensemble**: Track A (XGBoost 60% + Random Forest 40%)
- **Comprehensive Visualizations**: Performance dashboards, feature importance, forecast charts
- **Timestamped Outputs**: Automatic audit trails and version control
- **CSV-Driven Configuration**: 13 configuration files, no hardcoded parameters

## 📊 System Overview

### Architecture
- **Track A**: Production forecasting using ML regression models (XGBoost + Random Forest)
- **Track B**: Consumption/demand forecasting with hierarchical product taxonomy
- **Dual Scope**: Production vs. consumption - complementary market intelligence

### Performance Targets
- **MAPE < 4%** for excellent performance
- **R² > 0.85** minimum acceptable accuracy
- **Uncertainty quantification** with 90% confidence intervals

## 📚 Documentation

| File | Purpose |
|------|---------|
| **[CLAUDE.md](CLAUDE.md)** | 🎯 **START HERE** - Main development guide |
| **[SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md)** | Comprehensive technical documentation |
| **[README_HIERARCHICAL.md](README_HIERARCHICAL.md)** | Hierarchical forecasting methodology |
| **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** | Project organization and file structure |
| **[TIMESTAMPED_OUTPUTS_IMPLEMENTATION.md](TIMESTAMPED_OUTPUTS_IMPLEMENTATION.md)** | Output management details |

## ⚡ Quick Commands

```bash
# Primary forecasting (RECOMMENDED)
python run_hierarchical_forecasting.py

# Traditional ML ensemble
python train_steel_demand_models.py

# Data validation
python verify_data_setup.py

# System test
python test_system.py

# Generate visualizations
python create_visualizations.py forecasts/latest_results/
```

## 🔧 Requirements

- **Python 3.12+** with virtual environment
- **8GB+ RAM** recommended for full ensemble
- **Complete data files**: WSA steel data + WM macro projections
- **All configuration files**: 13 CSV files in `config/` directory

## 📈 Output Structure

```
forecasts/
├── hierarchical_run_YYYYMMDD_HHMMSS/    # Dual-track results
├── training_YYYYMMDD_HHMMSS/             # ML ensemble training
└── [timestamped directories]             # Historical runs preserved
```

## 🎯 For New Users

1. **Read [CLAUDE.md](CLAUDE.md)** for comprehensive development guide
2. **Install dependencies** using `requirements_fixed.txt`
3. **Verify setup** with `python verify_data_setup.py`
4. **Run forecasting** with `python run_hierarchical_forecasting.py`

---

**Part of the larger industrial modeling workspace with cross-project integration capabilities for energy forecasting, cost modeling, and regional economic analysis.**