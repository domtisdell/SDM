# Australian Steel Demand Hierarchical Forecasting System

## Overview

The Australian Steel Demand Model (SDM) now implements a **dual-track forecasting approach** that combines traditional econometric modeling with hierarchical product taxonomy forecasting. This enhanced system provides comprehensive steel demand forecasting through 2050 with renewable energy integration and automatic timestamped outputs.

## Quick Start Guide

### **Running the Dual-Track System**

```bash
# Dual-track forecasting with timestamped output (RECOMMENDED)
python run_hierarchical_forecasting.py

# Custom output directory (automatically timestamped)
python run_hierarchical_forecasting.py --output-dir production_run

# Disable timestamping for rapid development
python run_hierarchical_forecasting.py --output-dir test_run --no-timestamp

# Alternative: Traditional ML ensemble only
python train_steel_demand_models.py
python train_regression_ml_models.py
```

### **Data Verification**

```bash
# Verify all data files and configuration
python verify_data_setup.py

# Quick system functionality test
python test_system.py
```

### **Key Features**

#### **Dual-Track Forecasting Architecture**
- **Track A (Traditional)**: Enhanced current methodology with renewable energy time series
- **Track B (Hierarchical)**: Multi-level product taxonomy with consistency constraints
- **Unified Output**: Combined forecasts with cross-validation and performance comparison

#### **Renewable Energy Integration**
- Direct time series integration: Wind_Onshore, Wind_Offshore, Solar_Grid, Solar_Distributed
- No complex feature engineering - raw capacity data used as model inputs
- Steel intensity mapping for infrastructure demand calculation
- Renewable capacity targets through 2050 alignment

#### **Advanced ML Ensemble**
- **Track A Ensemble**: XGBoost (60%), Random Forest (40%) - Optimized 2-model system
- Cross-validation with time series splits
- Uncertainty quantification with bootstrap sampling
- Simplified system focusing on best-performing algorithms only

#### **Hierarchical Product Structure**
- **Level 0**: Total Steel Consumption
- **Level 1**: Major Categories (Flat Products, Long Products, etc.)
- **Level 2**: Product Families (Hot Rolled, Cold Rolled, etc.)
- **Level 3**: Specific Products (Hot Rolled Structural Steel, Rail Products, etc.)

#### **Timestamped Output Management**
- Automatic timestamped directories prevent overwrites
- Clear audit trail of all forecasting runs
- Option to disable timestamping for rapid development
- Structured output with comprehensive logging

### **Output Files Structure**

#### **Timestamped Directory Structure**
```
forecasts/
├── hierarchical_run_20250710_181517/    # Latest dual-track run
├── training_20250710_120156/             # ML ensemble training
├── regression_ml_20250710_143829/        # Simplified ML algorithms
└── [previous timestamped directories]    # Historical runs preserved
```

#### **Dual-Track Forecasting Outputs**
**Primary Forecasts**:
- `unified_forecasts_2025-2050.csv` - Combined Track A + Track B results
- `track_a_forecasts_2025-2050.csv` - Traditional approach with renewables
- `track_b_forecasts_2025-2050.csv` - Hierarchical approach results
- `performance_comparison.csv` - Track performance metrics

**Traditional ML Ensemble Outputs**:
- `Combined_ML_Steel_Forecasts_2025-2050.csv` - All categories combined
- `{Category}_ML_Forecasts_2025-2050.csv` - Individual category forecasts
- `model_training_summary.csv` - Performance metrics by category
- `{Category}_uncertainty_report.csv` - Confidence intervals

**Analysis and Diagnostics**:
- `feature_importance_analysis.csv` - Economic driver importance
- `renewable_impact_assessment.csv` - Renewable energy contribution analysis
- `cross_validation_summary.csv` - Model validation results
- `training_log_YYYYMMDD_HHMMSS.log` - Detailed execution log

**Visualizations**:
- `visualizations/algorithm_accuracy_dashboard.png` - Performance dashboard with MAPE, R² metrics
- `visualizations/algorithm_performance_comparison.png` - Heatmaps showing algorithm performance by category
- `visualizations/feature_importance_heatmap.png` - Feature importance visualization for tree-based models
- `visualizations/historical_ensemble_vs_algorithms.png` - Historical data vs Track A ensemble forecasts
- `visualizations/model_summary_report.png` - Comprehensive training summary report

### **Validation Framework**

#### **Performance Targets**
- **MAPE < 4%** for excellent performance
- **R² > 0.85** minimum acceptable accuracy
- **Cross-validation** with time series splits
- **Track comparison** to ensure consistency

#### **Multi-Level Validation**
- **Hierarchical Consistency**: Mathematical consistency across aggregation levels
- **Economic Relationships**: GDP, population, and industrial production correlations
- **Renewable Energy**: Steel intensity validation against technology specifications
- **Historical Validation**: Back-testing against known consumption patterns

#### **Track Comparison Metrics**
- **Forecast Divergence**: Track A vs Track B difference analysis
- **Uncertainty Overlap**: Confidence interval comparison
- **Feature Importance**: Driver significance across approaches
- **Performance Benchmarking**: MAPE and R² comparison

### **Configuration System**

#### **CSV-Driven Configuration (13 Files)**

**Core Configuration**:
- `model_config.csv` - ML hyperparameters and training settings
- `data_sources.csv` - Data file locations and descriptions
- `steel_categories.csv` - Product definitions and elasticity factors
- `economic_indicators.csv` - Economic driver definitions
- `validation_benchmarks.csv` - Performance targets

**Renewable Energy**:
- `renewable_steel_intensity.csv` - Steel intensities by technology type
- `renewable_technology_mapping.csv` - Technology definitions and mapping

**Hierarchical Structure**:
- `hierarchical_steel_products_L1.csv` - Level 1 category definitions
- `hierarchical_steel_products_L2.csv` - Level 2 product families
- `hierarchical_steel_products_L3.csv` - Level 3 specific products
- `sectoral_steel_mapping.csv` - Sector-to-product mapping matrices

**Regional and Validation**:
- `regional_adjustment_factors.csv` - State-level adjustments
- `end_use_sector_mapping.csv` - End-use sector definitions

### **System Requirements**

#### **Python Environment**
- **Python 3.12+** with virtual environment
- **Dependencies**: Install via `pip install -r requirements_fixed.txt`
- **Alternative**: Minimal install via `pip install -r requirements_minimal.txt`

#### **System Resources**
- **RAM**: 8GB+ recommended for full ensemble training
- **CPU**: 2-4 cores recommended (configurable in model_config.csv)
- **Storage**: ~1GB for data, models, and outputs

#### **Data Requirements**
- **WSA Steel Data**: `data/WSA_Au_2004-2023.csv` (steel consumption 2004-2023)
- **WM Macro Data**: `data/WM_macros.csv` (economic + renewable 2004-2050)
- **Reference Data**: `data/Steel_Consumption_Forecasts_2025-2050.csv`
- **Complete Configuration**: All 13 CSV files in `config/` directory

#### **Installation Steps**
```bash
# 1. Clone/navigate to project directory
cd wm_wsa_model/

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements_fixed.txt

# 4. Verify setup
python verify_data_setup.py

# 5. Run system test
python test_system.py

# 6. Execute dual-track forecasting
python run_hierarchical_forecasting.py

# 7. Generate visualizations (if not auto-generated)
python create_visualizations.py forecasts/latest_results/
```

### **Documentation References**

- **Main Documentation**: `CLAUDE.md` - Complete system overview
- **Implementation Details**: `TIMESTAMPED_OUTPUTS_IMPLEMENTATION.md`
- **Configuration Guide**: Individual CSV files in `config/` directory
- **Development Guide**: See CLAUDE.md for common development tasks

