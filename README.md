# Australian Steel Demand Model - ML Forecasting System

This directory contains a **fully self-contained** machine learning system for forecasting Australian steel demand using ensemble models. All required data files have been copied into the `data/` directory, making the system completely independent and portable. The system operates entirely from CSV configuration files with no hardcoded parameters.

## System Overview

The ML system provides:
- **Ensemble forecasting** using XGBoost, Random Forest, LSTM, and Prophet models
- **150+ engineered features** from real economic and steel consumption data
- **Uncertainty quantification** with confidence intervals and SHAP interpretability
- **Comprehensive validation** against historical data and regression benchmarks
- **CSV-driven configuration** with no synthetic data or hardcoded values

## Quick Start

1. **Verify Setup** (optional but recommended)
```bash
python verify_data_setup.py
```

2. **Install Dependencies**
```bash
# Install main requirements
pip install -r requirements.txt

# Or with virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Verify installation
python verify_requirements.py
```

3. **Train Models**
```bash
python train_steel_demand_models.py
```

4. **View Results**
Check the `forecasts/` directory for:
- Model training summary
- Individual steel category forecasts
- Feature importance analysis
- Uncertainty analysis reports

## Self-Contained Data Structure

All required data is now included in the `data/` directory:

```
ML_model/
├── data/
│   ├── historical_consumption/      # Steel consumption data 2007-2024
│   │   ├── Hot_Rolled_Structural_Steel_Consumption_2007-2024.csv
│   │   ├── Rail_Products_Consumption_2007-2024.csv
│   │   ├── Steel_Billets_Consumption_2007-2024.csv
│   │   └── Steel_Slabs_Consumption_2007-2024.csv
│   ├── macro_drivers/              # Economic drivers and projections
│   │   ├── Australian_Macro_Drivers_2007-2024.csv
│   │   └── Macro_Driver_Projections_2025-2050.csv
│   └── reference_forecasts/        # Reference forecasts for validation
│       └── Steel_Consumption_Forecasts_2025-2050.csv
├── config/                         # All configuration files
├── models/                         # ML model implementations
├── training/                       # Training framework
└── evaluation/                     # Validation and uncertainty analysis
```

## System Architecture

### Core Components

**Data Management**
- `data/data_loader.py` - Loads and validates all CSV data sources
- `data/feature_engineering.py` - Creates 150+ features from real economic data
- `config/` - All parameters and assumptions in CSV files

**ML Models**
- `models/ensemble_models.py` - XGBoost, Random Forest, LSTM, Prophet ensemble
- `training/model_trainer.py` - Training framework with cross-validation
- `evaluation/` - Uncertainty quantification and validation

**Main Scripts**
- `train_steel_demand_models.py` - Main training script
- Configuration files define all behavior

### Configuration Files

All system behavior is controlled by CSV files in `config/`:

**`model_config.csv`** - Model hyperparameters and training settings
```csv
parameter,value,description,data_type
train_test_split,0.8,Training data percentage,float
xgb_n_estimators,1000,Number of boosting rounds,int
lstm_units,64,LSTM hidden units,int
target_mape,4.0,Target MAPE percentage,float
```

**`data_sources.csv`** - Data file locations and descriptions
```csv
source_name,file_path,description,data_type,quality_score
hrs_consumption,data/historical_consumption/Hot_Rolled_Structural_Steel_Consumption_2007-2024.csv,Historical HRS consumption data,time_series,0.95
macro_drivers_historical,data/macro_drivers/Australian_Macro_Drivers_2007-2024.csv,Historical macro economic drivers,time_series,0.90
```

**`steel_categories.csv`** - Steel product definitions and properties
```csv
category,target_column,elasticity_gdp,elasticity_population,regional_sensitivity
Hot Rolled Structural Steel,Hot_Rolled_Structural_Steel_tonnes,1.8,2.2,high
Rail Products,Rail_Products_tonnes,1.2,1.8,medium
```

**`economic_indicators.csv`** - Economic variable definitions and weights
```csv
indicator,column_name,weight,correlation_threshold,transformation
GDP,GDP_Real_AUD_Billion,0.30,0.85,log
Population,Total_Population_Millions,0.25,0.80,linear
```

**`regional_adjustment_factors.csv`** - State-level factors
```csv
state,population_share_2025,steel_share_2025,adjustment_factor,growth_multiplier
New South Wales,33.1,36.5,1.05,0.95
Victoria,26.4,29.2,1.00,1.08
```

**`validation_benchmarks.csv`** - Performance targets
```csv
category,metric,target_value,acceptable_value,critical_value
Hot Rolled Structural Steel,MAPE,3.5,6.0,10.0
Hot Rolled Structural Steel,R2,0.92,0.85,0.75
```

## Usage Examples

### Basic Training
```bash
# Train all models with default settings
python train_steel_demand_models.py

# Train with custom configuration
python train_steel_demand_models.py --config-path custom_config/ --output-dir results/

# Skip cross-validation for faster training
python train_steel_demand_models.py --no-cv

# Train on specific time period
python train_steel_demand_models.py --start-year 2010 --end-year 2020
```

### Updating Data

To update forecasts with new data:

1. **Update data files** in `historic_consumption/` directory
2. **Modify configuration** if needed in `config/` files
3. **Retrain models**:
```bash
python train_steel_demand_models.py --output-dir updated_forecasts/
```

The system automatically:
- Validates all data sources
- Retrains models with new data
- Updates forecasts through 2050
- Exports updated results to CSV

### Output Files

The system generates:

**Forecasts**
- `Combined_ML_Steel_Forecasts_2025-2050.csv` - All categories combined
- `{Category}_ML_Forecasts_2025-2050.csv` - Individual category forecasts

**Model Performance**
- `model_training_summary.csv` - Performance metrics summary
- `training_results.json` - Detailed training results
- `validation_summary.csv` - Validation against historical data

**Feature Analysis**
- `{Category}_{Model}_feature_importance.csv` - Feature importance by model
- `{Category}_{Model}_shap_importance.csv` - SHAP-based feature analysis

**Uncertainty Analysis**
- `{Category}_uncertainty_report.csv` - Confidence intervals and uncertainty metrics

## Model Performance

The system targets:
- **MAPE < 4%** for excellent performance
- **R² > 0.85** minimum acceptable accuracy
- **25-35% improvement** over regression baselines

Expected performance by category:
- **Hot Rolled Structural Steel**: MAPE ~3.5%, R² ~0.92
- **Rail Products**: MAPE ~4.5%, R² ~0.88
- **Steel Billets**: MAPE ~3.0%, R² ~0.90
- **Steel Slabs**: MAPE ~4.0%, R² ~0.87

## Advanced Features

### Ensemble Weighting
Models are automatically weighted based on validation performance:
- XGBoost: 30% (adjustable)
- Random Forest: 25%
- LSTM: 25%
- Prophet: 20%

### Feature Engineering
Automatically creates 150+ features:
- **Lag features**: 1-5 year historical values
- **Rolling statistics**: Moving averages, volatility measures
- **Growth rates**: Year-over-year changes and acceleration
- **Ratios**: Steel intensity, efficiency metrics
- **Regional factors**: State-level adjustments
- **Economic cycles**: Trend and cycle decomposition

### Uncertainty Quantification
- **Bootstrap sampling**: 1000 iterations for confidence intervals
- **Monte Carlo**: 10000 samples for prediction uncertainty
- **SHAP values**: Model interpretability and feature attribution

### Validation Framework
- **Historical validation**: Against 2007-2024 consumption data
- **Baseline comparison**: vs. regression models from Historical_Regression_Analysis_2007-2024.md
- **Temporal backtesting**: Rolling window validation
- **Cross-validation**: 5-fold time series validation

## Technical Requirements

**Python 3.8+** with packages:
- pandas>=2.0.0, numpy>=1.24.0
- scikit-learn>=1.3.0, xgboost>=2.0.0
- tensorflow>=2.13.0, prophet>=1.1.4
- shap>=0.42.0 (interpretability)

**Data Requirements**:
- Historical consumption data (2007-2024)
- Macro economic drivers (2007-2050)
- All configuration CSV files

**System Resources**:
- 8GB+ RAM recommended
- 2-4 CPU cores
- ~2GB disk space for models and outputs

## Troubleshooting

**Common Issues**:

1. **Missing data files**
```
ERROR: Missing data files: [.../historic_consumption/...]
```
Solution: Ensure all data files exist as specified in `data_sources.csv`

2. **Configuration errors**
```
ERROR: Missing configuration files: [model_config.csv]
```
Solution: Verify all CSV files exist in `config/` directory

3. **Memory issues during training**
```
WARNING: Training time exceeded limit
```
Solution: Reduce `bootstrap_samples` or `monte_carlo_samples` in `model_config.csv`

4. **Poor model performance**
```
Performance status: POOR
```
Solution: Check data quality, adjust hyperparameters, or increase training data

**Logs**: Check `forecasts/training_YYYYMMDD_HHMMSS.log` for detailed error information

## System Design Principles

1. **No Hardcoded Values**: All parameters in CSV configuration files
2. **Real Data Only**: No synthetic or mock data - only verified Australian market data
3. **Annual Frequency**: No seasonal factors - purely annual increments
4. **CSV-Driven Updates**: System updates when input CSV files are modified
5. **Comprehensive Validation**: Multiple validation approaches against real historical data
6. **Uncertainty Awareness**: Full uncertainty quantification and confidence intervals

This design ensures the system can be updated by modifying CSV files without code changes, maintaining full transparency and auditability of all assumptions and parameters.