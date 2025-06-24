# Requirements Management

## Current Setup

The project now uses a **single, consolidated requirements file** with tested and verified package versions.

### Files:

- **`requirements.txt`** - Main production requirements (use this)
- **`requirements-dev.txt`** - Optional development tools
- **`backup_requirements/`** - Archived old requirements files

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
pip install -r requirements-dev.txt
```

### Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Package Overview

### Core ML Stack
- **Data Science**: pandas, numpy, scipy, scikit-learn
- **ML Frameworks**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: TensorFlow, Keras
- **Time Series**: Prophet, statsmodels

### Features
- **Feature Engineering**: category_encoders, feature_engine
- **Model Interpretation**: SHAP, LIME, ELI5
- **Visualization**: matplotlib, seaborn, plotly
- **Hyperparameter Tuning**: Optuna

### Quality & Testing
- **Data Validation**: pandera, evidently
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, isort, flake8

### API & Services
- **Web Framework**: FastAPI, uvicorn
- **Data Models**: pydantic

## Version Notes

- All versions are **pinned** for reproducible builds
- Tested and verified to work together
- Based on working Python 3.12 environment
- Some packages excluded due to version conflicts:
  - `great-expectations` (pandas conflicts)
  - `sktime` (pandas conflicts)
  - `torch` (large size - available in dev requirements)

## Troubleshooting

If you encounter dependency conflicts:

1. Use a fresh virtual environment
2. Install exact versions from `requirements.txt`
3. Check `backup_requirements/` for alternative configurations

## Maintenance

- Package versions should be updated carefully and tested
- Use `pip freeze > requirements_new.txt` to capture current environment
- Test full training pipeline after any updates