#!/usr/bin/env python3
"""
Standalone Visualization Generator for Steel Demand Model

Creates visualizations from existing training results.
Useful for regenerating charts or creating visualizations for older runs.

Usage:
    python create_visualizations.py <results_directory>
    python create_visualizations.py forecasts/ml_regression_features_20250710_185347/

Requirements:
    - Algorithm_Performance_Comparison.csv (required)
    - ML_Algorithm_Forecasts_2025-2050.csv or Combined_ML_Steel_Forecasts_2025-2050.csv (required)
    - Ensemble_Forecasts_2025-2050.csv (optional)
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import logging

def main():
    """Main function to generate visualizations from results directory."""
    parser = argparse.ArgumentParser(
        description='Generate visualizations from Steel Demand Model training results'
    )
    parser.add_argument(
        'results_dir',
        type=str,
        help='Directory containing training results (CSV files)'
    )
    parser.add_argument(
        '--performance-file',
        type=str,
        default='Algorithm_Performance_Comparison.csv',
        help='Name of performance comparison file (default: Algorithm_Performance_Comparison.csv)'
    )
    parser.add_argument(
        '--forecast-file',
        type=str,
        default=None,
        help='Name of forecast file (auto-detected if not specified)'
    )
    parser.add_argument(
        '--ensemble-file',
        type=str,
        default='Ensemble_Forecasts_2025-2050.csv',
        help='Name of ensemble file (default: Ensemble_Forecasts_2025-2050.csv)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing visualizations'
    )
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    
    # Validate results directory
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1
    
    if not results_dir.is_dir():
        logger.error(f"Path is not a directory: {results_dir}")
        return 1
    
    # Check for required files
    performance_file = results_dir / args.performance_file
    if not performance_file.exists():
        logger.error(f"Performance file not found: {performance_file}")
        logger.info("Available files in directory:")
        for file in results_dir.glob("*.csv"):
            logger.info(f"  {file.name}")
        return 1
    
    # Auto-detect forecast file if not specified
    forecast_file = None
    if args.forecast_file:
        forecast_file = results_dir / args.forecast_file
        if not forecast_file.exists():
            logger.error(f"Forecast file not found: {forecast_file}")
            return 1
    else:
        # Try common forecast file names
        for fname in ['ML_Algorithm_Forecasts_2025-2050.csv', 'Combined_ML_Steel_Forecasts_2025-2050.csv']:
            candidate = results_dir / fname
            if candidate.exists():
                forecast_file = candidate
                break
        
        if forecast_file is None:
            logger.error("No forecast file found. Available CSV files:")
            for file in results_dir.glob("*.csv"):
                logger.info(f"  {file.name}")
            return 1
    
    # Check for ensemble file (optional)
    ensemble_file = results_dir / args.ensemble_file
    if not ensemble_file.exists():
        logger.warning(f"Ensemble file not found: {args.ensemble_file}")
        ensemble_file = None
    
    # Check if visualizations already exist
    viz_dir = results_dir / "visualizations"
    if viz_dir.exists() and not args.force:
        existing_files = list(viz_dir.glob("*.png"))
        if existing_files:
            logger.warning(f"Visualizations already exist in {viz_dir}")
            logger.warning("Use --force to overwrite existing visualizations")
            logger.info(f"Existing files: {[f.name for f in existing_files]}")
            return 0
    
    # Import and generate visualizations
    try:
        from visualization.steel_demand_visualizer import create_visualizations_for_results
        
        logger.info(f"Generating visualizations for: {results_dir}")
        logger.info(f"Performance file: {performance_file.name}")
        logger.info(f"Forecast file: {forecast_file.name}")
        if ensemble_file:
            logger.info(f"Ensemble file: {ensemble_file.name}")
        
        # Generate visualizations
        viz_files = create_visualizations_for_results(
            str(results_dir),
            performance_file=performance_file.name,
            forecast_file=forecast_file.name,
            ensemble_file=ensemble_file.name if ensemble_file else None
        )
        
        print("\n" + "="*60)
        print("✅ VISUALIZATION GENERATION COMPLETED")
        print("="*60)
        print(f"📊 Results directory: {results_dir}")
        print(f"📁 Visualizations saved to: {viz_dir}")
        print("\n📋 Generated files:")
        
        for viz_type, filepath in viz_files.items():
            print(f"  • {viz_type}: {Path(filepath).name}")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Visualization module not available: {e}")
        logger.error("Install required packages: matplotlib, seaborn")
        return 1
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)