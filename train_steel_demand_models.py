#!/usr/bin/env python3
"""
Australian Steel Demand Model - ML Training Script
Main script to train ensemble models for steel demand forecasting.
All parameters and data loaded from CSV configuration files.

Usage:
    python train_steel_demand_models.py [--config-path CONFIG_PATH] [--output-dir OUTPUT_DIR]
    
Example:
    python train_steel_demand_models.py --config-path config/ --output-dir forecasts/
"""

import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import logging
import time
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import our modules
from training.model_trainer import SteelDemandModelTrainer

def setup_logging(log_file: str = "training.log") -> logging.Logger:
    """Setup comprehensive logging with guaranteed file creation."""
    # Ensure log file directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup logging with both file and console output
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),  # Force create new log file
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing configuration
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - log file: {log_file}")
    
    return logger

def validate_config_files(config_path: Path) -> bool:
    """Validate that all required configuration files exist."""
    required_files = [
        'model_config.csv',
        'data_sources.csv',
        'steel_categories.csv',
        'regional_adjustment_factors.csv',
        'economic_indicators.csv',
        'validation_benchmarks.csv'
    ]
    
    missing_files = []
    for file_name in required_files:
        if not (config_path / file_name).exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"ERROR: Missing configuration files: {missing_files}")
        return False
    
    return True

def check_data_files(config_path: Path) -> bool:
    """Check that all required data files exist."""
    try:
        # Load data sources configuration
        data_sources_file = config_path / 'data_sources.csv'
        data_sources = pd.read_csv(data_sources_file)
        
        missing_data_files = []
        for _, source in data_sources.iterrows():
            file_path = source['file_path']
            
            # Handle relative paths
            if not file_path.startswith('/'):
                full_path = config_path.parent / file_path
            else:
                full_path = Path(file_path)
            
            if not full_path.exists():
                missing_data_files.append(str(full_path))
        
        if missing_data_files:
            print(f"ERROR: Missing data files: {missing_data_files}")
            return False
        
        return True
        
    except Exception as e:
        print(f"ERROR: Could not validate data files: {str(e)}")
        return False

def generate_forecasts(trainer: SteelDemandModelTrainer, 
                      output_dir: Path) -> None:
    """Generate forecasts using trained models."""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Generating forecasts using trained models")
        
        # Load projection data
        projection_data = trainer.data_loader.get_projection_data(2025, 2050)
        steel_categories = trainer.data_loader.get_steel_categories()
        
        forecast_results = {}
        
        for _, category_info in steel_categories.iterrows():
            category_name = category_info['category']
            target_column = category_info['target_column']
            
            if category_name in trainer.trained_models:
                try:
                    logger.info(f"Generating forecasts for {category_name}")
                    
                    # Get trained model
                    model = trainer.trained_models[category_name]
                    
                    # Prepare features for projection data
                    # Note: This is simplified - in practice, you'd need to engineer
                    # features for the projection data similar to training
                    projection_features = trainer.feature_engineer.create_features(
                        projection_data, target_column
                    )
                    
                    # Get feature columns used in training
                    feature_info = trainer.feature_sets.get(target_column, {})
                    feature_columns = feature_info.get('feature_columns', [])
                    
                    # Select available features
                    available_features = [col for col in feature_columns 
                                        if col in projection_features.columns]
                    
                    if available_features:
                        X_forecast = projection_features[available_features]
                        
                        # Generate predictions
                        predictions = model.predict(X_forecast)
                        
                        # Create forecast DataFrame
                        forecast_df = pd.DataFrame({
                            'Year': projection_data['Year'],
                            f'{category_name}_Forecast_tonnes': predictions,
                            f'{category_name}_Model_Type': 'ML_Ensemble'
                        })
                        
                        forecast_results[category_name] = forecast_df
                        
                        # Save individual forecast
                        forecast_file = output_dir / f'{category_name}_ML_Forecasts_2025-2050.csv'
                        forecast_df.to_csv(forecast_file, index=False)
                        
                        logger.info(f"Saved forecasts for {category_name} to {forecast_file}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate forecasts for {category_name}: {str(e)}")
        
        # Combine all forecasts into single file
        if forecast_results:
            combined_forecast = projection_data[['Year']].copy()
            
            for category_name, forecast_df in forecast_results.items():
                forecast_col = f'{category_name}_Forecast_tonnes'
                combined_forecast[forecast_col] = forecast_df[forecast_col]
            
            # Calculate total forecast
            forecast_columns = [col for col in combined_forecast.columns 
                              if col.endswith('_Forecast_tonnes')]
            combined_forecast['Total_Steel_Forecast_tonnes'] = combined_forecast[forecast_columns].sum(axis=1)
            
            # Save combined forecasts
            combined_file = output_dir / 'Combined_ML_Steel_Forecasts_2025-2050.csv'
            combined_forecast.to_csv(combined_file, index=False)
            
            logger.info(f"Saved combined forecasts to {combined_file}")
        
    except Exception as e:
        logger.error(f"Error generating forecasts: {str(e)}")
        raise

def main():
    """Main training function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train ML models for Australian Steel Demand forecasting'
    )
    parser.add_argument(
        '--config-path', 
        type=str, 
        default='config/',
        help='Path to configuration directory (default: config/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='forecasts/',
        help='Output directory for results (default: forecasts/)'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Skip cross-validation (faster training)'
    )
    parser.add_argument(
        '--start-year',
        type=int,
        default=None,
        help='Start year for training data (default: all available)'
    )
    parser.add_argument(
        '--end-year',
        type=int,
        default=None,
        help='End year for training data (default: all available)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    config_path = Path(args.config_path)
    output_dir = Path(args.output_dir)
    
    # Create output directory early
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return 1
    
    # Setup logging immediately after output directory creation
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    try:
        logger = setup_logging(str(log_file))
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to setup file logging, using console only: {e}")
    
    # Ensure we always have a log entry at the start
    logger.info("="*60)
    logger.info("Australian Steel Demand Model - ML Training")
    logger.info("="*60)
    logger.info(f"Configuration path: {config_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Cross-validation: {'No' if args.no_cv else 'Yes'}")
    
    try:
        # Validate configuration files
        logger.info("Validating configuration files...")
        if not validate_config_files(config_path):
            return 1
        
        # Check data files
        logger.info("Checking data files...")
        if not check_data_files(config_path):
            return 1
        
        logger.info("All files validated successfully")
        
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = SteelDemandModelTrainer(str(config_path))
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        data = trainer.load_and_prepare_data(args.start_year, args.end_year)
        
        # Train models
        logger.info("Training models for all steel categories...")
        start_time = time.time()
        
        training_results = trainer.train_all_categories(
            data, 
            perform_cv=not args.no_cv
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f} seconds")
        
        # Generate summary report
        logger.info("Generating summary report...")
        summary_report = trainer.generate_summary_report()
        
        # Save summary report
        summary_file = output_dir / 'model_training_summary.csv'
        summary_report.to_csv(summary_file, index=False)
        logger.info(f"Summary report saved to {summary_file}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(summary_report.to_string(index=False))
        
        # Save models and detailed results
        logger.info("Saving models and results...")
        saved_files = trainer.save_models_and_results(str(output_dir))
        
        logger.info("Saved files:")
        for file_type, file_path in saved_files.items():
            logger.info(f"  {file_type}: {file_path}")
        
        # Generate forecasts
        logger.info("Generating forecasts...")
        generate_forecasts(trainer, output_dir)
        
        # Final summary
        logger.info("="*60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        successful_models = summary_report[summary_report['Status'] == 'SUCCESS']
        failed_models = summary_report[summary_report['Status'] == 'FAILED']
        
        logger.info(f"Successful models: {len(successful_models)}")
        logger.info(f"Failed models: {len(failed_models)}")
        
        if len(successful_models) > 0:
            avg_mape = successful_models['MAPE'].mean()
            avg_r2 = successful_models['R2'].mean()
            logger.info(f"Average MAPE: {avg_mape:.2f}%")
            logger.info(f"Average R²: {avg_r2:.3f}")
        
        logger.info(f"All outputs saved to: {output_dir}")
        logger.info(f"Log file saved to: {log_file}")
        
        # Final cleanup
        logging.shutdown()
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Check the log file for detailed error information")
        logger.error(f"Log file location: {log_file}")
        
        # Ensure logging is flushed even on error
        logging.shutdown()
        return 1
    
    finally:
        # Always ensure logging is properly closed
        try:
            logging.shutdown()
        except:
            pass

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)