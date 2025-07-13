#!/usr/bin/env python3
"""
Australian Steel Demand Forecasting System - Main Execution Script
Runs the complete hierarchical forecasting system with dual-track methodology.
Generates both current model and hierarchical forecasts with comprehensive validation.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse

# Add current directory to path for imports
sys.path.append('.')

from forecasting.dual_track_forecasting import DualTrackSteelForecastingSystem
from validation.comprehensive_validation import ComprehensiveValidationFramework

def _generate_pdf_reports(output_dir: str) -> None:
    """Generate PDF reports from markdown files."""
    import subprocess
    from pathlib import Path
    
    # Create pdf_reports directory
    pdf_dir = Path(output_dir) / "pdf_reports"
    pdf_dir.mkdir(exist_ok=True)
    
    # Find all markdown files in the output directory
    md_files = list(Path(output_dir).rglob("*.md"))
    
    if md_files:
        # Call the PDF converter script
        cmd = [
            sys.executable,
            "convert_md_to_pdf_final.py",
            str(output_dir),
            str(pdf_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"PDF reports generated successfully in {pdf_dir}")
            else:
                print(f"Error generating PDFs: {result.stderr}")
        except Exception as e:
            print(f"Failed to generate PDFs: {e}")
    else:
        print("No markdown files found to convert to PDF")

def setup_logging(log_level="INFO"):
    """Setup comprehensive logging for the forecasting system."""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Setup logging configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/hierarchical_forecasting_{timestamp}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filename}")
    
    return logger

def main():
    """Main execution function for hierarchical steel demand forecasting."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Australian Steel Demand Hierarchical Forecasting System')
    parser.add_argument('--start-year', type=int, default=2025, help='Start year for forecasting (default: 2025)')
    parser.add_argument('--end-year', type=int, default=2050, help='End year for forecasting (default: 2050)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for results (if not specified, uses timestamped folder)')
    parser.add_argument('--config-path', type=str, default='config/', help='Path to configuration files')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only (no forecasting)')
    parser.add_argument('--export-detailed', action='store_true', help='Export detailed results and intermediate data')
    parser.add_argument('--no-timestamp', action='store_true', help='Disable timestamped folders (use fixed output directory)')
    
    args = parser.parse_args()
    
    # Create timestamped output directory if not specified
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f'forecasts/track_b_{timestamp}'
    elif not args.no_timestamp and not args.output_dir.endswith('_' + datetime.now().strftime("%Y%m%d")):
        # Add timestamp to specified directory unless --no-timestamp is used
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f'{args.output_dir}_{timestamp}'
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("="*80)
    logger.info("AUSTRALIAN STEEL DEMAND HIERARCHICAL FORECASTING SYSTEM")
    logger.info("="*80)
    logger.info(f"Forecast Period: {args.start_year}-{args.end_year}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Configuration Path: {args.config_path}")
    logger.info("="*80)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Initialize the dual-track forecasting system
        logger.info("Initializing Dual-Track Steel Forecasting System...")
        forecasting_system = DualTrackSteelForecastingSystem(config_path=args.config_path)
        
        # Initialize validation framework
        logger.info("Initializing Comprehensive Validation Framework...")
        validation_framework = ComprehensiveValidationFramework(config_path=args.config_path)
        
        if not args.validate_only:
            # Generate unified forecasts
            logger.info(f"Generating unified forecasts for {args.start_year}-{args.end_year}...")
            unified_results = forecasting_system.generate_unified_forecasts(
                start_year=args.start_year,
                end_year=args.end_year
            )
            
            # Export unified results
            logger.info(f"Exporting unified results to {args.output_dir}...")
            forecasting_system.export_unified_results(unified_results, args.output_dir)
            
            # Generate client insights
            logger.info("Generating client-specific insights...")
            client_insights = forecasting_system.generate_client_insights(unified_results)
            
            # Export client insights
            client_insights_df = pd.DataFrame.from_dict(client_insights, orient='index')
            client_insights_df.to_csv(f"{args.output_dir}/client_insights_analysis.csv")
            
            # Generate comprehensive validation report
            logger.info("Generating comprehensive validation report...")
            
            # Load macro data for validation
            macro_data = pd.read_csv("data/macro_drivers/WM_macros.csv")
            
            validation_report = validation_framework.generate_comprehensive_validation_report(
                forecasts=unified_results['hierarchical_forecasts'],
                current_forecasts=unified_results['current_model_forecasts'],
                macro_data=macro_data,
                renewable_data=macro_data  # WM_macros contains renewable data
            )
            
            # Export validation report
            validation_summary = pd.DataFrame.from_dict(validation_report, orient='index')
            validation_summary.to_csv(f"{args.output_dir}/comprehensive_validation_report.csv")
            
            # Generate Track B visualizations
            logger.info("Generating Track B (hierarchical) visualizations...")
            try:
                from visualization.hierarchical_visualizer import create_track_b_visualizations
                viz_files = create_track_b_visualizations(args.output_dir)
                
                logger.info("Track B visualizations created successfully:")
                for viz_type, filepath in viz_files.items():
                    if filepath:
                        from pathlib import Path
                        logger.info(f"  • {viz_type}: {Path(filepath).name}")
                        
            except ImportError:
                logger.warning("Track B visualization module not available - skipping visualization generation")
            except Exception as e:
                logger.error(f"Error creating Track B visualizations: {str(e)}")
            
            # Generate Track B hierarchy diagrams for multiple years
            logger.info("Generating Track B hierarchy diagrams for years 2015, 2020, 2025, 2035, 2050...")
            try:
                from visualization.hierarchy_diagram_generator import HierarchyDiagramGenerator
                
                # Create hierarchy diagram generator
                diagram_generator = HierarchyDiagramGenerator(
                    forecast_results_path=args.output_dir,
                    output_dir=f"{args.output_dir}/hierarchy_diagrams"
                )
                
                # Generate diagrams for multiple years (including historical snapshots)
                generated_files = diagram_generator.generate_all_years([2015, 2020, 2025, 2035, 2050])
                
                # Create index file
                index_file = diagram_generator.create_index_file(generated_files)
                
                logger.info("Track B hierarchy diagrams created successfully:")
                for year, filepath in generated_files.items():
                    from pathlib import Path
                    logger.info(f"  • {year}: {Path(filepath).name}")
                logger.info(f"  • Index: {Path(index_file).name}")
                        
            except ImportError:
                logger.warning("Hierarchy diagram generator module not available - skipping diagram generation")
            except Exception as e:
                logger.error(f"Error creating Track B hierarchy diagrams: {str(e)}")
            
            # Generate Track B Taxonomy Analysis (similar to WSA analysis for Track A)
            logger.info("Generating Track B Steel Taxonomy Analysis...")
            try:
                from analysis.track_b_steel_taxonomy import TrackBSteelTaxonomyAnalyzer
                
                # Create Track B taxonomy analyzer
                taxonomy_analyzer = TrackBSteelTaxonomyAnalyzer()
                
                # Run comprehensive Track B taxonomy analysis
                taxonomy_results = taxonomy_analyzer.analyze_track_b_forecasts(
                    forecast_data_dir=args.output_dir,
                    output_dir=f"{args.output_dir}/track_b_taxonomy_analysis",
                    sample_years=[2015, 2020, 2025, 2035, 2050]
                )
                
                logger.info("🏗️ Track B Steel Taxonomy Analysis completed successfully!")
                logger.info("   Comprehensive hierarchical structure analysis with:")
                logger.info(f"   • {len(taxonomy_results['csv_exports'])} CSV analysis files")
                logger.info(f"   • {len(taxonomy_results['visualizations'])} visualization charts")
                logger.info(f"   • {len(taxonomy_results['mermaid_diagrams'])} mermaid hierarchy diagrams")
                logger.info("   • Complete product relationship mappings")
                logger.info("   • Growth pattern analysis across all levels")
                logger.info("   • Market share evolution tracking")
                logger.info(f"   Results saved to: {taxonomy_results['output_directory']}")
                
                # Note: Taxonomy analysis will be copied to outputs later with other files
                        
            except ImportError:
                logger.warning("Track B taxonomy analyzer module not available - skipping taxonomy analysis")
            except Exception as e:
                logger.error(f"Error creating Track B taxonomy analysis: {str(e)}")
            
            # Log validation summary
            if 'validation_summary' in validation_report:
                summary = validation_report['validation_summary']
                logger.info("="*60)
                logger.info("VALIDATION SUMMARY")
                logger.info("="*60)
                logger.info(f"Overall Score: {summary['overall_score']:.1f}%")
                logger.info(f"Validation Status: {summary['validation_status']}")
                
                for category, score in summary['category_scores'].items():
                    logger.info(f"{category}: {score:.1f}%")
                
                if summary['recommendations']:
                    logger.info("\nRecommendations:")
                    for rec in summary['recommendations']:
                        logger.info(f"- {rec}")
                logger.info("="*60)
            
            # Export detailed results if requested
            if args.export_detailed:
                logger.info("Exporting detailed intermediate results...")
                
                # Export hierarchical forecasts separately
                hierarchical_forecasts = unified_results['hierarchical_forecasts']
                for level, forecast_df in hierarchical_forecasts.items():
                    detailed_filename = f"{args.output_dir}/detailed_{level}_forecasts.csv"
                    forecast_df.to_csv(detailed_filename, index=False)
                
                # Export cross-validation details
                cross_validation = unified_results['cross_validation']
                cross_val_df = pd.DataFrame.from_dict(cross_validation, orient='index')
                cross_val_df.to_csv(f"{args.output_dir}/detailed_cross_validation.csv")
                
                # Export renewable energy analysis
                renewable_analysis = {}
                if 'level_0' in hierarchical_forecasts:
                    renewable_analysis['renewable_steel_share'] = (
                        macro_data['total_renewable_steel_demand'].sum() / 
                        hierarchical_forecasts['level_0']['total_steel_demand'].sum() * 100
                    )
                
                renewable_df = pd.DataFrame.from_dict(renewable_analysis, orient='index', columns=['value'])
                renewable_df.to_csv(f"{args.output_dir}/renewable_energy_analysis.csv")
        
        else:
            logger.info("Validation-only mode: Loading existing forecasts for validation...")
            # In validation-only mode, you would load existing forecast files
            # This is a placeholder for validation-only functionality
            logger.warning("Validation-only mode not fully implemented - requires existing forecast files")
        
        # Generate executive summary
        logger.info("Generating executive summary...")
        
        if not args.validate_only:
            summary_data = {
                'Generation Timestamp': datetime.now().isoformat(),
                'Forecast Period': f"{args.start_year}-{args.end_year}",
                'Total Years Forecast': args.end_year - args.start_year + 1,
                'Current Products Forecast': 13,
                'Hierarchical Levels': 4,
                'Configuration Source': args.config_path,
                'Output Location': args.output_dir,
                'Validation Status': validation_report.get('validation_summary', {}).get('validation_status', 'UNKNOWN'),
                'Overall Validation Score': f"{validation_report.get('validation_summary', {}).get('overall_score', 0):.1f}%"
            }
            
            if 'level_0' in unified_results['hierarchical_forecasts']:
                level_0_data = unified_results['hierarchical_forecasts']['level_0']
                summary_data.update({
                    'Average Annual Steel Demand (kt)': level_0_data['total_steel_demand'].mean(),
                    'Total Steel Demand 2025-2050 (Mt)': level_0_data['total_steel_demand'].sum() / 1000,
                    'Steel Demand Growth Rate (CAGR %)': ((level_0_data['total_steel_demand'].iloc[-1] / level_0_data['total_steel_demand'].iloc[0]) ** (1/25) - 1) * 100
                })
        
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index', columns=['value'])
        summary_df.to_csv(f"{args.output_dir}/executive_summary.csv")
        
        # Generate PDF reports from markdown files
        logger.info("Generating PDF reports from markdown files...")
        _generate_pdf_reports(args.output_dir)
        
        logger.info("="*80)
        logger.info("FORECASTING SYSTEM EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"All results saved to: {args.output_dir}")
        logger.info(f"Log file: logs/hierarchical_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        if not args.validate_only and 'validation_summary' in validation_report:
            validation_status = validation_report['validation_summary']['validation_status']
            if validation_status in ['EXCELLENT', 'GOOD']:
                logger.info("✅ VALIDATION PASSED - Forecasts meet quality standards")
            elif validation_status == 'ACCEPTABLE':
                logger.info("⚠️  VALIDATION ACCEPTABLE - Review recommendations for improvements")
            else:
                logger.warning("❌ VALIDATION NEEDS IMPROVEMENT - Review validation report")
        
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in forecasting system execution: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Print stack trace for debugging
        import traceback
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)