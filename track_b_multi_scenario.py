#!/usr/bin/env python3
"""
Australian Steel Demand Forecasting System - Multi-Scenario Execution Script
Runs Track B hierarchical forecasting across multiple research-based scenarios.
Supports comparison analysis and comprehensive validation across scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
import shutil

# Add current directory to path for imports
sys.path.append('.')

from forecasting.dual_track_forecasting import DualTrackSteelForecastingSystem
from validation.comprehensive_validation import ComprehensiveValidationFramework

class MultiScenarioTrackB:
    """
    Multi-scenario Track B execution system.
    Manages execution of Track B across different research-based sectoral weight scenarios.
    """
    
    def __init__(self, config_path: str = "config/"):
        self.config_path = config_path
        self.scenarios_path = os.path.join(config_path, "scenarios")
        self.logger = self._setup_logging()
        
        # Available scenarios
        self.available_scenarios = [
            "baseline",
            "claude", 
            "perplexity",
            "chatgpt",
            "gemini"
        ]
        
        # Scenario metadata
        self.scenario_metadata = {
            "baseline": {
                "name": "Baseline Configuration",
                "source": "Current SDM Track B configuration",
                "description": "Research-validated weights with IA infrastructure data and post-automotive manufacturing"
            },
            "claude": {
                "name": "Claude Research Analysis",
                "source": "Claude - Sectorial Weights.md",
                "description": "Comprehensive 2024 data update with normalized weights and detailed product mappings"
            },
            "perplexity": {
                "name": "Perplexity Analysis",
                "source": "Perplexity - Sectorial Weights.md", 
                "description": "Evidence-based registry with global benchmarks and Australian data convergence"
            },
            "chatgpt": {
                "name": "ChatGPT o3 Pro Analysis",
                "source": "ChatGPT o3 Pro - Sectorial Weights.docx",
                "description": "Updated decomposition factors with latest government statistics and industry data"
            },
            "gemini": {
                "name": "Gemini 25 Pro Analysis", 
                "source": "Gemini 25 Pro - Sectorial Weights.docx",
                "description": "Enhanced accuracy validation with cross-referenced sources and mining sector focus"
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for multi-scenario execution."""
        os.makedirs("logs", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/track_b_multi_scenario_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Multi-scenario logging initialized. Log file: {log_filename}")
        
        return logger
    
    def validate_scenario(self, scenario: str) -> bool:
        """Validate that a scenario has all required configuration files."""
        scenario_dir = os.path.join(self.scenarios_path, scenario)
        
        required_files = [
            "sectoral_weights.csv",
            "sector_to_level1_mapping.csv"
        ]
        
        for file in required_files:
            file_path = os.path.join(scenario_dir, file)
            if not os.path.exists(file_path):
                self.logger.error(f"Missing required file for scenario '{scenario}': {file}")
                return False
        
        return True
    
    def run_single_scenario(self, scenario: str, output_dir: str, 
                          start_year: int = 2025, end_year: int = 2050) -> Dict[str, Any]:
        """
        Run Track B for a single scenario.
        
        Args:
            scenario: Scenario name
            output_dir: Output directory for this scenario
            start_year: Start year for forecasting
            end_year: End year for forecasting
            
        Returns:
            Dictionary with scenario results and metadata
        """
        scenario_dir = os.path.join(self.scenarios_path, scenario)
        
        # Validate scenario
        if not self.validate_scenario(scenario):
            raise ValueError(f"Invalid scenario configuration: {scenario}")
        
        self.logger.info(f"Running scenario: {scenario}")
        self.logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Copy scenario configuration to output for reproducibility
        config_backup_dir = os.path.join(output_dir, "scenario_config")
        os.makedirs(config_backup_dir, exist_ok=True)
        
        for file in os.listdir(scenario_dir):
            if file.endswith('.csv'):
                shutil.copy2(
                    os.path.join(scenario_dir, file),
                    os.path.join(config_backup_dir, file)
                )
        
        # Create scenario metadata file
        metadata = {
            "scenario_name": scenario,
            "execution_timestamp": datetime.now().isoformat(),
            "forecast_period": f"{start_year}-{end_year}",
            "scenario_info": self.scenario_metadata.get(scenario, {}),
            "data_quality": self._assess_data_quality(scenario)
        }
        
        with open(os.path.join(output_dir, "scenario_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        try:
            # Create a temporary copy of the base config directory with scenario overrides
            temp_config_dir = "temp_config_" + scenario
            if os.path.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            
            # Copy entire base config directory
            shutil.copytree("config", temp_config_dir)
            
            # Override with scenario-specific files
            for file in os.listdir(scenario_dir):
                if file.endswith('.csv'):
                    shutil.copy2(
                        os.path.join(scenario_dir, file),
                        os.path.join(temp_config_dir, file)
                    )
            
            # Initialize dual-track forecasting system with scenario config
            forecasting_system = DualTrackSteelForecastingSystem(
                config_path=temp_config_dir + "/"
            )
            
            # Initialize validation framework
            validation_framework = ComprehensiveValidationFramework(
                config_path=temp_config_dir + "/"
            )
            
            self.logger.info(f"Generating unified forecasts for {start_year}-{end_year}...")
            
            # Generate forecasts
            forecast_results = forecasting_system.generate_unified_forecasts(
                start_year=start_year,
                end_year=end_year
            )
            
            # Export unified results to scenario output directory
            forecasting_system.export_unified_results(
                unified_results=forecast_results,
                output_dir=output_dir
            )
            
            # Run validation (if available)
            validation_results = {}
            try:
                if hasattr(validation_framework, 'run_all_validations'):
                    self.logger.info("Running comprehensive validation...")
                    validation_results = validation_framework.run_all_validations()
                else:
                    self.logger.info("Validation framework available but method not found - skipping validation")
            except Exception as e:
                self.logger.warning(f"Validation failed: {e}")
                validation_results = {"status": "validation_failed", "error": str(e)}
            
            # Generate PDF reports
            self._generate_pdf_reports(output_dir)
            
            self.logger.info(f"Scenario '{scenario}' completed successfully")
            
            # Cleanup temporary config directory
            if os.path.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            
            return {
                "scenario": scenario,
                "status": "success",
                "output_dir": output_dir,
                "metadata": metadata,
                "forecast_results": forecast_results,
                "validation_results": validation_results
            }
            
        except Exception as e:
            self.logger.error(f"Error running scenario '{scenario}': {str(e)}")
            
            # Cleanup temporary config directory
            temp_config_dir = "temp_config_" + scenario
            if os.path.exists(temp_config_dir):
                shutil.rmtree(temp_config_dir)
            
            # Save error info
            error_info = {
                "scenario": scenario,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, "error_log.json"), 'w') as f:
                json.dump(error_info, f, indent=2)
            
            return {
                "scenario": scenario,
                "status": "error",
                "error": str(e),
                "output_dir": output_dir
            }
    
    def _assess_data_quality(self, scenario: str) -> Dict[str, Any]:
        """Assess data quality and completeness for a scenario."""
        scenario_dir = os.path.join(self.scenarios_path, scenario)
        
        quality_assessment = {
            "sectoral_weights": {"available": True, "source": "research"},
            "level_0_to_1_mapping": {"available": True, "source": "research"},
            "level_1_to_2_breakdown": {"available": False, "source": "baseline"},
            "level_2_to_3_specifications": {"available": False, "source": "baseline"}
        }
        
        # Check for Level 2 breakdown
        level_2_file = os.path.join(scenario_dir, "level_2_products.csv")
        if os.path.exists(level_2_file):
            quality_assessment["level_1_to_2_breakdown"] = {
                "available": True, 
                "source": "research" if scenario != "baseline" else "baseline"
            }
        
        # Check for Level 3 specifications
        level_3_file = os.path.join(scenario_dir, "level_3_specifications.csv")
        if os.path.exists(level_3_file):
            quality_assessment["level_2_to_3_specifications"] = {
                "available": True,
                "source": "research" if scenario in ["claude", "perplexity"] else "baseline"
            }
        
        return quality_assessment
    
    def run_multiple_scenarios(self, scenarios: List[str], output_base_dir: str,
                              start_year: int = 2025, end_year: int = 2050,
                              parallel: bool = False) -> Dict[str, Any]:
        """
        Run Track B for multiple scenarios.
        
        Args:
            scenarios: List of scenario names
            output_base_dir: Base output directory
            start_year: Start year for forecasting
            end_year: End year for forecasting
            parallel: Run scenarios in parallel
            
        Returns:
            Dictionary with all scenario results
        """
        # Create base output directory
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Validate all scenarios first
        for scenario in scenarios:
            if scenario not in self.available_scenarios:
                raise ValueError(f"Unknown scenario: {scenario}. Available: {self.available_scenarios}")
            
            if not self.validate_scenario(scenario):
                raise ValueError(f"Invalid configuration for scenario: {scenario}")
        
        self.logger.info(f"Running {len(scenarios)} scenarios: {', '.join(scenarios)}")
        
        results = {}
        
        if parallel and len(scenarios) > 1:
            # Run scenarios in parallel
            self.logger.info("Running scenarios in parallel...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(scenarios), 3)) as executor:
                future_to_scenario = {}
                
                for scenario in scenarios:
                    scenario_output_dir = os.path.join(output_base_dir, scenario)
                    future = executor.submit(
                        self.run_single_scenario, 
                        scenario, 
                        scenario_output_dir,
                        start_year,
                        end_year
                    )
                    future_to_scenario[future] = scenario
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_scenario):
                    scenario = future_to_scenario[future]
                    try:
                        results[scenario] = future.result()
                    except Exception as e:
                        self.logger.error(f"Scenario {scenario} failed: {e}")
                        results[scenario] = {
                            "scenario": scenario,
                            "status": "error", 
                            "error": str(e)
                        }
        else:
            # Run scenarios sequentially
            for i, scenario in enumerate(scenarios, 1):
                self.logger.info(f"Running scenario {i}/{len(scenarios)}: {scenario}")
                
                scenario_output_dir = os.path.join(output_base_dir, scenario)
                results[scenario] = self.run_single_scenario(
                    scenario, 
                    scenario_output_dir,
                    start_year, 
                    end_year
                )
        
        # Save execution summary
        execution_summary = {
            "execution_timestamp": datetime.now().isoformat(),
            "scenarios_run": scenarios,
            "forecast_period": f"{start_year}-{end_year}",
            "execution_mode": "parallel" if parallel else "sequential",
            "results_summary": {
                scenario: result.get("status", "unknown") 
                for scenario, result in results.items()
            }
        }
        
        with open(os.path.join(output_base_dir, "execution_summary.json"), 'w') as f:
            json.dump(execution_summary, f, indent=2)
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any], output_dir: str):
        """Generate comparison report across scenarios."""
        self.logger.info("Generating scenario comparison report...")
        
        comparison_dir = os.path.join(output_dir, "comparison_report")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Create comparison summary
        successful_scenarios = [
            scenario for scenario, result in results.items() 
            if result.get("status") == "success"
        ]
        
        if len(successful_scenarios) < 2:
            self.logger.warning("Need at least 2 successful scenarios for comparison")
            return
        
        # Collect sectoral weights comparison
        sectoral_weights_comparison = []
        
        for scenario in successful_scenarios:
            scenario_dir = results[scenario]["output_dir"]
            weights_file = os.path.join(scenario_dir, "scenario_config", "sectoral_weights.csv")
            
            if os.path.exists(weights_file):
                weights_df = pd.read_csv(weights_file)
                # Use 2025-2030 period for comparison
                base_weights = weights_df[weights_df['period'] == '2025-2030'].iloc[0]
                
                sectoral_weights_comparison.append({
                    'scenario': scenario,
                    'construction': base_weights['gdp_construction'],
                    'infrastructure': base_weights['infrastructure_traditional'],
                    'manufacturing': base_weights['manufacturing_ip'],
                    'renewable_energy': base_weights['wm_renewable_energy'],
                    'other_sectors': base_weights['other_sectors']
                })
        
        # Save sectoral weights comparison
        comparison_df = pd.DataFrame(sectoral_weights_comparison)
        comparison_df.to_csv(
            os.path.join(comparison_dir, "sectoral_weights_comparison.csv"), 
            index=False
        )
        
        # Generate comparison markdown report
        self._generate_comparison_markdown(comparison_df, successful_scenarios, comparison_dir)
        
        self.logger.info(f"Comparison report generated in {comparison_dir}")
    
    def _generate_comparison_markdown(self, weights_df: pd.DataFrame, 
                                    scenarios: List[str], output_dir: str):
        """Generate markdown comparison report."""
        
        report_content = f"""# Track B Multi-Scenario Comparison Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report compares Track B hierarchical steel demand forecasting results across {len(scenarios)} research-based scenarios.

### Scenarios Analyzed

"""
        
        for scenario in scenarios:
            metadata = self.scenario_metadata.get(scenario, {})
            report_content += f"""
#### {metadata.get('name', scenario.title())}
- **Source**: {metadata.get('source', 'N/A')}
- **Description**: {metadata.get('description', 'N/A')}
"""
        
        report_content += """
## Sectoral Weights Comparison (2025-2030 Period)

| Scenario | Construction | Infrastructure | Manufacturing | Renewable Energy | Other Sectors |
|----------|--------------|----------------|---------------|------------------|---------------|
"""
        
        for _, row in weights_df.iterrows():
            report_content += f"| {row['scenario'].title()} | {row['construction']:.1%} | {row['infrastructure']:.1%} | {row['manufacturing']:.1%} | {row['renewable_energy']:.1%} | {row['other_sectors']:.1%} |\n"
        
        report_content += """
## Key Differences

### Manufacturing Sector Variation
The manufacturing sector shows the highest variation across scenarios, ranging from:
"""
        min_mfg = weights_df['manufacturing'].min()
        max_mfg = weights_df['manufacturing'].max()
        min_scenario = weights_df.loc[weights_df['manufacturing'].idxmin(), 'scenario']
        max_scenario = weights_df.loc[weights_df['manufacturing'].idxmax(), 'scenario']
        
        report_content += f"- **Lowest**: {min_mfg:.1%} ({min_scenario.title()})\n"
        report_content += f"- **Highest**: {max_mfg:.1%} ({max_scenario.title()})\n"
        report_content += f"- **Range**: {(max_mfg - min_mfg):.1%} percentage points\n\n"
        
        report_content += """
### Renewable Energy Growth
Renewable energy sectoral weights show significant variation reflecting different assumptions about clean energy transition speed.

### Infrastructure Consistency
Infrastructure weights remain relatively consistent across scenarios, reflecting the high confidence in Infrastructure Australia data.

---

*Generated by Australian Steel Demand Model (SDM) Multi-Scenario Analysis System*
"""
        
        with open(os.path.join(output_dir, "scenario_comparison_summary.md"), 'w') as f:
            f.write(report_content)
    
    def _generate_pdf_reports(self, output_dir: str) -> None:
        """Generate PDF reports from markdown files."""
        import subprocess
        
        # Create pdf_reports directory
        pdf_dir = Path(output_dir) / "pdf_reports"
        pdf_dir.mkdir(exist_ok=True)
        
        # Find all markdown files
        md_files = list(Path(output_dir).rglob("*.md"))
        
        if md_files:
            cmd = [
                sys.executable,
                "convert_md_to_pdf_final.py",
                str(output_dir),
                str(pdf_dir)
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    self.logger.info(f"PDF reports generated in {pdf_dir}")
                else:
                    self.logger.warning(f"PDF generation issues: {result.stderr}")
            except Exception as e:
                self.logger.warning(f"Could not generate PDFs: {e}")


def main():
    """Main execution function for multi-scenario Track B forecasting."""
    
    parser = argparse.ArgumentParser(
        description='Australian Steel Demand Model - Multi-Scenario Track B Analysis'
    )
    
    # Scenario selection
    parser.add_argument('--scenario', nargs='+', 
                       choices=['baseline', 'claude', 'perplexity', 'chatgpt', 'gemini', 'all'],
                       default=['baseline'],
                       help='Scenarios to run (default: baseline). Use "all" for all scenarios.')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run scenarios in parallel (faster for multiple scenarios)')
    parser.add_argument('--compare', action='store_true',
                       help='Generate comparison report (automatically enabled for multiple scenarios)')
    
    # Forecast parameters
    parser.add_argument('--start-year', type=int, default=2025,
                       help='Start year for forecasting (default: 2025)')
    parser.add_argument('--end-year', type=int, default=2050,
                       help='End year for forecasting (default: 2050)')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Base output directory (default: timestamped folder)')
    parser.add_argument('--config-path', type=str, default='config/',
                       help='Path to configuration files')
    
    args = parser.parse_args()
    
    # Handle 'all' scenario
    if 'all' in args.scenario:
        scenarios = ['baseline', 'claude', 'perplexity', 'chatgpt', 'gemini']
    else:
        scenarios = args.scenario
    
    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(scenarios) == 1:
            output_dir = f'forecasts/track_b_{scenarios[0]}_{timestamp}'
        else:
            output_dir = f'forecasts/track_b_multi_scenario_{timestamp}'
    else:
        output_dir = args.output_dir
    
    # Initialize multi-scenario system
    multi_scenario = MultiScenarioTrackB(config_path=args.config_path)
    
    try:
        print("="*80)
        print("AUSTRALIAN STEEL DEMAND MODEL - TRACK B MULTI-SCENARIO ANALYSIS")
        print("="*80)
        print(f"Scenarios: {', '.join(scenarios)}")
        print(f"Forecast Period: {args.start_year}-{args.end_year}")
        print(f"Output Directory: {output_dir}")
        print(f"Execution Mode: {'Parallel' if args.parallel else 'Sequential'}")
        print("="*80)
        
        if len(scenarios) == 1:
            # Single scenario execution
            scenario_output_dir = output_dir
            result = multi_scenario.run_single_scenario(
                scenarios[0], 
                scenario_output_dir,
                args.start_year,
                args.end_year
            )
            
            if result['status'] == 'success':
                print(f"\\n✓ Scenario '{scenarios[0]}' completed successfully")
                print(f"Results saved to: {scenario_output_dir}")
            else:
                print(f"\\n✗ Scenario '{scenarios[0]}' failed: {result.get('error', 'Unknown error')}")
        
        else:
            # Multi-scenario execution
            results = multi_scenario.run_multiple_scenarios(
                scenarios,
                output_dir,
                args.start_year,
                args.end_year,
                args.parallel
            )
            
            # Print results summary
            print("\\nExecution Summary:")
            print("-" * 40)
            for scenario, result in results.items():
                status = result.get('status', 'unknown')
                if status == 'success':
                    print(f"✓ {scenario.title()}: Completed successfully")
                else:
                    print(f"✗ {scenario.title()}: Failed ({result.get('error', 'Unknown error')})")
            
            # Generate comparison report if requested or multiple scenarios
            if args.compare or len([r for r in results.values() if r.get('status') == 'success']) > 1:
                multi_scenario.generate_comparison_report(results, output_dir)
                print(f"\\n📊 Comparison report generated")
            
            print(f"\\nAll results saved to: {output_dir}")
        
        print("\\n" + "="*80)
        print("EXECUTION COMPLETED")
        print("="*80)
        
    except Exception as e:
        print(f"\\n✗ Execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()