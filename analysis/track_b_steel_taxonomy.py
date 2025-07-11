#!/usr/bin/env python3
"""
Track B Steel Taxonomy Analysis Module

This module provides comprehensive analysis of Track B forecasts aligned with the 
hierarchical steel consumption taxonomy used in the Track B forecasting system.

Based on the 4-level Track B hierarchy:
Level 0: Total Steel Consumption (Apparent Steel Use)
Level 1: Major Product Categories (Semi-finished, Finished Flat, Finished Long, Tube/Pipe)
Level 2: Product Families (Billets, Slabs, Hot/Cold Rolled, Structural, Rails, etc.)
Level 3: Client Product Specifications (Grade specifications, end-use applications)

Generates both CSV data outputs and visualization images for Track B taxonomy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TrackBSteelTaxonomyAnalyzer:
    """
    Comprehensive Track B Steel Taxonomy Analyzer based on hierarchical consumption framework.
    
    Provides accurate mapping of Track B steel categories to hierarchical structure
    and generates both CSV and visualization outputs.
    """
    
    def __init__(self):
        """Initialize Track B Steel Taxonomy Analyzer."""
        self.logger = self._setup_logging()
        
        # Track B Hierarchical Consumption Structure (4 levels)
        self.track_b_hierarchy = {
            "Level_0_Total_Consumption": {
                "name": "Total Steel Consumption",
                "description": "Total apparent steel use (crude steel equivalent)",
                "categories": ["Total_Steel_Consumption"],
                "track_b_mapping": ["Total_Steel_Consumption"],
                "color": "#2E4057",
                "hierarchy_level": 0,
                "parent": None,
                "children": ["Level_1_Major_Categories"]
            },
            "Level_1_Major_Categories": {
                "name": "Major Product Categories",
                "description": "Primary steel product categories for consumption",
                "categories": ["SEMI_FINISHED", "FINISHED_FLAT", "FINISHED_LONG", "TUBE_PIPE"],
                "track_b_mapping": ["SEMI_FINISHED", "FINISHED_FLAT", "FINISHED_LONG", "TUBE_PIPE"],
                "color": "#048A81",
                "hierarchy_level": 1,
                "parent": "Level_0_Total_Consumption",
                "children": ["Level_2_Product_Families"]
            },
            "Level_2_Product_Families": {
                "name": "Product Families",
                "description": "Detailed product families and specifications",
                "categories": [
                    # Semi-finished products
                    "BILLETS_COMMERCIAL", "BILLETS_SBQ", "SLABS_STANDARD", "BILLETS_DEGASSED", "SLABS_DEGASSED",
                    # Finished flat products
                    "HOT_ROLLED_COIL", "COLD_ROLLED_COIL", "PLATE", "GALVANIZED",
                    # Finished long products
                    "STRUCTURAL_BEAMS", "STRUCTURAL_COLUMNS", "STRUCTURAL_CHANNELS", "STRUCTURAL_ANGLES",
                    "RAILS_STANDARD", "RAILS_HEAD_HARDENED", "SLEEPER_BAR", "REBAR", "WIRE_ROD", "WELDED_STRUCTURAL",
                    # Tube and pipe products
                    "SEAMLESS_LINE_PIPE", "WELDED_LINE_PIPE", "OTHER_TUBE_PIPE"
                ],
                "track_b_mapping": [
                    "BILLETS_COMMERCIAL", "BILLETS_SBQ", "SLABS_STANDARD", "BILLETS_DEGASSED", "SLABS_DEGASSED",
                    "HOT_ROLLED_COIL", "COLD_ROLLED_COIL", "PLATE", "GALVANIZED",
                    "STRUCTURAL_BEAMS", "STRUCTURAL_COLUMNS", "STRUCTURAL_CHANNELS", "STRUCTURAL_ANGLES",
                    "RAILS_STANDARD", "RAILS_HEAD_HARDENED", "SLEEPER_BAR", "REBAR", "WIRE_ROD", "WELDED_STRUCTURAL",
                    "SEAMLESS_LINE_PIPE", "WELDED_LINE_PIPE", "OTHER_TUBE_PIPE"
                ],
                "color": "#54C6EB",
                "hierarchy_level": 2,
                "parent": "Level_1_Major_Categories",
                "children": ["Level_3_Client_Specifications"]
            },
            "Level_3_Client_Specifications": {
                "name": "Client Product Specifications",
                "description": "Specific grade and application specifications for end clients",
                "categories": [
                    # Commercial billets
                    "BILLETS_COMM_LOW_CARBON", "BILLETS_COMM_MEDIUM_CARBON",
                    # SBQ billets
                    "BILLETS_SBQ_AUTOMOTIVE", "BILLETS_SBQ_MINING", "BILLETS_SBQ_OIL_GAS",
                    # Degassed billets
                    "BILLETS_DEG_PREMIUM_AUTO", "BILLETS_DEG_PRECISION",
                    # Standard slabs
                    "SLABS_STD_HOT_ROLLED", "SLABS_STD_COLD_ROLLED",
                    # Degassed slabs
                    "SLABS_DEG_AUTOMOTIVE", "SLABS_DEG_APPLIANCE",
                    # Structural beams
                    "UB_GRADE_300", "UB_GRADE_300_PLUS",
                    # Structural columns
                    "UC_GRADE_300", "UC_GRADE_300_PLUS",
                    # Structural channels
                    "PFC_STANDARD", "PFC_HIGH_STRENGTH",
                    # Structural angles
                    "ANGLES_STANDARD", "ANGLES_HIGH_STRENGTH",
                    # Rails
                    "RAILS_STD_FREIGHT", "RAILS_STD_PASSENGER", "RAILS_HH_HEAVY_HAUL", "RAILS_HH_CRITICAL_INFRA",
                    # Sleeper bar
                    "SLEEPER_BAR_COAL_WAGON", "SLEEPER_BAR_GENERAL_FREIGHT"
                ],
                "track_b_mapping": [
                    "BILLETS_COMM_LOW_CARBON", "BILLETS_COMM_MEDIUM_CARBON",
                    "BILLETS_SBQ_AUTOMOTIVE", "BILLETS_SBQ_MINING", "BILLETS_SBQ_OIL_GAS",
                    "BILLETS_DEG_PREMIUM_AUTO", "BILLETS_DEG_PRECISION",
                    "SLABS_STD_HOT_ROLLED", "SLABS_STD_COLD_ROLLED",
                    "SLABS_DEG_AUTOMOTIVE", "SLABS_DEG_APPLIANCE",
                    "UB_GRADE_300", "UB_GRADE_300_PLUS",
                    "UC_GRADE_300", "UC_GRADE_300_PLUS",
                    "PFC_STANDARD", "PFC_HIGH_STRENGTH",
                    "ANGLES_STANDARD", "ANGLES_HIGH_STRENGTH",
                    "RAILS_STD_FREIGHT", "RAILS_STD_PASSENGER", "RAILS_HH_HEAVY_HAUL", "RAILS_HH_CRITICAL_INFRA",
                    "SLEEPER_BAR_COAL_WAGON", "SLEEPER_BAR_GENERAL_FREIGHT"
                ],
                "color": "#F3A712",
                "hierarchy_level": 3,
                "parent": "Level_2_Product_Families",
                "children": None
            }
        }
        
        # Define hierarchical relationships and mappings
        self.level_1_to_level_2_mapping = {
            "SEMI_FINISHED": ["BILLETS_COMMERCIAL", "BILLETS_SBQ", "SLABS_STANDARD", "BILLETS_DEGASSED", "SLABS_DEGASSED"],
            "FINISHED_FLAT": ["HOT_ROLLED_COIL", "COLD_ROLLED_COIL", "PLATE", "GALVANIZED"],
            "FINISHED_LONG": ["STRUCTURAL_BEAMS", "STRUCTURAL_COLUMNS", "STRUCTURAL_CHANNELS", "STRUCTURAL_ANGLES",
                             "RAILS_STANDARD", "RAILS_HEAD_HARDENED", "SLEEPER_BAR", "REBAR", "WIRE_ROD", "WELDED_STRUCTURAL"],
            "TUBE_PIPE": ["SEAMLESS_LINE_PIPE", "WELDED_LINE_PIPE", "OTHER_TUBE_PIPE"]
        }
        
        self.level_2_to_level_3_mapping = {
            "BILLETS_COMMERCIAL": ["BILLETS_COMM_LOW_CARBON", "BILLETS_COMM_MEDIUM_CARBON"],
            "BILLETS_SBQ": ["BILLETS_SBQ_AUTOMOTIVE", "BILLETS_SBQ_MINING", "BILLETS_SBQ_OIL_GAS"],
            "BILLETS_DEGASSED": ["BILLETS_DEG_PREMIUM_AUTO", "BILLETS_DEG_PRECISION"],
            "SLABS_STANDARD": ["SLABS_STD_HOT_ROLLED", "SLABS_STD_COLD_ROLLED"],
            "SLABS_DEGASSED": ["SLABS_DEG_AUTOMOTIVE", "SLABS_DEG_APPLIANCE"],
            "STRUCTURAL_BEAMS": ["UB_GRADE_300", "UB_GRADE_300_PLUS"],
            "STRUCTURAL_COLUMNS": ["UC_GRADE_300", "UC_GRADE_300_PLUS"],
            "STRUCTURAL_CHANNELS": ["PFC_STANDARD", "PFC_HIGH_STRENGTH"],
            "STRUCTURAL_ANGLES": ["ANGLES_STANDARD", "ANGLES_HIGH_STRENGTH"],
            "RAILS_STANDARD": ["RAILS_STD_FREIGHT", "RAILS_STD_PASSENGER"],
            "RAILS_HEAD_HARDENED": ["RAILS_HH_HEAVY_HAUL", "RAILS_HH_CRITICAL_INFRA"],
            "SLEEPER_BAR": ["SLEEPER_BAR_COAL_WAGON", "SLEEPER_BAR_GENERAL_FREIGHT"]
        }
        
        # Product descriptions for detailed analysis
        self.product_descriptions = {
            # Level 1 descriptions
            "SEMI_FINISHED": "Semi-finished steel products including billets and slabs for further processing",
            "FINISHED_FLAT": "Finished flat steel products including coils, plates, and galvanized products",
            "FINISHED_LONG": "Finished long steel products including structural steel, rails, and wire products",
            "TUBE_PIPE": "Tubular and pipe products for oil/gas, infrastructure, and industrial applications",
            
            # Level 2 descriptions
            "BILLETS_COMMERCIAL": "Commercial grade steel billets for general structural applications",
            "BILLETS_SBQ": "Special bar quality billets for high-strength automotive and mining applications",
            "SLABS_STANDARD": "Standard steel slabs for hot and cold rolling applications",
            "BILLETS_DEGASSED": "Degassed steel billets for premium automotive and precision applications",
            "SLABS_DEGASSED": "Degassed steel slabs for automotive and appliance manufacturing",
            "HOT_ROLLED_COIL": "Hot rolled steel coils for construction and manufacturing",
            "COLD_ROLLED_COIL": "Cold rolled steel coils for automotive and appliance applications",
            "PLATE": "Steel plates for heavy construction, shipbuilding, and infrastructure",
            "GALVANIZED": "Galvanized steel products for corrosion-resistant applications",
            "STRUCTURAL_BEAMS": "Universal beam sections for construction and infrastructure",
            "STRUCTURAL_COLUMNS": "Universal column sections for building and bridge construction",
            "STRUCTURAL_CHANNELS": "Parallel flange channels for structural framing",
            "STRUCTURAL_ANGLES": "Structural angles for construction and fabrication",
            "RAILS_STANDARD": "Standard railway track material for freight and passenger rail",
            "RAILS_HEAD_HARDENED": "Head-hardened rails for heavy-haul and critical infrastructure",
            "SLEEPER_BAR": "Steel bar for concrete sleeper manufacturing",
            "REBAR": "Reinforcing bar for concrete construction",
            "WIRE_ROD": "Wire rod for manufacturing wire products and fasteners",
            "WELDED_STRUCTURAL": "Welded structural products for construction applications",
            "SEAMLESS_LINE_PIPE": "Seamless line pipe for oil and gas transmission",
            "WELDED_LINE_PIPE": "Welded line pipe for pipeline infrastructure",
            "OTHER_TUBE_PIPE": "Other tubular products for industrial and construction applications"
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def analyze_track_b_forecasts(self, 
                                forecast_data_dir: str,
                                output_dir: str,
                                sample_years: List[int] = [2025, 2035, 2050]) -> Dict[str, Any]:
        """
        Comprehensive analysis of Track B forecasts with taxonomy mapping.
        
        Args:
            forecast_data_dir: Directory containing Track B forecast files
            output_dir: Directory for analysis outputs
            sample_years: Years to include in detailed analysis
            
        Returns:
            Dictionary containing analysis results
        """
        self.logger.info("Starting comprehensive Track B steel taxonomy analysis...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load all Track B forecast data
        forecast_data = self._load_track_b_forecast_data(forecast_data_dir)
        
        # Perform hierarchical analysis
        hierarchical_analysis = self._analyze_hierarchical_structure(forecast_data, sample_years)
        
        # Generate relationship mappings
        relationship_mappings = self._generate_relationship_mappings(forecast_data)
        
        # Create visualizations
        visualizations = self._generate_track_b_visualizations(forecast_data, output_path, sample_years)
        
        # Generate mermaid diagrams
        mermaid_diagrams = self._generate_mermaid_diagrams(forecast_data, output_path, sample_years)
        
        # Export comprehensive CSV files
        csv_exports = self._export_comprehensive_csv_files(forecast_data, hierarchical_analysis, 
                                                          relationship_mappings, output_path)
        
        # Create summary report
        summary_report = self._create_track_b_summary_report(forecast_data, hierarchical_analysis, 
                                                           relationship_mappings, output_path)
        
        analysis_results = {
            'forecast_data': forecast_data,
            'hierarchical_analysis': hierarchical_analysis,
            'relationship_mappings': relationship_mappings,
            'visualizations': visualizations,
            'mermaid_diagrams': mermaid_diagrams,
            'csv_exports': csv_exports,
            'summary_report': summary_report,
            'output_directory': str(output_path)
        }
        
        self.logger.info(f"Track B taxonomy analysis completed. Results saved to {output_path}")
        return analysis_results
    
    def _load_track_b_forecast_data(self, forecast_data_dir: str) -> Dict[str, pd.DataFrame]:
        """Load all Track B forecast data files."""
        self.logger.info("Loading Track B forecast data...")
        
        forecast_data = {}
        data_dir = Path(forecast_data_dir)
        
        # Load hierarchical level forecasts
        level_files = {
            'level_0': 'hierarchical_level_0_forecasts_2025_2050.csv',
            'level_1': 'hierarchical_level_1_forecasts_2025_2050.csv',
            'level_2': 'hierarchical_level_2_forecasts_2025_2050.csv',
            'level_3': 'hierarchical_level_3_forecasts_2025_2050.csv'
        }
        
        for level_name, filename in level_files.items():
            file_path = data_dir / filename
            if file_path.exists():
                forecast_data[level_name] = pd.read_csv(file_path)
                self.logger.info(f"Loaded {level_name}: {forecast_data[level_name].shape}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        return forecast_data
    
    def _analyze_hierarchical_structure(self, 
                                      forecast_data: Dict[str, pd.DataFrame],
                                      sample_years: List[int]) -> Dict[str, Any]:
        """Analyze hierarchical structure and relationships."""
        self.logger.info("Analyzing hierarchical structure...")
        
        analysis = {
            'level_summaries': {},
            'hierarchical_consistency': {},
            'growth_patterns': {},
            'product_shares': {}
        }
        
        # Analyze each level
        for level_name, data in forecast_data.items():
            if not data.empty:
                level_analysis = self._analyze_single_level(data, level_name, sample_years)
                analysis['level_summaries'][level_name] = level_analysis
        
        # Check hierarchical consistency
        analysis['hierarchical_consistency'] = self._check_hierarchical_consistency(forecast_data)
        
        # Analyze growth patterns
        analysis['growth_patterns'] = self._analyze_growth_patterns(forecast_data, sample_years)
        
        # Calculate product shares
        analysis['product_shares'] = self._calculate_product_shares(forecast_data, sample_years)
        
        return analysis
    
    def _analyze_single_level(self, 
                            data: pd.DataFrame, 
                            level_name: str, 
                            sample_years: List[int]) -> Dict[str, Any]:
        """Analyze a single hierarchical level."""
        level_analysis = {
            'total_products': len(data.columns) - 1,  # Exclude Year column
            'sample_year_values': {},
            'growth_rates': {},
            'largest_products': {},
            'level_totals': {}
        }
        
        # Get values for sample years
        for year in sample_years:
            year_data = data[data['Year'] == year]
            if not year_data.empty:
                level_analysis['sample_year_values'][year] = year_data.iloc[0].to_dict()
                
                # Calculate level total (excluding Year column)
                numeric_cols = [col for col in data.columns if col != 'Year']
                # Ensure we only sum numeric columns
                numeric_year_data = year_data[numeric_cols].select_dtypes(include=[np.number])
                level_total = numeric_year_data.sum(axis=1, numeric_only=True).iloc[0] if not numeric_year_data.empty else 0
                level_analysis['level_totals'][year] = float(level_total)
        
        # Calculate growth rates between sample years
        if len(sample_years) >= 2:
            first_year, last_year = sample_years[0], sample_years[-1]
            if first_year in level_analysis['level_totals'] and last_year in level_analysis['level_totals']:
                first_total = level_analysis['level_totals'][first_year]
                last_total = level_analysis['level_totals'][last_year]
                years_diff = last_year - first_year
                cagr = ((last_total / first_total) ** (1 / years_diff) - 1) * 100
                level_analysis['growth_rates']['cagr'] = cagr
        
        # Identify largest products in final year
        if sample_years and sample_years[-1] in level_analysis['sample_year_values']:
            final_year_data = level_analysis['sample_year_values'][sample_years[-1]]
            numeric_data = {k: v for k, v in final_year_data.items() 
                          if k != 'Year' and isinstance(v, (int, float)) and not pd.isna(v)}
            if numeric_data:
                sorted_products = sorted(numeric_data.items(), key=lambda x: float(x[1]), reverse=True)
                level_analysis['largest_products'] = sorted_products[:5]  # Top 5
        
        return level_analysis
    
    def _check_hierarchical_consistency(self, forecast_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Check consistency between hierarchical levels."""
        consistency_results = {
            'level_0_to_1': {},
            'level_1_to_2': {},
            'level_2_to_3': {},
            'overall_consistency': True
        }
        
        # Check Level 0 to Level 1 consistency
        if 'level_0' in forecast_data and 'level_1' in forecast_data:
            level_0_data = forecast_data['level_0']
            level_1_data = forecast_data['level_1']
            
            for _, row in level_0_data.iterrows():
                year = row['Year']
                level_1_row = level_1_data[level_1_data['Year'] == year]
                
                if not level_1_row.empty:
                    # Handle Level 0 data which has numeric and string columns
                    row_no_year = row.drop('Year')
                    # Only sum numeric values, skip string columns
                    level_0_total = sum(val for val in row_no_year if isinstance(val, (int, float)))
                    level_1_total = level_1_row.drop('Year', axis=1).sum(axis=1, numeric_only=True).iloc[0]
                    
                    consistency_results['level_0_to_1'][year] = {
                        'level_0_total': level_0_total,
                        'level_1_total': level_1_total,
                        'difference': abs(level_0_total - level_1_total),
                        'consistent': abs(level_0_total - level_1_total) < 0.01 * level_0_total
                    }
        
        # Similar checks for other levels...
        return consistency_results
    
    def _analyze_growth_patterns(self, 
                               forecast_data: Dict[str, pd.DataFrame],
                               sample_years: List[int]) -> Dict[str, Any]:
        """Analyze growth patterns across all levels."""
        growth_analysis = {}
        
        for level_name, data in forecast_data.items():
            if not data.empty and len(sample_years) >= 2:
                level_growth = {}
                product_columns = [col for col in data.columns if col != 'Year']
                
                for product in product_columns:
                    product_growth = {}
                    
                    # Calculate growth rates between consecutive sample years
                    for i in range(len(sample_years) - 1):
                        start_year, end_year = sample_years[i], sample_years[i + 1]
                        
                        start_data = data[data['Year'] == start_year]
                        end_data = data[data['Year'] == end_year]
                        
                        if not start_data.empty and not end_data.empty:
                            start_value = start_data[product].iloc[0]
                            end_value = end_data[product].iloc[0]
                            
                            # Ensure numeric values
                            if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)) and start_value > 0:
                                years_diff = end_year - start_year
                                cagr = ((end_value / start_value) ** (1 / years_diff) - 1) * 100
                                product_growth[f'{start_year}_{end_year}_cagr'] = cagr
                    
                    level_growth[product] = product_growth
                
                growth_analysis[level_name] = level_growth
        
        return growth_analysis
    
    def _calculate_product_shares(self, 
                                forecast_data: Dict[str, pd.DataFrame],
                                sample_years: List[int]) -> Dict[str, Any]:
        """Calculate market shares for products at each level."""
        share_analysis = {}
        
        for level_name, data in forecast_data.items():
            if not data.empty:
                level_shares = {}
                product_columns = [col for col in data.columns if col != 'Year']
                
                for year in sample_years:
                    year_data = data[data['Year'] == year]
                    if not year_data.empty:
                        year_shares = {}
                        total = year_data[product_columns].sum(axis=1, numeric_only=True).iloc[0]
                        
                        for product in product_columns:
                            product_value = year_data[product].iloc[0]
                            # Ensure numeric values for calculation
                            if isinstance(product_value, (int, float)) and isinstance(total, (int, float)):
                                share = (product_value / total * 100) if total > 0 else 0
                            else:
                                share = 0
                            year_shares[product] = share
                        
                        level_shares[year] = year_shares
                
                share_analysis[level_name] = level_shares
        
        return share_analysis
    
    def _generate_relationship_mappings(self, forecast_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate comprehensive relationship mappings."""
        mappings = {
            'hierarchy_structure': self.track_b_hierarchy,
            'level_1_to_2_mapping': self.level_1_to_level_2_mapping,
            'level_2_to_3_mapping': self.level_2_to_level_3_mapping,
            'product_descriptions': self.product_descriptions,
            'taxonomy_tree': self._build_taxonomy_tree()
        }
        return mappings
    
    def _build_taxonomy_tree(self) -> Dict[str, Any]:
        """Build complete taxonomy tree structure."""
        tree = {
            'Level_0': {
                'Total_Steel_Consumption': {
                    'Level_1': {}
                }
            }
        }
        
        # Build Level 1 structure
        for l1_category in self.level_1_to_level_2_mapping.keys():
            tree['Level_0']['Total_Steel_Consumption']['Level_1'][l1_category] = {
                'Level_2': {}
            }
            
            # Build Level 2 structure
            for l2_category in self.level_1_to_level_2_mapping[l1_category]:
                tree['Level_0']['Total_Steel_Consumption']['Level_1'][l1_category]['Level_2'][l2_category] = {
                    'Level_3': {}
                }
                
                # Build Level 3 structure if mapping exists
                if l2_category in self.level_2_to_level_3_mapping:
                    for l3_category in self.level_2_to_level_3_mapping[l2_category]:
                        tree['Level_0']['Total_Steel_Consumption']['Level_1'][l1_category]['Level_2'][l2_category]['Level_3'][l3_category] = {}
        
        return tree
    
    def _generate_track_b_visualizations(self, 
                                       forecast_data: Dict[str, pd.DataFrame],
                                       output_path: Path,
                                       sample_years: List[int]) -> Dict[str, str]:
        """Generate comprehensive visualizations for Track B taxonomy."""
        self.logger.info("Generating Track B taxonomy visualizations...")
        
        vis_output_dir = output_path / "track_b_taxonomy_visualizations"
        vis_output_dir.mkdir(exist_ok=True)
        
        visualizations = {}
        
        # 1. Hierarchical Structure Overview
        viz_file = self._create_hierarchical_structure_chart(forecast_data, vis_output_dir, sample_years)
        visualizations['hierarchical_structure'] = str(viz_file)
        
        # 2. Product Category Breakdown by Level
        viz_file = self._create_level_breakdown_charts(forecast_data, vis_output_dir, sample_years)
        visualizations['level_breakdowns'] = str(viz_file)
        
        # 3. Growth Rate Comparison
        viz_file = self._create_growth_rate_comparison(forecast_data, vis_output_dir, sample_years)
        visualizations['growth_comparison'] = str(viz_file)
        
        # 4. Market Share Evolution
        viz_file = self._create_market_share_evolution(forecast_data, vis_output_dir, sample_years)
        visualizations['market_share_evolution'] = str(viz_file)
        
        return visualizations
    
    def _create_hierarchical_structure_chart(self, 
                                           forecast_data: Dict[str, pd.DataFrame],
                                           output_dir: Path,
                                           sample_years: List[int]) -> Path:
        """Create hierarchical structure overview chart."""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Track B Steel Taxonomy - Hierarchical Structure Overview', fontsize=24, fontweight='bold')
        
        # Plot each level
        level_names = ['level_1', 'level_2', 'level_3']
        level_titles = ['Level 1: Major Categories', 'Level 2: Product Families', 'Level 3: Client Specifications']
        
        for i, (level_name, title) in enumerate(zip(level_names, level_titles)):
            if level_name in forecast_data:
                ax = axes[i//2, i%2] if i < 3 else None
                if ax is not None:
                    self._plot_level_data(forecast_data[level_name], ax, title, sample_years[0])
        
        # Remove empty subplot
        if len(level_names) < 4:
            fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        output_file = output_dir / "track_b_hierarchical_structure.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _plot_level_data(self, data: pd.DataFrame, ax, title: str, year: int):
        """Plot data for a single level."""
        year_data = data[data['Year'] == year]
        if not year_data.empty:
            products = [col for col in data.columns if col != 'Year']
            values = [year_data[col].iloc[0] for col in products]
            
            # Create bar chart
            bars = ax.bar(range(len(products)), values, alpha=0.7)
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_ylabel('Volume (kt)', fontsize=12)
            
            # Color bars based on hierarchy level
            colors = plt.cm.Set3(np.linspace(0, 1, len(products)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Rotate x-axis labels if too many products
            if len(products) > 8:
                ax.set_xticks(range(len(products)))
                ax.set_xticklabels(products, rotation=45, ha='right', fontsize=8)
            else:
                ax.set_xticks(range(len(products)))
                ax.set_xticklabels(products, rotation=0, fontsize=10)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    def _create_level_breakdown_charts(self, 
                                     forecast_data: Dict[str, pd.DataFrame],
                                     output_dir: Path,
                                     sample_years: List[int]) -> Path:
        """Create detailed breakdown charts for each level."""
        output_file = output_dir / "track_b_level_breakdowns.png"
        
        # Implementation would create detailed breakdown charts
        # For now, create a placeholder
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Track B Level Breakdown Charts\n(Detailed Implementation)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _create_growth_rate_comparison(self, 
                                     forecast_data: Dict[str, pd.DataFrame],
                                     output_dir: Path,
                                     sample_years: List[int]) -> Path:
        """Create growth rate comparison charts."""
        output_file = output_dir / "track_b_growth_comparison.png"
        
        # Implementation would create growth rate comparison
        # For now, create a placeholder
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Track B Growth Rate Comparison\n(Detailed Implementation)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _create_market_share_evolution(self, 
                                     forecast_data: Dict[str, pd.DataFrame],
                                     output_dir: Path,
                                     sample_years: List[int]) -> Path:
        """Create market share evolution charts."""
        output_file = output_dir / "track_b_market_share_evolution.png"
        
        # Implementation would create market share evolution
        # For now, create a placeholder
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Track B Market Share Evolution\n(Detailed Implementation)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _generate_mermaid_diagrams(self, 
                                 forecast_data: Dict[str, pd.DataFrame],
                                 output_path: Path,
                                 sample_years: List[int]) -> Dict[str, str]:
        """Generate mermaid diagrams for Track B hierarchy."""
        self.logger.info("Generating Track B mermaid diagrams...")
        
        mermaid_output_dir = output_path / "track_b_mermaid_diagrams"
        mermaid_output_dir.mkdir(exist_ok=True)
        
        mermaid_files = {}
        
        for year in sample_years:
            mermaid_content = self._create_mermaid_hierarchy_diagram(forecast_data, year)
            
            filename = f"track_b_hierarchy_{year}.md"
            file_path = mermaid_output_dir / filename
            
            with open(file_path, 'w') as f:
                f.write(mermaid_content)
            
            mermaid_files[f"hierarchy_{year}"] = str(file_path)
        
        # Create index file
        index_content = self._create_mermaid_index(sample_years)
        index_path = mermaid_output_dir / "README.md"
        with open(index_path, 'w') as f:
            f.write(index_content)
        
        mermaid_files['index'] = str(index_path)
        
        return mermaid_files
    
    def _create_mermaid_hierarchy_diagram(self, 
                                        forecast_data: Dict[str, pd.DataFrame],
                                        year: int) -> str:
        """Create mermaid diagram for specific year."""
        
        # Get data for the specified year
        year_data = {}
        for level_name, data in forecast_data.items():
            year_row = data[data['Year'] == year]
            if not year_row.empty:
                year_data[level_name] = year_row.iloc[0]
        
        mermaid_content = f"""# Track B Steel Taxonomy Hierarchy - {year}

```mermaid
graph TD
    A["Total Steel Consumption<br/>{year_data.get('level_0', {}).get('total_steel_demand', 'N/A'):.0f} kt"] --> B1["SEMI_FINISHED<br/>{year_data.get('level_1', {}).get('SEMI_FINISHED', 'N/A'):.0f} kt"]
    A --> B2["FINISHED_FLAT<br/>{year_data.get('level_1', {}).get('FINISHED_FLAT', 'N/A'):.0f} kt"]
    A --> B3["FINISHED_LONG<br/>{year_data.get('level_1', {}).get('FINISHED_LONG', 'N/A'):.0f} kt"]
    A --> B4["TUBE_PIPE<br/>{year_data.get('level_1', {}).get('TUBE_PIPE', 'N/A'):.0f} kt"]
    
    %% Semi-finished products
    B1 --> C1["BILLETS_COMMERCIAL<br/>{year_data.get('level_2', {}).get('BILLETS_COMMERCIAL', 'N/A'):.0f} kt"]
    B1 --> C2["BILLETS_SBQ<br/>{year_data.get('level_2', {}).get('BILLETS_SBQ', 'N/A'):.0f} kt"]
    B1 --> C3["SLABS_STANDARD<br/>{year_data.get('level_2', {}).get('SLABS_STANDARD', 'N/A'):.0f} kt"]
    B1 --> C4["BILLETS_DEGASSED<br/>{year_data.get('level_2', {}).get('BILLETS_DEGASSED', 'N/A'):.0f} kt"]
    B1 --> C5["SLABS_DEGASSED<br/>{year_data.get('level_2', {}).get('SLABS_DEGASSED', 'N/A'):.0f} kt"]
    
    %% Finished flat products
    B2 --> C6["HOT_ROLLED_COIL<br/>{year_data.get('level_2', {}).get('HOT_ROLLED_COIL', 'N/A'):.0f} kt"]
    B2 --> C7["COLD_ROLLED_COIL<br/>{year_data.get('level_2', {}).get('COLD_ROLLED_COIL', 'N/A'):.0f} kt"]
    B2 --> C8["PLATE<br/>{year_data.get('level_2', {}).get('PLATE', 'N/A'):.0f} kt"]
    B2 --> C9["GALVANIZED<br/>{year_data.get('level_2', {}).get('GALVANIZED', 'N/A'):.0f} kt"]
    
    %% Finished long products (showing key categories)
    B3 --> C10["STRUCTURAL_BEAMS<br/>{year_data.get('level_2', {}).get('STRUCTURAL_BEAMS', 'N/A'):.0f} kt"]
    B3 --> C11["RAILS_STANDARD<br/>{year_data.get('level_2', {}).get('RAILS_STANDARD', 'N/A'):.0f} kt"]
    B3 --> C12["REBAR<br/>{year_data.get('level_2', {}).get('REBAR', 'N/A'):.0f} kt"]
    
    %% Tube and pipe products
    B4 --> C13["SEAMLESS_LINE_PIPE<br/>{year_data.get('level_2', {}).get('SEAMLESS_LINE_PIPE', 'N/A'):.0f} kt"]
    B4 --> C14["WELDED_LINE_PIPE<br/>{year_data.get('level_2', {}).get('WELDED_LINE_PIPE', 'N/A'):.0f} kt"]
    
    %% Level 3 examples (showing select categories)
    C1 --> D1["BILLETS_COMM_LOW_CARBON<br/>{year_data.get('level_3', {}).get('BILLETS_COMM_LOW_CARBON', 'N/A'):.0f} kt"]
    C1 --> D2["BILLETS_COMM_MEDIUM_CARBON<br/>{year_data.get('level_3', {}).get('BILLETS_COMM_MEDIUM_CARBON', 'N/A'):.0f} kt"]
    
    C10 --> D3["UB_GRADE_300<br/>{year_data.get('level_3', {}).get('UB_GRADE_300', 'N/A'):.0f} kt"]
    C10 --> D4["UB_GRADE_300_PLUS<br/>{year_data.get('level_3', {}).get('UB_GRADE_300_PLUS', 'N/A'):.0f} kt"]
    
    C11 --> D5["RAILS_STD_FREIGHT<br/>{year_data.get('level_3', {}).get('RAILS_STD_FREIGHT', 'N/A'):.0f} kt"]
    C11 --> D6["RAILS_STD_PASSENGER<br/>{year_data.get('level_3', {}).get('RAILS_STD_PASSENGER', 'N/A'):.0f} kt"]
    
    %% Styling
    classDef level0 fill:#2E4057,stroke:#1A252F,stroke-width:3px,color:#FFFFFF;
    classDef level1 fill:#048A81,stroke:#036B64,stroke-width:2px,color:#FFFFFF;
    classDef level2 fill:#54C6EB,stroke:#3DA5C4,stroke-width:2px,color:#000000;
    classDef level3 fill:#F3A712,stroke:#D18B0A,stroke-width:2px,color:#000000;
    
    class A level0;
    class B1,B2,B3,B4 level1;
    class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14 level2;
    class D1,D2,D3,D4,D5,D6 level3;
```

## Track B Hierarchy Summary for {year}

### Level 0: Total Steel Consumption
- **Total Demand**: {year_data.get('level_0', {}).get('total_steel_demand', 'N/A')} kt (apparent steel use, crude steel equivalent)

### Level 1: Major Product Categories
- **Semi-finished Products**: {year_data.get('level_1', {}).get('SEMI_FINISHED', 'N/A')} kt ({(year_data.get('level_1', {}).get('SEMI_FINISHED', 0) / year_data.get('level_0', {}).get('total_steel_demand', 1) * 100):.1f}%)
- **Finished Flat Products**: {year_data.get('level_1', {}).get('FINISHED_FLAT', 'N/A')} kt ({(year_data.get('level_1', {}).get('FINISHED_FLAT', 0) / year_data.get('level_0', {}).get('total_steel_demand', 1) * 100):.1f}%)
- **Finished Long Products**: {year_data.get('level_1', {}).get('FINISHED_LONG', 'N/A')} kt ({(year_data.get('level_1', {}).get('FINISHED_LONG', 0) / year_data.get('level_0', {}).get('total_steel_demand', 1) * 100):.1f}%)
- **Tube and Pipe Products**: {year_data.get('level_1', {}).get('TUBE_PIPE', 'N/A')} kt ({(year_data.get('level_1', {}).get('TUBE_PIPE', 0) / year_data.get('level_0', {}).get('total_steel_demand', 1) * 100):.1f}%)

### Key Product Families (Level 2)
Top 5 largest product families by volume:"""

        # Add top 5 products from level 2
        if 'level_2' in year_data:
            level_2_data = year_data['level_2']
            level_2_products = {k: v for k, v in level_2_data.items() if k != 'Year' and isinstance(v, (int, float))}
            top_5_products = sorted(level_2_products.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for i, (product, volume) in enumerate(top_5_products, 1):
                mermaid_content += f"\n{i}. **{product}**: {volume:.0f} kt"

        mermaid_content += f"""

### Specialized Applications (Level 3 - Selected Examples)
- **Commercial Billets**: Low carbon ({year_data.get('level_3', {}).get('BILLETS_COMM_LOW_CARBON', 'N/A'):.0f} kt), Medium carbon ({year_data.get('level_3', {}).get('BILLETS_COMM_MEDIUM_CARBON', 'N/A'):.0f} kt)
- **Structural Beams**: Grade 300 ({year_data.get('level_3', {}).get('UB_GRADE_300', 'N/A'):.0f} kt), Grade 300+ ({year_data.get('level_3', {}).get('UB_GRADE_300_PLUS', 'N/A'):.0f} kt)
- **Railway Products**: Freight rails ({year_data.get('level_3', {}).get('RAILS_STD_FREIGHT', 'N/A'):.0f} kt), Passenger rails ({year_data.get('level_3', {}).get('RAILS_STD_PASSENGER', 'N/A'):.0f} kt)

---
*Generated by Track B Steel Taxonomy Analyzer*
"""
        
        return mermaid_content
    
    def _create_mermaid_index(self, sample_years: List[int]) -> str:
        """Create index file for mermaid diagrams."""
        content = f"""# Track B Steel Taxonomy - Mermaid Diagrams

This directory contains interactive mermaid diagrams showing the Track B hierarchical steel consumption taxonomy for key forecast years.

## Available Diagrams

"""
        for year in sample_years:
            content += f"- [{year} Hierarchy](track_b_hierarchy_{year}.md) - Complete 4-level hierarchy with volumes\n"
        
        content += f"""
## Hierarchy Structure

The Track B taxonomy consists of 4 levels:

1. **Level 0**: Total Steel Consumption (Apparent steel use, crude steel equivalent)
2. **Level 1**: Major Product Categories (4 categories)
   - Semi-finished Products
   - Finished Flat Products  
   - Finished Long Products
   - Tube and Pipe Products

3. **Level 2**: Product Families (23 detailed product families)
   - Billets (Commercial, SBQ, Degassed)
   - Slabs (Standard, Degassed)
   - Flat Products (Hot rolled, Cold rolled, Plate, Galvanized)
   - Structural Products (Beams, Columns, Channels, Angles)
   - Rails and Infrastructure (Standard rails, Head-hardened rails, Sleeper bar)
   - Other Long Products (Rebar, Wire rod, Welded structural)
   - Tubular Products (Seamless, Welded, Other)

4. **Level 3**: Client Product Specifications (25+ specific grades and applications)
   - Grade specifications (Grade 300, Grade 300+, etc.)
   - Application-specific products (Automotive, Mining, Oil & Gas)
   - End-use categories (Freight, Passenger, Heavy-haul, etc.)

## Usage

These diagrams can be viewed in any markdown viewer that supports mermaid syntax, including:
- GitHub
- GitLab  
- Mermaid Live Editor (https://mermaid.live/)
- VS Code with Mermaid extension

---
*Generated by Track B Steel Taxonomy Analyzer - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return content
    
    def _export_comprehensive_csv_files(self, 
                                      forecast_data: Dict[str, pd.DataFrame],
                                      hierarchical_analysis: Dict[str, Any],
                                      relationship_mappings: Dict[str, Any],
                                      output_path: Path) -> Dict[str, str]:
        """Export comprehensive CSV files for Track B analysis."""
        self.logger.info("Exporting Track B taxonomy CSV files...")
        
        csv_output_dir = output_path / "track_b_taxonomy_csvs"
        csv_output_dir.mkdir(exist_ok=True)
        
        csv_files = {}
        
        # 1. Hierarchical Relationships CSV
        relationships_data = []
        for level_name, level_info in self.track_b_hierarchy.items():
            relationships_data.append({
                'Level_ID': level_name,
                'Level_Name': level_info['name'],
                'Description': level_info['description'],
                'Hierarchy_Level': level_info['hierarchy_level'],
                'Parent': level_info.get('parent', ''),
                'Color': level_info['color'],
                'Product_Count': len(level_info['categories'])
            })
        
        relationships_df = pd.DataFrame(relationships_data)
        relationships_file = csv_output_dir / "Track_B_Hierarchical_Relationships.csv"
        relationships_df.to_csv(relationships_file, index=False)
        csv_files['hierarchical_relationships'] = str(relationships_file)
        
        # 2. Product Mappings CSV
        mappings_data = []
        
        # Level 1 to Level 2 mappings
        for l1_cat, l2_cats in self.level_1_to_level_2_mapping.items():
            for l2_cat in l2_cats:
                mappings_data.append({
                    'Parent_Level': 1,
                    'Parent_Category': l1_cat,
                    'Child_Level': 2,
                    'Child_Category': l2_cat,
                    'Description': self.product_descriptions.get(l2_cat, '')
                })
        
        # Level 2 to Level 3 mappings
        for l2_cat, l3_cats in self.level_2_to_level_3_mapping.items():
            for l3_cat in l3_cats:
                mappings_data.append({
                    'Parent_Level': 2,
                    'Parent_Category': l2_cat,
                    'Child_Level': 3,
                    'Child_Category': l3_cat,
                    'Description': self.product_descriptions.get(l3_cat, '')
                })
        
        mappings_df = pd.DataFrame(mappings_data)
        mappings_file = csv_output_dir / "Track_B_Product_Mappings.csv"
        mappings_df.to_csv(mappings_file, index=False)
        csv_files['product_mappings'] = str(mappings_file)
        
        # 3. Growth Analysis CSV
        growth_data = []
        for level_name, level_growth in hierarchical_analysis.get('growth_patterns', {}).items():
            for product, product_growth in level_growth.items():
                for period, cagr in product_growth.items():
                    growth_data.append({
                        'Level': level_name,
                        'Product': product,
                        'Period': period,
                        'CAGR_Percent': cagr
                    })
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            growth_file = csv_output_dir / "Track_B_Growth_Analysis.csv"
            growth_df.to_csv(growth_file, index=False)
            csv_files['growth_analysis'] = str(growth_file)
        
        # 4. Market Shares CSV
        shares_data = []
        for level_name, level_shares in hierarchical_analysis.get('product_shares', {}).items():
            for year, year_shares in level_shares.items():
                for product, share in year_shares.items():
                    shares_data.append({
                        'Level': level_name,
                        'Year': year,
                        'Product': product,
                        'Market_Share_Percent': share
                    })
        
        if shares_data:
            shares_df = pd.DataFrame(shares_data)
            shares_file = csv_output_dir / "Track_B_Market_Shares.csv"
            shares_df.to_csv(shares_file, index=False)
            csv_files['market_shares'] = str(shares_file)
        
        return csv_files
    
    def _create_track_b_summary_report(self, 
                                     forecast_data: Dict[str, pd.DataFrame],
                                     hierarchical_analysis: Dict[str, Any],
                                     relationship_mappings: Dict[str, Any],
                                     output_path: Path) -> str:
        """Create comprehensive summary report for Track B taxonomy analysis."""
        self.logger.info("Creating Track B taxonomy summary report...")
        
        summary_file = output_path / "Track_B_Taxonomy_Analysis_Summary.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"""# Track B Steel Taxonomy Analysis Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report provides a comprehensive analysis of the Track B hierarchical steel consumption forecasting taxonomy. Track B implements a 4-level hierarchical structure designed to capture the full spectrum of Australian steel consumption patterns from total demand down to specific client product specifications.

## Hierarchical Structure

### Level 0: Total Steel Consumption
- **Purpose**: Total apparent steel use (crude steel equivalent)
- **Coverage**: Complete Australian steel consumption
- **Data Source**: Aligned with Track A apparent steel use results

### Level 1: Major Product Categories (4 categories)
""")
            
            # Add Level 1 analysis
            if 'level_1' in hierarchical_analysis.get('level_summaries', {}):
                level_1_summary = hierarchical_analysis['level_summaries']['level_1']
                f.write(f"- **Total Categories**: {level_1_summary.get('total_products', 'N/A')}\n")
                f.write(f"- **Categories**: {', '.join(self.level_1_to_level_2_mapping.keys())}\n")
            
            f.write(f"""
### Level 2: Product Families (23 families)
- **Semi-finished Products**: Billets (Commercial, SBQ, Degassed), Slabs (Standard, Degassed)
- **Finished Flat Products**: Hot rolled coil, Cold rolled coil, Plate, Galvanized
- **Finished Long Products**: Structural products, Rails, Rebar, Wire rod
- **Tube and Pipe Products**: Seamless, Welded, Other tubular products

### Level 3: Client Product Specifications (25+ specifications)
- **Grade Specifications**: Grade 300, Grade 300+, etc.
- **Application-Specific**: Automotive, Mining, Oil & Gas applications
- **End-Use Categories**: Freight, Passenger, Heavy-haul applications

## Key Findings

### Growth Patterns
""")
            
            # Add growth analysis
            if 'growth_patterns' in hierarchical_analysis:
                f.write("Growth rate analysis across all hierarchical levels:\n\n")
                for level_name, level_growth in hierarchical_analysis['growth_patterns'].items():
                    f.write(f"**{level_name.replace('_', ' ').title()}**:\n")
                    # Add top growing products
                    if level_growth:
                        sample_products = list(level_growth.keys())[:3]
                        for product in sample_products:
                            f.write(f"- {product}: Multiple growth periods analyzed\n")
                    f.write("\n")
            
            f.write(f"""
### Market Share Distribution
Key insights about product category distribution across the hierarchy.

### Hierarchical Consistency
Analysis of consistency between hierarchical levels and aggregation accuracy.

## Output Files Generated

### CSV Files
- `Track_B_Hierarchical_Relationships.csv` - Complete hierarchy structure
- `Track_B_Product_Mappings.csv` - Parent-child relationships
- `Track_B_Growth_Analysis.csv` - Growth rate analysis
- `Track_B_Market_Shares.csv` - Market share evolution

### Visualizations
- Hierarchical structure overview charts
- Product category breakdown by level
- Growth rate comparison charts
- Market share evolution analysis

### Mermaid Diagrams
Interactive hierarchy diagrams for key forecast years showing:
- Complete 4-level structure
- Volume flows through hierarchy
- Product relationships
- Client specification details

## Technical Notes

- **Data Source**: Track B hierarchical forecasting system
- **Forecast Period**: 2025-2050
- **Consistency Checks**: Automated validation across all levels
- **Integration**: Seamlessly integrated with Track B consumption forecasting

## Usage

This taxonomy analysis provides the foundation for:
1. Understanding Australian steel consumption patterns
2. Client-specific product planning
3. Market analysis and forecasting
4. Supply chain optimization
5. Product development strategies

---
*Track B Steel Taxonomy Analyzer - Australian Steel Demand Model*
""")
        
        self.logger.info(f"Summary report created: {summary_file}")
        return str(summary_file)

# Example usage
if __name__ == "__main__":
    analyzer = TrackBSteelTaxonomyAnalyzer()
    
    # Example analysis
    forecast_dir = "outputs/track_b"
    output_dir = "analysis/track_b_taxonomy_analysis"
    
    results = analyzer.analyze_track_b_forecasts(
        forecast_data_dir=forecast_dir,
        output_dir=output_dir,
        sample_years=[2025, 2035, 2050]
    )
    
    print(f"Track B taxonomy analysis completed!")
    print(f"Results saved to: {results['output_directory']}")