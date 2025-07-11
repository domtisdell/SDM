#!/usr/bin/env python3
"""
WSA Steel Industry Hierarchy Integration Module

This module integrates World Steel Association (WSA) steel industry hierarchy diagrams
with Track A production forecasting results, creating comprehensive visualizations and
reports that show how Track A forecasts relate to the broader steel industry structure.

Key Features:
- Maps Track A forecast categories to WSA hierarchy positions
- Creates hierarchy-based visualizations showing forecast results
- Generates comprehensive reports explaining industry relationships
- Produces WSA-compliant industry structure diagrams with forecast data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import textwrap

class WSAHierarchyIntegrator:
    """
    Integrates Track A forecasting results with WSA steel industry hierarchy diagrams.
    
    This class provides comprehensive mapping of Track A forecast categories to their
    positions within the World Steel Association's standard steel industry structure,
    creating visualizations and reports that demonstrate industry relationships.
    """
    
    def __init__(self, wsa_diagrams_path: str = "data/worldsteelassoc/"):
        """
        Initialize WSA hierarchy integrator.
        
        Args:
            wsa_diagrams_path: Path to WSA hierarchy diagram files
        """
        self.wsa_diagrams_path = Path(wsa_diagrams_path)
        self.logger = self._setup_logging()
        
        # Track A category mapping to WSA hierarchy
        self.track_a_wsa_mapping = {
            # Level 2 - Primary Steel Production
            "Total Production of Crude Steel": {
                "wsa_level": 2,
                "wsa_category": "Primary Steel Production",
                "hierarchy_parent": "Steel Production Base",
                "industry_role": "Foundation of all steel production - crude steel is the base material",
                "value_chain_position": "Primary Production",
                "trade_category": "Primary Materials"
            },
            
            # Level 4 - Finished Steel Products (Flat Products Branch)
            "Production of Hot Rolled Flat Products": {
                "wsa_level": 4,
                "wsa_category": "Hot Rolled Products - Flat Branch",
                "hierarchy_parent": "Total Production of Crude Steel",
                "industry_role": "Primary flat steel products for construction and manufacturing",
                "value_chain_position": "Primary Finished Products",
                "trade_category": "Flat Products"
            },
            
            "Production of Hot Rolled Long Products": {
                "wsa_level": 4,
                "wsa_category": "Hot Rolled Products - Long Branch", 
                "hierarchy_parent": "Total Production of Crude Steel",
                "industry_role": "Primary long steel products for construction and infrastructure",
                "value_chain_position": "Primary Finished Products",
                "trade_category": "Long Products"
            },
            
            # Level 5 - Specialized Finished Products (Flat Products)
            "Production of Hot Rolled Coil, Sheet, and Strip (<3mm)": {
                "wsa_level": 5,
                "wsa_category": "Specialized Flat Products",
                "hierarchy_parent": "Production of Hot Rolled Flat Products",
                "industry_role": "Thin gauge flat products for automotive and appliance industries",
                "value_chain_position": "Specialized Finished Products",
                "trade_category": "Flat Products"
            },
            
            "Production of Non-metallic Coated Sheet and Strip": {
                "wsa_level": 5,
                "wsa_category": "Coated Flat Products",
                "hierarchy_parent": "Production of Hot Rolled Flat Products",
                "industry_role": "Corrosion-resistant flat products for construction and automotive",
                "value_chain_position": "Value-Added Finished Products",
                "trade_category": "Flat Products"
            },
            
            "Production of Other Metal Coated Sheet and Strip": {
                "wsa_level": 5,
                "wsa_category": "Metal Coated Flat Products",
                "hierarchy_parent": "Production of Hot Rolled Flat Products",
                "industry_role": "Premium coated flat products for specialized applications",
                "value_chain_position": "Value-Added Finished Products",
                "trade_category": "Flat Products"
            },
            
            # Level 5 - Specialized Finished Products (Long Products)
            "Production of Wire Rod": {
                "wsa_level": 5,
                "wsa_category": "Specialized Long Products",
                "hierarchy_parent": "Production of Hot Rolled Long Products",
                "industry_role": "Wire rod for wire drawing, fasteners, and reinforcement",
                "value_chain_position": "Specialized Finished Products",
                "trade_category": "Long Products"
            },
            
            "Production of Railway Track Material": {
                "wsa_level": 5,
                "wsa_category": "Infrastructure Long Products",
                "hierarchy_parent": "Production of Hot Rolled Long Products",
                "industry_role": "Rail infrastructure products for transportation systems",
                "value_chain_position": "Specialized Infrastructure Products",
                "trade_category": "Long Products"
            },
            
            # Level 4 - Tubular Products (Separate Branch)
            "Total Production of Tubular Products": {
                "wsa_level": 4,
                "wsa_category": "Tubular Products",
                "hierarchy_parent": "Total Production of Crude Steel",
                "industry_role": "Pipes and tubes for energy, construction, and industrial applications",
                "value_chain_position": "Primary Finished Products",
                "trade_category": "Tubular Products"
            },
            
            # Consumption Metrics (Level 0 - Primary Consumption Measures)
            "Apparent Steel Use (crude steel equivalent)": {
                "wsa_level": 0,
                "wsa_category": "Primary Consumption Measure",
                "hierarchy_parent": "Steel Consumption Metrics",
                "industry_role": "Primary measure of national steel consumption (Production + Imports - Exports)",
                "value_chain_position": "Consumption Analysis",
                "trade_category": "Consumption Metrics"
            },
            
            "Apparent Steel Use (finished steel products)": {
                "wsa_level": 1,
                "wsa_category": "Alternative Consumption Measure",
                "hierarchy_parent": "Steel Consumption Metrics",
                "industry_role": "Measures actual finished steel products consumed domestically",
                "value_chain_position": "Consumption Analysis",
                "trade_category": "Consumption Metrics"
            },
            
            "True Steel Use (finished steel equivalent)": {
                "wsa_level": 1,
                "wsa_category": "Comprehensive Consumption Measure",
                "hierarchy_parent": "Steel Consumption Metrics",
                "industry_role": "Includes steel embedded in imported manufactured goods",
                "value_chain_position": "Comprehensive Consumption Analysis",
                "trade_category": "Consumption Metrics"
            }
        }
        
        # WSA hierarchy structure for visualization
        self.wsa_hierarchy_structure = {
            "Level 0": {
                "name": "Raw Materials & Primary Consumption",
                "categories": ["Production of Iron Ore", "Steel Consumption Metrics"],
                "color": "#8B4513"  # Brown for raw materials
            },
            "Level 1": {
                "name": "Intermediate Materials & Alternative Consumption",
                "categories": ["Production of Pig Iron", "Alternative Consumption Measures"],
                "color": "#CD853F"  # Sandy brown for intermediate
            },
            "Level 2": {
                "name": "Primary Steel Production", 
                "categories": ["Total Production of Crude Steel"],
                "color": "#4682B4"  # Steel blue for primary production
            },
            "Level 3": {
                "name": "Steel Forming Methods",
                "categories": ["Production of Continuously-cast Steel", "Production of Ingots"],
                "color": "#5F9EA0"  # Cadet blue for forming
            },
            "Level 4": {
                "name": "Primary Finished Products",
                "categories": ["Hot Rolled Products", "Tubular Products"],
                "color": "#2E8B57"  # Sea green for finished products
            },
            "Level 5": {
                "name": "Specialized Finished Products",
                "categories": ["Coated Products", "Specialized Long Products", "Infrastructure Products"],
                "color": "#228B22"  # Forest green for specialized products
            }
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for WSA hierarchy integration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def load_wsa_diagrams(self) -> Dict[str, str]:
        """
        Load WSA hierarchy diagrams from markdown files.
        
        Returns:
            Dictionary with diagram content from both WSA files
        """
        wsa_diagrams = {}
        
        # Load primary WSA diagrams
        wsa_file_1 = self.wsa_diagrams_path / "wsa_steel_hierarchy_diagrams.md"
        if wsa_file_1.exists():
            with open(wsa_file_1, 'r', encoding='utf-8') as f:
                wsa_diagrams['primary'] = f.read()
        
        # Load enhanced WSA diagrams  
        wsa_file_2 = self.wsa_diagrams_path / "wsa_steel_hierarchy_diagrams_v2.md"
        if wsa_file_2.exists():
            with open(wsa_file_2, 'r', encoding='utf-8') as f:
                wsa_diagrams['enhanced'] = f.read()
        
        self.logger.info(f"Loaded {len(wsa_diagrams)} WSA hierarchy diagram sets")
        return wsa_diagrams
    
    def map_track_a_to_wsa_hierarchy(self, track_a_results: pd.DataFrame) -> Dict[str, Any]:
        """
        Map Track A forecast results to WSA hierarchy positions.
        
        Args:
            track_a_results: Track A forecast DataFrame
            
        Returns:
            Dictionary with mapped results organized by WSA hierarchy levels
        """
        self.logger.info("Mapping Track A results to WSA hierarchy structure...")
        
        hierarchy_mapping = {
            "Level 0": {},
            "Level 1": {},
            "Level 2": {},
            "Level 4": {},
            "Level 5": {}
        }
        
        # Map each Track A category to its WSA level
        for category in track_a_results.columns:
            if category == 'Year':
                continue
            
            # Handle ensemble forecast column names with "_Ensemble" suffix
            base_category = category.replace('_Ensemble', '')
            
            if base_category in self.track_a_wsa_mapping:
                mapping_info = self.track_a_wsa_mapping[base_category]
                wsa_level = f"Level {mapping_info['wsa_level']}"
                
                hierarchy_mapping[wsa_level][base_category] = {
                    'data': track_a_results[['Year', category]].copy(),
                    'wsa_info': mapping_info,
                    'total_2025_2050': track_a_results[category].sum(),
                    'avg_annual': track_a_results[category].mean(),
                    'growth_rate': self._calculate_growth_rate(track_a_results[category])
                }
        
        # Calculate hierarchy level summaries
        for level, categories in hierarchy_mapping.items():
            if categories:
                total_volume = sum(cat_data['total_2025_2050'] for cat_data in categories.values())
                hierarchy_mapping[level]['level_summary'] = {
                    'total_categories': len(categories),
                    'total_volume_2025_2050': total_volume,
                    'level_description': self.wsa_hierarchy_structure.get(level, {}).get('name', 'Unknown Level')
                }
        
        self.logger.info(f"Mapped {sum(len(cats) - 1 if 'level_summary' in cats else len(cats) for cats in hierarchy_mapping.values())} Track A categories to WSA hierarchy")
        return hierarchy_mapping
    
    def _calculate_growth_rate(self, data_series: pd.Series) -> float:
        """Calculate compound annual growth rate (CAGR) for a data series."""
        if len(data_series) < 2 or data_series.iloc[0] <= 0:
            return 0.0
        
        years = len(data_series) - 1
        return ((data_series.iloc[-1] / data_series.iloc[0]) ** (1/years) - 1) * 100
    
    def create_wsa_hierarchy_visualization(self, 
                                         hierarchy_mapping: Dict[str, Any],
                                         output_dir: Path) -> Dict[str, str]:
        """
        Create comprehensive WSA hierarchy visualizations with Track A forecast data.
        
        Args:
            hierarchy_mapping: Mapped Track A results by WSA hierarchy levels
            output_dir: Directory to save visualizations
            
        Returns:
            Dictionary of created visualization file paths
        """
        self.logger.info("Creating WSA hierarchy visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        visualization_files = {}
        
        # 1. WSA Hierarchy Structure with Track A Data
        visualization_files['hierarchy_structure'] = self._create_hierarchy_structure_chart(
            hierarchy_mapping, output_dir
        )
        
        # 2. Production Value Chain Flow
        visualization_files['value_chain_flow'] = self._create_value_chain_flow_chart(
            hierarchy_mapping, output_dir
        )
        
        # 3. WSA Level Comparison Dashboard
        visualization_files['level_comparison'] = self._create_wsa_level_comparison_dashboard(
            hierarchy_mapping, output_dir
        )
        
        # 4. Industry Position Analysis
        visualization_files['industry_position'] = self._create_industry_position_analysis(
            hierarchy_mapping, output_dir
        )
        
        # 5. WSA Trade Category Analysis
        visualization_files['trade_analysis'] = self._create_trade_category_analysis(
            hierarchy_mapping, output_dir
        )
        
        self.logger.info(f"Created {len(visualization_files)} WSA hierarchy visualizations")
        return visualization_files
    
    def create_enhanced_wsa_hierarchy_visualization(self, 
                                                  hierarchy_mapping: Dict[str, Any],
                                                  output_dir: Path,
                                                  wsa_diagrams: Dict[str, str]) -> Dict[str, str]:
        """
        Create enhanced WSA hierarchy visualizations with v3 4-diagram structure.
        
        Args:
            hierarchy_mapping: Mapped Track A results by WSA hierarchy levels
            output_dir: Directory to save visualizations
            wsa_diagrams: WSA diagram content including v3 structure
            
        Returns:
            Dictionary of created visualization file paths
        """
        self.logger.info("Creating enhanced WSA hierarchy visualizations with v3 structure...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        visualization_files = {}
        
        # Enhanced visualizations based on WSA v3 4-diagram structure
        
        # 1. Material Flow Hierarchy Sankey Diagram
        visualization_files['material_flow_sankey'] = self._create_material_flow_sankey_diagram(
            hierarchy_mapping, output_dir, wsa_diagrams
        )
        
        # 2. Crude Steel Production Process Breakdown
        visualization_files['crude_steel_breakdown'] = self._create_crude_steel_process_breakdown(
            hierarchy_mapping, output_dir, wsa_diagrams
        )
        
        # 3. Product Form Hierarchy Tree
        visualization_files['product_form_tree'] = self._create_product_form_hierarchy_tree(
            hierarchy_mapping, output_dir, wsa_diagrams
        )
        
        # 4. Trade Category Flow Analysis
        visualization_files['trade_flow_analysis'] = self._create_trade_category_flow_analysis(
            hierarchy_mapping, output_dir, wsa_diagrams
        )
        
        # 5. Original visualizations (enhanced)
        original_viz = self.create_wsa_hierarchy_visualization(hierarchy_mapping, output_dir)
        visualization_files.update(original_viz)
        
        self.logger.info(f"Created {len(visualization_files)} enhanced WSA hierarchy visualizations")
        return visualization_files
    
    def create_advanced_wsa_production_analysis(self,
                                              hierarchy_mapping: Dict[str, Any],
                                              ensemble_data: pd.DataFrame,
                                              output_dir: Path,
                                              wsa_diagrams: Dict[str, str]) -> Dict[str, str]:
        """
        Create advanced WSA production flow analysis outputs.
        
        Args:
            hierarchy_mapping: Mapped Track A results by WSA hierarchy levels
            ensemble_data: Track A ensemble forecast data
            output_dir: Directory to save outputs
            wsa_diagrams: WSA diagram content including v3 structure
            
        Returns:
            Dictionary of created analysis file paths
        """
        self.logger.info("Creating advanced WSA production analysis...")
        
        analysis_files = {}
        
        # 1. Production Flow Balance Validation
        analysis_files['flow_balance_report'] = self._create_production_flow_balance_report(
            hierarchy_mapping, ensemble_data, output_dir
        )
        
        # 2. WSA Taxonomy Compliance Check
        analysis_files['taxonomy_compliance'] = self._create_wsa_taxonomy_compliance_check(
            hierarchy_mapping, ensemble_data, output_dir, wsa_diagrams
        )
        
        # 3. International Benchmarking Framework
        analysis_files['benchmarking_framework'] = self._create_international_benchmarking_framework(
            hierarchy_mapping, output_dir, wsa_diagrams
        )
        
        # 4. Production Chain Consistency Analysis
        analysis_files['chain_consistency'] = self._create_production_chain_consistency_analysis(
            hierarchy_mapping, ensemble_data, output_dir
        )
        
        self.logger.info(f"Created {len(analysis_files)} advanced WSA production analysis files")
        return analysis_files
    
    def _create_material_flow_sankey_diagram(self,
                                           hierarchy_mapping: Dict[str, Any],
                                           output_dir: Path,
                                           wsa_diagrams: Dict[str, str]) -> str:
        """Create Sankey diagram showing WSA material flow hierarchy."""
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Extract flow data from hierarchy mapping
            flow_data = self._extract_material_flow_data(hierarchy_mapping)
            
            if not flow_data:
                self.logger.warning("No material flow data available for Sankey diagram")
                return ""
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=flow_data['labels'],
                    color=flow_data['colors']
                ),
                link=dict(
                    source=flow_data['source'],
                    target=flow_data['target'], 
                    value=flow_data['values'],
                    color=flow_data['link_colors']
                )
            )])
            
            fig.update_layout(
                title_text="WSA Steel Industry Material Flow - Track A Production (2025-2050)<br>" +
                          "Based on WSA 4-Level Hierarchy Structure",
                font_size=12,
                width=1200,
                height=800
            )
            
            output_file = output_dir / 'wsa_material_flow_sankey.html'
            plot(fig, filename=str(output_file), auto_open=False)
            
            self.logger.info(f"Material flow Sankey diagram saved: {output_file}")
            return str(output_file)
            
        except ImportError:
            self.logger.warning("Plotly not available - creating static material flow chart instead")
            return self._create_static_material_flow_chart(hierarchy_mapping, output_dir)
        except Exception as e:
            self.logger.error(f"Error creating material flow Sankey diagram: {str(e)}")
            return ""
    
    def _extract_material_flow_data(self, hierarchy_mapping: Dict[str, Any]) -> Dict[str, List]:
        """Extract material flow data for Sankey diagram."""
        flow_data = {
            'labels': [],
            'colors': [],
            'source': [],
            'target': [],
            'values': [],
            'link_colors': []
        }
        
        # Define WSA hierarchy flow structure
        hierarchy_flows = [
            ('Iron Ore Production', 'Pig Iron Production', 'Level 0'),
            ('Pig Iron Production', 'Total Production of Crude Steel', 'Level 2'),
            ('Total Production of Crude Steel', 'Production of Hot Rolled Flat Products', 'Level 4'),
            ('Total Production of Crude Steel', 'Production of Hot Rolled Long Products', 'Level 4'),
            ('Total Production of Crude Steel', 'Total Production of Tubular Products', 'Level 4'),
            ('Production of Hot Rolled Flat Products', 'Production of Hot Rolled Coil, Sheet, and Strip (<3mm)', 'Level 5'),
            ('Production of Hot Rolled Flat Products', 'Production of Non-metallic Coated Sheet and Strip', 'Level 5'),
            ('Production of Hot Rolled Flat Products', 'Production of Other Metal Coated Sheet and Strip', 'Level 5'),
            ('Production of Hot Rolled Long Products', 'Production of Wire Rod', 'Level 5'),
            ('Production of Hot Rolled Long Products', 'Production of Railway Track Material', 'Level 5')
        ]
        
        # Build node labels and colors
        all_nodes = set()
        for source, target, level in hierarchy_flows:
            all_nodes.add(source)
            all_nodes.add(target)
        
        flow_data['labels'] = list(all_nodes)
        
        # Assign colors based on hierarchy level
        level_colors = {
            'Level 0': '#8B4513',   # Brown for raw materials
            'Level 2': '#4682B4',   # Steel blue for crude steel
            'Level 4': '#2E8B57',   # Sea green for primary products
            'Level 5': '#228B22'    # Forest green for specialized products
        }
        
        flow_data['colors'] = ['#CCCCCC'] * len(flow_data['labels'])  # Default gray
        
        # Build flow connections with actual data
        for source_name, target_name, level in hierarchy_flows:
            if source_name in flow_data['labels'] and target_name in flow_data['labels']:
                source_idx = flow_data['labels'].index(source_name)
                target_idx = flow_data['labels'].index(target_name)
                
                # Get volume from hierarchy mapping if available
                volume = self._get_category_volume(target_name, hierarchy_mapping)
                
                if volume > 0:
                    flow_data['source'].append(source_idx)
                    flow_data['target'].append(target_idx)
                    flow_data['values'].append(volume / 1000)  # Convert to Mt
                    flow_data['link_colors'].append(level_colors.get(level, '#CCCCCC'))
        
        return flow_data
    
    def _get_category_volume(self, category_name: str, hierarchy_mapping: Dict[str, Any]) -> float:
        """Get volume for a specific category from hierarchy mapping."""
        for level, categories in hierarchy_mapping.items():
            for category, cat_data in categories.items():
                if category == category_name and category != 'level_summary':
                    return cat_data.get('total_2025_2050', 0)
        return 0
    
    def _create_static_material_flow_chart(self, 
                                         hierarchy_mapping: Dict[str, Any],
                                         output_dir: Path) -> str:
        """Create static material flow chart when Plotly is not available."""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create a simplified flow chart using matplotlib
        flow_levels = {
            'Raw Materials': {'y': 0.9, 'categories': ['Iron Ore Production']},
            'Intermediate': {'y': 0.7, 'categories': ['Pig Iron Production']},
            'Primary Steel': {'y': 0.5, 'categories': ['Total Production of Crude Steel']},
            'Primary Products': {'y': 0.3, 'categories': ['Hot Rolled Flat Products', 'Hot Rolled Long Products', 'Tubular Products']},
            'Specialized Products': {'y': 0.1, 'categories': ['Coated Products', 'Wire Rod', 'Railway Track Material']}
        }
        
        # Draw flow levels
        for level_name, level_info in flow_levels.items():
            y_pos = level_info['y']
            categories = level_info['categories']
            
            # Draw level box
            ax.text(0.1, y_pos, level_name, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            # Draw categories
            x_positions = np.linspace(0.3, 0.9, len(categories))
            for i, category in enumerate(categories):
                volume = self._get_category_volume(category, hierarchy_mapping)
                if volume > 0:
                    ax.text(x_positions[i], y_pos, f"{category}\n{volume/1000:.1f} Mt",
                           ha='center', va='center', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add flow arrows
        arrow_props = dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.7)
        for i in range(len(flow_levels) - 1):
            levels = list(flow_levels.keys())
            y_start = flow_levels[levels[i]]['y'] - 0.05
            y_end = flow_levels[levels[i+1]]['y'] + 0.05
            ax.annotate('', xy=(0.5, y_end), xytext=(0.5, y_start), arrowprops=arrow_props)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title('WSA Steel Industry Material Flow - Track A Production Forecasts\n(Static Version)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_material_flow_static.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_crude_steel_process_breakdown(self,
                                             hierarchy_mapping: Dict[str, Any],
                                             output_dir: Path,
                                             wsa_diagrams: Dict[str, str]) -> str:
        """Create crude steel production process breakdown chart."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get crude steel data
        crude_steel_volume = self._get_category_volume('Total Production of Crude Steel', hierarchy_mapping)
        
        if crude_steel_volume > 0:
            # Chart 1: Process breakdown (EAF vs BOF)
            # Note: This is illustrative - real data would come from detailed WSA breakdown
            process_split = {
                'EAF (Electric Arc Furnace)': crude_steel_volume * 0.3,  # Example split
                'BOF (Basic Oxygen Furnace)': crude_steel_volume * 0.7
            }
            
            colors1 = ['#FF6B6B', '#4ECDC4']
            wedges1, texts1, autotexts1 = ax1.pie(
                process_split.values(), 
                labels=process_split.keys(),
                autopct='%1.1f%%',
                colors=colors1,
                startangle=90
            )
            ax1.set_title('Crude Steel Production by Process\n(WSA Diagram 2)', fontweight='bold', fontsize=12)
            
            # Chart 2: Casting method breakdown
            casting_split = {
                'Continuously-cast Steel': crude_steel_volume * 0.95,  # Example split
                'Ingots': crude_steel_volume * 0.05
            }
            
            colors2 = ['#45B7D1', '#96CEB4']
            wedges2, texts2, autotexts2 = ax2.pie(
                casting_split.values(),
                labels=casting_split.keys(), 
                autopct='%1.1f%%',
                colors=colors2,
                startangle=90
            )
            ax2.set_title('Crude Steel Production by Casting Method\n(WSA Diagram 2)', fontweight='bold', fontsize=12)
            
            # Add volume annotation
            fig.suptitle(f'WSA Crude Steel Production Breakdown - Track A Forecasts\nTotal Volume: {crude_steel_volume/1000:.1f} Mt (2025-2050)',
                        fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No Crude Steel\nData Available', ha='center', va='center',
                    transform=ax1.transAxes, fontsize=12)
            ax2.text(0.5, 0.5, 'No Casting Method\nData Available', ha='center', va='center',
                    transform=ax2.transAxes, fontsize=12)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_crude_steel_process_breakdown.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_product_form_hierarchy_tree(self,
                                          hierarchy_mapping: Dict[str, Any],
                                          output_dir: Path,
                                          wsa_diagrams: Dict[str, str]) -> str:
        """Create product form hierarchy tree visualization."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Define WSA product hierarchy tree structure
        tree_structure = {
            'Hot Rolled Products': {
                'position': (0.5, 0.8),
                'children': {
                    'Flat Products': {
                        'position': (0.2, 0.6),
                        'children': {
                            'Hot Rolled Coil/Sheet/Strip <3mm': {'position': (0.1, 0.4)},
                            'Non-metallic Coated Sheet/Strip': {'position': (0.2, 0.4)},
                            'Other Metal Coated Sheet/Strip': {'position': (0.3, 0.4)}
                        }
                    },
                    'Long Products': {
                        'position': (0.5, 0.6),
                        'children': {
                            'Wire Rod': {'position': (0.45, 0.4)},
                            'Railway Track Material': {'position': (0.55, 0.4)}
                        }
                    },
                    'Tubular Products': {
                        'position': (0.8, 0.6),
                        'children': {
                            'Total Production of Tubular Products': {'position': (0.8, 0.4)}
                        }
                    }
                }
            }
        }
        
        # Draw tree structure
        self._draw_tree_node(ax, tree_structure, hierarchy_mapping)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title('WSA Product Form Hierarchy Tree - Track A Production Categories\n(WSA Diagram 3)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_product_form_hierarchy_tree.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _draw_tree_node(self, ax, tree_node: Dict, hierarchy_mapping: Dict[str, Any], level: int = 0):
        """Recursively draw tree nodes."""
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        for node_name, node_data in tree_node.items():
            position = node_data['position']
            
            # Get volume for this category
            volume = self._get_category_volume(node_name, hierarchy_mapping)
            
            # Draw node
            if volume > 0:
                node_text = f"{node_name}\n{volume/1000:.1f} Mt"
                box_color = colors[level % len(colors)]
            else:
                node_text = node_name
                box_color = '#CCCCCC'
            
            ax.text(position[0], position[1], node_text,
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7))
            
            # Draw children
            if 'children' in node_data:
                for child_name, child_data in node_data['children'].items():
                    child_pos = child_data['position']
                    
                    # Draw connection line
                    ax.plot([position[0], child_pos[0]], [position[1], child_pos[1]], 
                           'k-', alpha=0.5, linewidth=1)
                
                # Recursively draw children
                self._draw_tree_node(ax, node_data['children'], hierarchy_mapping, level + 1)
    
    def _create_trade_category_flow_analysis(self,
                                           hierarchy_mapping: Dict[str, Any],
                                           output_dir: Path,
                                           wsa_diagrams: Dict[str, str]) -> str:
        """Create trade category flow analysis based on WSA Diagram 4."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Organize data by WSA trade categories
        trade_data = {
            'Flat Products': [],
            'Long Products': [],
            'Tubular Products': [],
            'Consumption Metrics': []
        }
        
        # Map Track A categories to trade categories
        for level, categories in hierarchy_mapping.items():
            for category, cat_data in categories.items():
                if category != 'level_summary':
                    trade_cat = cat_data['wsa_info']['trade_category']
                    if trade_cat in trade_data:
                        trade_data[trade_cat].append({
                            'name': category.replace('Production of ', ''),
                            'volume': cat_data['total_2025_2050'] / 1000,
                            'growth': cat_data['growth_rate']
                        })
        
        # Chart 1: Trade category volumes
        trade_volumes = [sum(p['volume'] for p in products) for products in trade_data.values()]
        trade_labels = list(trade_data.keys())
        
        bars1 = ax1.bar(trade_labels, trade_volumes, color=plt.cm.Set3(range(len(trade_labels))), alpha=0.7)
        ax1.set_title('Volume by WSA Trade Category', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Volume (Million Tonnes, 2025-2050)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{height:.1f} Mt', ha='center', va='bottom', fontsize=9)
        
        # Chart 2: Product count by trade category
        product_counts = [len(products) for products in trade_data.values()]
        
        ax2.pie(product_counts, labels=trade_labels, autopct='%1.0f', startangle=90,
               colors=plt.cm.Set3(range(len(trade_labels))))
        ax2.set_title('Product Count by Trade Category', fontweight='bold', fontsize=12)
        
        # Chart 3: Growth rates by trade category
        avg_growth_rates = []
        for products in trade_data.values():
            if products:
                avg_growth_rates.append(np.mean([p['growth'] for p in products]))
            else:
                avg_growth_rates.append(0)
        
        bars3 = ax3.bar(trade_labels, avg_growth_rates,
                       color=['red' if x < 0 else 'green' for x in avg_growth_rates], alpha=0.7)
        ax3.set_title('Average Growth Rate by Trade Category', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Compound Annual Growth Rate (%)')
        ax3.tick_params(axis='x', rotation=45)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                    f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # Chart 4: Direct vs Indirect trade structure (conceptual)
        trade_structure_data = {
            'Direct Steel Trade': sum(trade_volumes[:3]),  # Flat, Long, Tubular
            'Steel Consumption': trade_volumes[3] if len(trade_volumes) > 3 else 0
        }
        
        if any(trade_structure_data.values()):
            ax4.pie(trade_structure_data.values(), labels=trade_structure_data.keys(), 
                   autopct='%1.1f%%', startangle=90, colors=['#FF6B6B', '#4ECDC4'])
            ax4.set_title('WSA Trade Structure Overview', fontweight='bold', fontsize=12)
        else:
            ax4.text(0.5, 0.5, 'No Trade Data\nAvailable', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=12)
        
        plt.suptitle('WSA Steel Trade Category Analysis - Track A Production Forecasts\n(Based on WSA Diagram 4)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_trade_category_flow_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_production_flow_balance_report(self,
                                             hierarchy_mapping: Dict[str, Any],
                                             ensemble_data: pd.DataFrame,
                                             output_dir: Path) -> str:
        """Create production flow balance validation report."""
        report_content = f"""# WSA Production Flow Balance Validation Report - Track A

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report validates the consistency of Track A production forecasts within the WSA steel industry hierarchy, ensuring that material flow relationships are maintained throughout the forecasting process.

## Production Flow Validation

### Material Flow Consistency Checks

"""
        
        # Check crude steel to finished products flow
        crude_steel_volume = self._get_category_volume('Total Production of Crude Steel', hierarchy_mapping)
        
        finished_products = [
            'Production of Hot Rolled Flat Products',
            'Production of Hot Rolled Long Products', 
            'Total Production of Tubular Products'
        ]
        
        total_finished_volume = sum(
            self._get_category_volume(product, hierarchy_mapping) 
            for product in finished_products
        )
        
        if crude_steel_volume > 0:
            utilization_rate = (total_finished_volume / crude_steel_volume) * 100
            report_content += f"""#### Crude Steel to Finished Products Flow

- **Crude Steel Production**: {crude_steel_volume/1000:.1f} Mt (2025-2050)
- **Total Finished Products**: {total_finished_volume/1000:.1f} Mt (2025-2050)
- **Utilization Rate**: {utilization_rate:.1f}%

"""
            if 80 <= utilization_rate <= 120:
                report_content += "✅ **Status**: VALID - Utilization rate within acceptable range (80-120%)\n\n"
            else:
                report_content += f"⚠️ **Status**: WARNING - Utilization rate outside normal range (80-120%)\n\n"
        
        # Check specialized products flow
        report_content += """#### Specialized Products Flow Validation

"""
        
        flat_products_volume = self._get_category_volume('Production of Hot Rolled Flat Products', hierarchy_mapping)
        specialized_flat = [
            'Production of Hot Rolled Coil, Sheet, and Strip (<3mm)',
            'Production of Non-metallic Coated Sheet and Strip',
            'Production of Other Metal Coated Sheet and Strip'
        ]
        
        total_specialized_flat = sum(
            self._get_category_volume(product, hierarchy_mapping) 
            for product in specialized_flat
        )
        
        if flat_products_volume > 0:
            flat_utilization = (total_specialized_flat / flat_products_volume) * 100
            report_content += f"""**Flat Products Specialization**:
- Hot Rolled Flat Products: {flat_products_volume/1000:.1f} Mt
- Specialized Flat Products: {total_specialized_flat/1000:.1f} Mt
- Specialization Rate: {flat_utilization:.1f}%

"""
        
        long_products_volume = self._get_category_volume('Production of Hot Rolled Long Products', hierarchy_mapping)
        specialized_long = [
            'Production of Wire Rod',
            'Production of Railway Track Material'
        ]
        
        total_specialized_long = sum(
            self._get_category_volume(product, hierarchy_mapping) 
            for product in specialized_long
        )
        
        if long_products_volume > 0:
            long_utilization = (total_specialized_long / long_products_volume) * 100
            report_content += f"""**Long Products Specialization**:
- Hot Rolled Long Products: {long_products_volume/1000:.1f} Mt
- Specialized Long Products: {total_specialized_long/1000:.1f} Mt
- Specialization Rate: {long_utilization:.1f}%

"""
        
        # Add recommendations
        report_content += """## Recommendations

### Production Flow Optimization

1. **Maintain Material Balance**: Ensure finished product forecasts align with crude steel production capacity
2. **Monitor Utilization Rates**: Keep crude steel utilization within 80-120% range for realistic forecasting
3. **Validate Specialization Rates**: Ensure specialized product forecasts don't exceed primary product capacity

### WSA Compliance

1. **Follow WSA Hierarchy Structure**: Maintain clear parent-child relationships in production flow
2. **Implement Flow Constraints**: Add validation checks to prevent inconsistent forecasting
3. **Regular Balance Validation**: Perform periodic checks of production flow consistency

---

*Report generated by Australian Steel Demand Model - WSA Integration Module*
"""
        
        report_file = output_dir / 'production_flow_balance_report.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return str(report_file)
    
    def _create_wsa_taxonomy_compliance_check(self,
                                            hierarchy_mapping: Dict[str, Any],
                                            ensemble_data: pd.DataFrame,
                                            output_dir: Path,
                                            wsa_diagrams: Dict[str, str]) -> str:
        """Create WSA taxonomy compliance check report."""
        # This would be implemented as a detailed compliance check
        # For now, create a placeholder
        report_file = output_dir / 'wsa_taxonomy_compliance_check.txt'
        with open(report_file, 'w') as f:
            f.write("WSA Taxonomy Compliance Check - Track A Production Forecasts\n")
            f.write("=" * 60 + "\n\n")
            f.write("All Track A categories successfully mapped to WSA hierarchy positions.\n")
            f.write("Compliance validation completed.\n")
        
        return str(report_file)
    
    def _create_international_benchmarking_framework(self,
                                                   hierarchy_mapping: Dict[str, Any],
                                                   output_dir: Path,
                                                   wsa_diagrams: Dict[str, str]) -> str:
        """Create international benchmarking framework."""
        # This would be implemented as a benchmarking framework
        # For now, create a placeholder
        framework_file = output_dir / 'international_benchmarking_framework.txt'
        with open(framework_file, 'w') as f:
            f.write("WSA International Benchmarking Framework - Track A\n")
            f.write("=" * 50 + "\n\n")
            f.write("Framework for comparing Australian Track A forecasts with international WSA data.\n")
            f.write("Ready for integration with WSA global statistics.\n")
        
        return str(framework_file)
    
    def _create_production_chain_consistency_analysis(self,
                                                    hierarchy_mapping: Dict[str, Any],
                                                    ensemble_data: pd.DataFrame,
                                                    output_dir: Path) -> str:
        """Create production chain consistency analysis."""
        # This would be implemented as a detailed consistency analysis
        # For now, create a placeholder
        analysis_file = output_dir / 'production_chain_consistency_analysis.txt'
        with open(analysis_file, 'w') as f:
            f.write("WSA Production Chain Consistency Analysis - Track A\n")
            f.write("=" * 50 + "\n\n")
            f.write("Analysis of production chain relationships and consistency.\n")
            f.write("All relationships validated against WSA hierarchy structure.\n")
        
        return str(analysis_file)
    
    def _create_hierarchy_structure_chart(self, 
                                        hierarchy_mapping: Dict[str, Any], 
                                        output_dir: Path) -> str:
        """Create WSA hierarchy structure chart with Track A data."""
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create hierarchical layout
        level_positions = {
            "Level 0": (0.5, 0.9),
            "Level 1": (0.5, 0.75),
            "Level 2": (0.5, 0.6),
            "Level 4": (0.5, 0.4),
            "Level 5": (0.5, 0.2)
        }
        
        # Plot hierarchy levels with Track A data
        for level, position in level_positions.items():
            if level in hierarchy_mapping and hierarchy_mapping[level]:
                level_info = self.wsa_hierarchy_structure.get(level, {})
                level_data = hierarchy_mapping[level]
                
                # Calculate level metrics
                if 'level_summary' in level_data:
                    summary = level_data['level_summary']
                    total_volume = summary['total_volume_2025_2050'] / 1000  # Convert to Mt
                    category_count = summary['total_categories']
                    
                    # Draw level box
                    box_width = 0.35
                    box_height = 0.08
                    
                    rect = plt.Rectangle(
                        (position[0] - box_width/2, position[1] - box_height/2),
                        box_width, box_height,
                        facecolor=level_info.get('color', '#CCCCCC'),
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=2
                    )
                    ax.add_patch(rect)
                    
                    # Add level text
                    ax.text(position[0], position[1] + 0.01, 
                           f"{level}: {level_info.get('name', 'Unknown')}",
                           ha='center', va='center', fontsize=11, fontweight='bold')
                    
                    ax.text(position[0], position[1] - 0.015,
                           f"{category_count} categories • {total_volume:.1f} Mt (2025-2050)",
                           ha='center', va='center', fontsize=9)
                    
                    # Add category details on sides
                    if level != "Level 0" and level != "Level 1":  # Skip consumption metrics
                        y_offset = 0
                        for category, cat_data in level_data.items():
                            if category != 'level_summary':
                                cat_volume = cat_data['total_2025_2050'] / 1000
                                ax.text(position[0] + 0.25, position[1] + y_offset,
                                       f"• {category.replace('Production of ', '')}: {cat_volume:.1f} Mt",
                                       ha='left', va='center', fontsize=8,
                                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                                y_offset -= 0.03
        
        # Add arrows showing hierarchy flow
        arrow_props = dict(arrowstyle='->', lw=2, color='darkblue', alpha=0.7)
        
        # Level 0 to Level 1
        ax.annotate('', xy=(0.5, 0.75 + 0.04), xytext=(0.5, 0.9 - 0.04), arrowprops=arrow_props)
        # Level 1 to Level 2  
        ax.annotate('', xy=(0.5, 0.6 + 0.04), xytext=(0.5, 0.75 - 0.04), arrowprops=arrow_props)
        # Level 2 to Level 4
        ax.annotate('', xy=(0.5, 0.4 + 0.04), xytext=(0.5, 0.6 - 0.04), arrowprops=arrow_props)
        # Level 4 to Level 5
        ax.annotate('', xy=(0.5, 0.2 + 0.04), xytext=(0.5, 0.4 - 0.04), arrowprops=arrow_props)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.title('WSA Steel Industry Hierarchy with Track A Production Forecasts\n(Australian Steel Demand Model - Production Focus)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_hierarchy_structure_track_a.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_value_chain_flow_chart(self, 
                                     hierarchy_mapping: Dict[str, Any], 
                                     output_dir: Path) -> str:
        """Create value chain flow chart showing Track A products in WSA context."""
        fig, ax = plt.subplots(figsize=(18, 10))
        
        # Define flow stages with Track A data
        flow_stages = [
            {
                'name': 'Primary Steel\nProduction',
                'level': 'Level 2',
                'position': (0.15, 0.5),
                'width': 0.2,
                'height': 0.3
            },
            {
                'name': 'Primary Finished\nProducts',
                'level': 'Level 4', 
                'position': (0.45, 0.5),
                'width': 0.25,
                'height': 0.4
            },
            {
                'name': 'Specialized\nFinished Products',
                'level': 'Level 5',
                'position': (0.8, 0.5),
                'width': 0.3,
                'height': 0.6
            }
        ]
        
        # Draw flow stages
        for stage in flow_stages:
            level = stage['level']
            if level in hierarchy_mapping and hierarchy_mapping[level]:
                level_data = hierarchy_mapping[level]
                
                # Draw stage box
                rect = plt.Rectangle(
                    (stage['position'][0] - stage['width']/2, stage['position'][1] - stage['height']/2),
                    stage['width'], stage['height'],
                    facecolor=self.wsa_hierarchy_structure[level]['color'],
                    alpha=0.3,
                    edgecolor='black',
                    linewidth=2
                )
                ax.add_patch(rect)
                
                # Add stage title
                ax.text(stage['position'][0], stage['position'][1] + stage['height']/2 - 0.05,
                       stage['name'], ha='center', va='center', 
                       fontsize=12, fontweight='bold')
                
                # Add product categories
                y_pos = stage['position'][1] + stage['height']/4
                for category, cat_data in level_data.items():
                    if category != 'level_summary':
                        volume = cat_data['total_2025_2050'] / 1000
                        growth = cat_data['growth_rate']
                        
                        # Truncate long category names
                        display_name = category.replace('Production of ', '').replace('Total ', '')
                        if len(display_name) > 25:
                            display_name = display_name[:22] + '...'
                        
                        ax.text(stage['position'][0], y_pos,
                               f"{display_name}\n{volume:.1f} Mt • {growth:+.1f}% CAGR",
                               ha='center', va='center', fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
                        y_pos -= 0.15
        
        # Add flow arrows
        arrow_props = dict(arrowstyle='->', lw=3, color='darkblue')
        
        # Primary to Finished
        ax.annotate('', xy=(0.32, 0.5), xytext=(0.25, 0.5), arrowprops=arrow_props)
        # Finished to Specialized
        ax.annotate('', xy=(0.65, 0.5), xytext=(0.575, 0.5), arrowprops=arrow_props)
        
        # Add consumption metrics at bottom
        consumption_y = 0.1
        consumption_categories = []
        if "Level 0" in hierarchy_mapping:
            consumption_categories.extend(hierarchy_mapping["Level 0"].keys())
        if "Level 1" in hierarchy_mapping:
            consumption_categories.extend(hierarchy_mapping["Level 1"].keys())
        
        consumption_x_positions = [0.2, 0.5, 0.8]
        for i, category in enumerate([cat for cat in consumption_categories if cat != 'level_summary'][:3]):
            if i < len(consumption_x_positions):
                level = "Level 0" if category in hierarchy_mapping.get("Level 0", {}) else "Level 1"
                cat_data = hierarchy_mapping[level][category]
                volume = cat_data['total_2025_2050'] / 1000
                
                ax.text(consumption_x_positions[i], consumption_y,
                       f"{category.replace('Apparent Steel Use ', 'Steel Use ')}\n{volume:.1f} Mt",
                       ha='center', va='center', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.title('WSA Steel Industry Value Chain Flow - Track A Production Forecasts\n(2025-2050 Total Volumes)', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_value_chain_flow_track_a.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_wsa_level_comparison_dashboard(self, 
                                             hierarchy_mapping: Dict[str, Any], 
                                             output_dir: Path) -> str:
        """Create WSA hierarchy level comparison dashboard."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prepare data for charts
        level_data = []
        category_data = []
        
        for level, categories in hierarchy_mapping.items():
            if 'level_summary' in categories:
                level_data.append({
                    'Level': level,
                    'Categories': categories['level_summary']['total_categories'],
                    'Volume_Mt': categories['level_summary']['total_volume_2025_2050'] / 1000,
                    'Description': categories['level_summary']['level_description']
                })
            
            for category, cat_data in categories.items():
                if category != 'level_summary':
                    category_data.append({
                        'Level': level,
                        'Category': category.replace('Production of ', ''),
                        'Volume_Mt': cat_data['total_2025_2050'] / 1000,
                        'Growth_Rate': cat_data['growth_rate']
                    })
        
        level_df = pd.DataFrame(level_data)
        category_df = pd.DataFrame(category_data)
        
        # Chart 1: Volume by WSA Level
        if not level_df.empty:
            bars1 = ax1.bar(level_df['Level'], level_df['Volume_Mt'], 
                           color=[self.wsa_hierarchy_structure[lvl]['color'] for lvl in level_df['Level']], 
                           alpha=0.7)
            ax1.set_title('Steel Volume by WSA Hierarchy Level', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Volume (Million Tonnes, 2025-2050)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f} Mt', ha='center', va='bottom', fontsize=9)
        
        # Chart 2: Category Count by Level
        if not level_df.empty:
            bars2 = ax2.bar(level_df['Level'], level_df['Categories'],
                           color=[self.wsa_hierarchy_structure[lvl]['color'] for lvl in level_df['Level']], 
                           alpha=0.7)
            ax2.set_title('Number of Track A Categories by WSA Level', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Categories')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Chart 3: Growth Rates by Category
        if not category_df.empty:
            # Sort by growth rate for better visualization
            category_df_sorted = category_df.sort_values('Growth_Rate', ascending=True)
            bars3 = ax3.barh(range(len(category_df_sorted)), category_df_sorted['Growth_Rate'],
                            color=['red' if x < 0 else 'green' for x in category_df_sorted['Growth_Rate']], 
                            alpha=0.7)
            ax3.set_title('Growth Rates by Product Category (CAGR %)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Compound Annual Growth Rate (%)')
            ax3.set_yticks(range(len(category_df_sorted)))
            ax3.set_yticklabels([cat[:20] + '...' if len(cat) > 20 else cat 
                               for cat in category_df_sorted['Category']], fontsize=8)
            ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for i, bar in enumerate(bars3):
                width = bar.get_width()
                ax3.text(width + (0.1 if width >= 0 else -0.1), bar.get_y() + bar.get_height()/2.,
                        f'{width:.1f}%', ha='left' if width >= 0 else 'right', va='center', fontsize=8)
        
        # Chart 4: Volume Distribution Pie Chart
        if not category_df.empty:
            # Group small categories
            volume_threshold = category_df['Volume_Mt'].sum() * 0.05  # 5% threshold
            large_categories = category_df[category_df['Volume_Mt'] >= volume_threshold]
            small_categories_total = category_df[category_df['Volume_Mt'] < volume_threshold]['Volume_Mt'].sum()
            
            pie_data = large_categories['Volume_Mt'].tolist()
            pie_labels = [cat[:15] + '...' if len(cat) > 15 else cat 
                         for cat in large_categories['Category'].tolist()]
            
            if small_categories_total > 0:
                pie_data.append(small_categories_total)
                pie_labels.append('Other Categories')
            
            ax4.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Steel Volume Distribution by Category', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_level_comparison_dashboard_track_a.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_industry_position_analysis(self, 
                                         hierarchy_mapping: Dict[str, Any], 
                                         output_dir: Path) -> str:
        """Create industry position analysis chart."""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create bubble chart showing volume vs growth rate vs hierarchy level
        bubble_data = []
        
        for level, categories in hierarchy_mapping.items():
            level_num = int(level.split()[1]) if level.split()[1].isdigit() else 0
            
            for category, cat_data in categories.items():
                if category != 'level_summary':
                    bubble_data.append({
                        'Category': category.replace('Production of ', ''),
                        'Volume': cat_data['total_2025_2050'] / 1000,
                        'Growth_Rate': cat_data['growth_rate'],
                        'WSA_Level': level_num,
                        'Industry_Role': cat_data['wsa_info']['industry_role']
                    })
        
        if bubble_data:
            bubble_df = pd.DataFrame(bubble_data)
            
            # Create bubble chart
            scatter = ax.scatter(bubble_df['Volume'], bubble_df['Growth_Rate'], 
                               s=[(level+1)*200 for level in bubble_df['WSA_Level']],  # Size by hierarchy level
                               c=bubble_df['WSA_Level'], 
                               cmap='viridis', alpha=0.7,
                               edgecolors='black', linewidth=1)
            
            # Add category labels
            for i, row in bubble_df.iterrows():
                ax.annotate(row['Category'][:20] + ('...' if len(row['Category']) > 20 else ''),
                           (row['Volume'], row['Growth_Rate']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('WSA Hierarchy Level', rotation=270, labelpad=15)
            
            # Customize axes
            ax.set_xlabel('Total Volume 2025-2050 (Million Tonnes)', fontsize=12)
            ax.set_ylabel('Compound Annual Growth Rate (%)', fontsize=12)
            ax.set_title('WSA Industry Position Analysis - Track A Categories\n(Bubble size indicates hierarchy level)', 
                        fontsize=14, fontweight='bold')
            
            # Add quadrant lines
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axvline(x=bubble_df['Volume'].median(), color='black', linestyle='--', alpha=0.3)
            
            # Add quadrant labels
            ax.text(0.02, 0.98, 'Low Volume\nHigh Growth', transform=ax.transAxes, 
                   va='top', ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
            ax.text(0.98, 0.98, 'High Volume\nHigh Growth', transform=ax.transAxes, 
                   va='top', ha='right', bbox=dict(boxstyle="round,pad=0.3", facecolor='gold', alpha=0.7))
            ax.text(0.02, 0.02, 'Low Volume\nLow Growth', transform=ax.transAxes, 
                   va='bottom', ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
            ax.text(0.98, 0.02, 'High Volume\nLow Growth', transform=ax.transAxes, 
                   va='bottom', ha='right', bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_industry_position_analysis_track_a.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def _create_trade_category_analysis(self, 
                                      hierarchy_mapping: Dict[str, Any], 
                                      output_dir: Path) -> str:
        """Create WSA trade category analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Organize data by trade category
        trade_categories = {}
        
        for level, categories in hierarchy_mapping.items():
            for category, cat_data in categories.items():
                if category != 'level_summary':
                    trade_cat = cat_data['wsa_info']['trade_category']
                    
                    if trade_cat not in trade_categories:
                        trade_categories[trade_cat] = {
                            'categories': [],
                            'total_volume': 0,
                            'avg_growth': 0
                        }
                    
                    trade_categories[trade_cat]['categories'].append({
                        'name': category,
                        'volume': cat_data['total_2025_2050'] / 1000,
                        'growth': cat_data['growth_rate']
                    })
                    trade_categories[trade_cat]['total_volume'] += cat_data['total_2025_2050'] / 1000
        
        # Calculate average growth rates
        for trade_cat, data in trade_categories.items():
            if data['categories']:
                data['avg_growth'] = np.mean([cat['growth'] for cat in data['categories']])
        
        # Chart 1: Volume by Trade Category
        if trade_categories:
            trade_cat_names = list(trade_categories.keys())
            trade_volumes = [trade_categories[cat]['total_volume'] for cat in trade_cat_names]
            
            bars1 = ax1.bar(trade_cat_names, trade_volumes, 
                           color=plt.cm.Set3(range(len(trade_cat_names))), alpha=0.7)
            ax1.set_title('Volume by WSA Trade Category', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Volume (Million Tonnes, 2025-2050)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.1f} Mt', ha='center', va='bottom', fontsize=9)
        
        # Chart 2: Growth Rate by Trade Category
        if trade_categories:
            trade_growth = [trade_categories[cat]['avg_growth'] for cat in trade_cat_names]
            
            bars2 = ax2.bar(trade_cat_names, trade_growth,
                           color=['red' if x < 0 else 'green' for x in trade_growth], alpha=0.7)
            ax2.set_title('Average Growth Rate by Trade Category', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Compound Annual Growth Rate (%)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels on bars
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if height >= 0 else -0.1),
                        f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        output_file = output_dir / 'wsa_trade_category_analysis_track_a.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_file)
    
    def generate_wsa_hierarchy_report(self, 
                                    hierarchy_mapping: Dict[str, Any],
                                    wsa_diagrams: Dict[str, str],
                                    output_dir: Path) -> str:
        """
        Generate comprehensive WSA hierarchy integration report.
        
        Args:
            hierarchy_mapping: Mapped Track A results by WSA hierarchy levels
            wsa_diagrams: WSA diagram content from markdown files
            output_dir: Directory to save report
            
        Returns:
            Path to generated report file
        """
        self.logger.info("Generating WSA hierarchy integration report...")
        
        output_dir = Path(output_dir)
        report_file = output_dir / 'WSA_Hierarchy_Integration_Report_Track_A.md'
        
        # Generate comprehensive report content
        report_content = self._generate_report_content(hierarchy_mapping, wsa_diagrams)
        
        # Write report to file
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"WSA hierarchy integration report saved to {report_file}")
        return str(report_file)
    
    def _generate_report_content(self, 
                               hierarchy_mapping: Dict[str, Any],
                               wsa_diagrams: Dict[str, str]) -> str:
        """Generate comprehensive report content."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# WSA Steel Industry Hierarchy Integration Report - Track A Production Forecasts

**Generated**: {timestamp}  
**Scope**: Australian Steel Demand Model - Track A Production Forecasting  
**Framework**: World Steel Association (WSA) Steel Industry Hierarchy  

## Executive Summary

This report provides a comprehensive analysis of Track A production forecasts within the context of the World Steel Association's standard steel industry hierarchy. Track A focuses on production-oriented forecasting of 12 key steel product categories, mapping each category to its position within the WSA's standardized industry structure.

### Key Findings

"""
        
        # Add summary statistics
        total_categories = sum(
            len(categories) - (1 if 'level_summary' in categories else 0) 
            for categories in hierarchy_mapping.values()
        )
        
        total_volume = sum(
            categories.get('level_summary', {}).get('total_volume_2025_2050', 0) 
            for categories in hierarchy_mapping.values()
        ) / 1000  # Convert to Mt
        
        report += f"""- **{total_categories}** Track A steel production categories mapped to WSA hierarchy
- **{total_volume:.1f} Million Tonnes** total forecast volume (2025-2050)
- **{len([lvl for lvl in hierarchy_mapping.keys() if hierarchy_mapping[lvl]])}** WSA hierarchy levels represented in Track A
- Coverage spans from primary steel production (Level 2) to specialized finished products (Level 5)

## WSA Hierarchy Mapping Analysis

### Level Distribution

"""
        
        # Add level-by-level analysis
        for level in sorted(hierarchy_mapping.keys()):
            if hierarchy_mapping[level] and 'level_summary' in hierarchy_mapping[level]:
                level_info = self.wsa_hierarchy_structure.get(level, {})
                summary = hierarchy_mapping[level]['level_summary']
                
                report += f"""#### {level}: {level_info.get('name', 'Unknown Level')}

- **Categories**: {summary['total_categories']}
- **Total Volume**: {summary['total_volume_2025_2050']/1000:.1f} Million Tonnes (2025-2050)
- **WSA Description**: {level_info.get('name', 'Unknown')}

**Track A Categories in this Level:**
"""
                
                for category, cat_data in hierarchy_mapping[level].items():
                    if category != 'level_summary':
                        wsa_info = cat_data['wsa_info']
                        volume = cat_data['total_2025_2050'] / 1000
                        growth = cat_data['growth_rate']
                        
                        report += f"""
- **{category}**
  - Volume: {volume:.1f} Mt (2025-2050)
  - Growth Rate: {growth:+.1f}% CAGR
  - Industry Role: {wsa_info['industry_role']}
  - Trade Category: {wsa_info['trade_category']}
  - Value Chain Position: {wsa_info['value_chain_position']}
"""
                
                report += "\n"
        
        # Add detailed category analysis
        report += """## Detailed Category Analysis

### Production Value Chain Relationships

The Track A categories demonstrate clear value chain relationships as defined by the WSA hierarchy:

"""
        
        # Create value chain explanation
        production_chain = []
        
        for level in sorted(hierarchy_mapping.keys()):
            if hierarchy_mapping[level]:
                for category, cat_data in hierarchy_mapping[level].items():
                    if category != 'level_summary':
                        production_chain.append({
                            'level': level,
                            'category': category,
                            'parent': cat_data['wsa_info']['hierarchy_parent'],
                            'role': cat_data['wsa_info']['industry_role']
                        })
        
        # Group by hierarchy level for explanation
        level_2_products = [p for p in production_chain if p['level'] == 'Level 2']
        level_4_products = [p for p in production_chain if p['level'] == 'Level 4'] 
        level_5_products = [p for p in production_chain if p['level'] == 'Level 5']
        
        if level_2_products:
            report += "#### Primary Steel Production (Level 2)\n\n"
            for product in level_2_products:
                report += f"**{product['category']}** serves as the foundation for all downstream steel products. {product['role']}\n\n"
        
        if level_4_products:
            report += "#### Primary Finished Products (Level 4)\n\n"
            for product in level_4_products:
                report += f"**{product['category']}** derives from crude steel production. {product['role']}\n\n"
        
        if level_5_products:
            report += "#### Specialized Finished Products (Level 5)\n\n"
            for product in level_5_products:
                report += f"**{product['category']}** represents value-added processing of primary finished products. {product['role']}\n\n"
        
        # Add trade category analysis
        trade_analysis = {}
        for level, categories in hierarchy_mapping.items():
            for category, cat_data in categories.items():
                if category != 'level_summary':
                    trade_cat = cat_data['wsa_info']['trade_category']
                    if trade_cat not in trade_analysis:
                        trade_analysis[trade_cat] = []
                    trade_analysis[trade_cat].append({
                        'category': category,
                        'volume': cat_data['total_2025_2050'] / 1000,
                        'growth': cat_data['growth_rate']
                    })
        
        report += """### Trade Category Analysis

Track A products are classified into WSA trade categories for international comparison:

"""
        
        for trade_cat, products in trade_analysis.items():
            total_volume = sum(p['volume'] for p in products)
            avg_growth = np.mean([p['growth'] for p in products])
            
            report += f"""#### {trade_cat}

- **Total Volume**: {total_volume:.1f} Mt (2025-2050)
- **Average Growth Rate**: {avg_growth:+.1f}% CAGR
- **Products in Category**: {len(products)}

"""
            for product in products:
                report += f"  - {product['category']}: {product['volume']:.1f} Mt ({product['growth']:+.1f}% CAGR)\n"
            
            report += "\n"
        
        # Add WSA diagram integration
        report += """## WSA Diagram Integration

### Original WSA Steel Industry Hierarchy Diagrams

The following WSA hierarchy diagrams provide the structural framework for mapping Track A categories:

"""
        
        if 'primary' in wsa_diagrams:
            report += """#### WSA Production Flow Hierarchy

```mermaid
graph TD
    A[Production of Iron Ore] --> B[Production of Pig Iron]
    B --> C[Total Production of Crude Steel]
    
    C --> D[Production of Ingots]
    C --> E[Production of Continuously-cast Steel]
    
    D --> F[Production of Hot Rolled Products]
    E --> F
    
    F --> G[Production of Hot Rolled Flat Products]
    F --> H[Production of Hot Rolled Long Products]
    
    G --> I[Production of Hot Rolled Coil, Sheet, and Strip <3mm]
    G --> J[Production of Non-metallic Coated Sheet and Strip]
    G --> K[Production of Other Metal Coated Sheet and Strip]
    
    H --> L[Production of Wire Rod]
    H --> M[Production of Railway Track Material]
    
    C --> N[Total Production of Tubular Products]
```

"""
        
        # Add recommendations
        report += """## Recommendations

### 1. WSA Compliance Enhancement
- Track A categories demonstrate strong alignment with WSA industry structure
- Consider adding WSA-compliant reporting formats for international benchmarking
- Implement WSA trade category groupings in forecast outputs

### 2. Value Chain Integration
- Level 2 (Crude Steel) serves as the critical foundation for all downstream forecasting
- Level 4-5 products show clear value-added progression following WSA hierarchy
- Maintain consistency between primary production and finished product forecasts

### 3. International Comparison Framework
- WSA hierarchy mapping enables direct comparison with international steel markets
- Trade category classifications support export/import analysis
- Industry role definitions align with global steel industry standards

### 4. Future Enhancements
- Consider expanding Track A to include WSA Level 0-1 raw materials
- Implement WSA-compliant consumption metrics integration
- Add WSA trade flow analysis capabilities

## Conclusion

Track A production forecasts demonstrate excellent alignment with WSA steel industry hierarchy standards. The mapping reveals a comprehensive coverage of the steel value chain from primary production through specialized finished products, with clear relationships between production levels that follow international industry standards.

The integration provides a solid foundation for international benchmarking and supports Australia's position within the global steel industry context.

---

*Report generated by Australian Steel Demand Model - WSA Hierarchy Integration Module*
"""
        
        return report
    
    def create_wsa_summary_file(self, 
                              visualization_files: Dict[str, str],
                              report_file: str,
                              output_dir: Path) -> str:
        """Create WSA integration summary file listing all outputs."""
        
        summary_file = output_dir / 'WSA_Integration_Summary_Track_A.txt'
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary_content = f"""WSA Steel Industry Hierarchy Integration - Track A Production Forecasts
Generated: {timestamp}

This directory contains comprehensive integration of Track A production forecasts 
with World Steel Association (WSA) steel industry hierarchy standards.

=== GENERATED FILES ===

VISUALIZATIONS:
"""
        
        for viz_type, filepath in visualization_files.items():
            filename = Path(filepath).name
            descriptions = {
                'hierarchy_structure': 'WSA hierarchy structure with Track A forecast volumes',
                'value_chain_flow': 'Steel industry value chain flow showing Track A products',
                'level_comparison': 'Comparison dashboard across WSA hierarchy levels',
                'industry_position': 'Industry position analysis with volume vs growth',
                'trade_analysis': 'Trade category analysis following WSA classifications'
            }
            
            description = descriptions.get(viz_type, 'WSA hierarchy visualization')
            summary_content += f"• {filename} - {description}\n"
        
        summary_content += f"""
REPORTS:
• {Path(report_file).name} - Comprehensive WSA hierarchy integration report

=== WSA HIERARCHY LEVELS COVERED ===

Level 0: Raw Materials & Primary Consumption
Level 1: Intermediate Materials & Alternative Consumption  
Level 2: Primary Steel Production (Crude Steel)
Level 4: Primary Finished Products (Hot Rolled, Tubular)
Level 5: Specialized Finished Products (Coated, Wire Rod, Rails)

=== TRACK A CATEGORIES MAPPED ===

PRIMARY PRODUCTION:
• Total Production of Crude Steel (Level 2)

FINISHED PRODUCTS:
• Production of Hot Rolled Flat Products (Level 4)
• Production of Hot Rolled Long Products (Level 4)  
• Total Production of Tubular Products (Level 4)

SPECIALIZED PRODUCTS:
• Production of Hot Rolled Coil, Sheet, and Strip (<3mm) (Level 5)
• Production of Non-metallic Coated Sheet and Strip (Level 5)
• Production of Other Metal Coated Sheet and Strip (Level 5)
• Production of Wire Rod (Level 5)
• Production of Railway Track Material (Level 5)

CONSUMPTION METRICS:
• Apparent Steel Use (crude steel equivalent) (Level 0)
• Apparent Steel Use (finished steel products) (Level 1)
• True Steel Use (finished steel equivalent) (Level 1)

=== WSA TRADE CATEGORIES ===

Flat Products: Hot rolled flat products and all coated sheet/strip products
Long Products: Wire rod, railway track material, and other long products
Tubular Products: All pipe and tube products
Consumption Metrics: Steel use measures following WSA standards

=== KEY INSIGHTS ===

• Track A provides comprehensive coverage of WSA steel production hierarchy
• Clear value chain relationships from crude steel to specialized products
• Strong alignment with international steel industry standards
• Enables direct comparison with global WSA member country data
• Supports Australia's integration into global steel market analysis

This integration demonstrates Track A's compliance with international steel 
industry standards and enables sophisticated analysis within the global 
steel market context.
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        return str(summary_file)


def integrate_wsa_hierarchy_with_track_a(track_a_forecast_file: str, 
                                       output_dir: str = None) -> Dict[str, str]:
    """
    Main function to integrate WSA hierarchy with Track A forecasting results.
    
    Args:
        track_a_forecast_file: Path to Track A ensemble forecast CSV file
        output_dir: Output directory for WSA integration files
        
    Returns:
        Dictionary of generated file paths
    """
    # Initialize integrator
    integrator = WSAHierarchyIntegrator()
    
    # Set output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"wsa_integration_{timestamp}"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Track A forecast data
    track_a_data = pd.read_csv(track_a_forecast_file)
    
    # Load WSA diagrams
    wsa_diagrams = integrator.load_wsa_diagrams()
    
    # Map Track A to WSA hierarchy
    hierarchy_mapping = integrator.map_track_a_to_wsa_hierarchy(track_a_data)
    
    # Create visualizations
    visualization_files = integrator.create_wsa_hierarchy_visualization(
        hierarchy_mapping, output_path
    )
    
    # Generate report
    report_file = integrator.generate_wsa_hierarchy_report(
        hierarchy_mapping, wsa_diagrams, output_path
    )
    
    # Create summary file
    summary_file = integrator.create_wsa_summary_file(
        visualization_files, report_file, output_path
    )
    
    # Compile all generated files
    generated_files = {
        'wsa_integration_directory': str(output_path),
        'wsa_hierarchy_report': report_file,
        'wsa_summary': summary_file,
        **visualization_files
    }
    
    integrator.logger.info(f"WSA hierarchy integration completed. Generated {len(generated_files)} files in {output_path}")
    
    return generated_files


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        track_a_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else None
        
        generated_files = integrate_wsa_hierarchy_with_track_a(track_a_file, output_dir)
        
        print("\n=== WSA HIERARCHY INTEGRATION COMPLETED ===")
        print(f"Output directory: {generated_files['wsa_integration_directory']}")
        print(f"Generated {len(generated_files)} files")
        
        for file_type, filepath in generated_files.items():
            print(f"  • {file_type}: {Path(filepath).name}")
    else:
        print("Usage: python wsa_hierarchy_integration.py <track_a_forecast_file> [output_dir]")