"""
Steel Demand Model Visualization Module

Contains visualization utilities for the Australian Steel Demand Model.
Generates comprehensive charts and dashboards for model results.

Track A: Current model with ensemble forecasting
Track B: Hierarchical forecasting with multi-level taxonomy
"""

from .steel_demand_visualizer import SteelDemandVisualizer, create_visualizations_for_results
from .hierarchical_visualizer import HierarchicalSteelDemandVisualizer, create_track_b_visualizations

def create_track_a_visualizations(output_dir: str, **kwargs):
    """
    Create Track A (current model ensemble) visualizations.
    
    Args:
        output_dir: Directory containing Track A results
        **kwargs: Additional arguments for visualization
    
    Returns:
        Dictionary of generated visualization files
    """
    return create_visualizations_for_results(output_dir, **kwargs)

__all__ = [
    'SteelDemandVisualizer', 
    'HierarchicalSteelDemandVisualizer',
    'create_visualizations_for_results',
    'create_track_a_visualizations', 
    'create_track_b_visualizations'
]