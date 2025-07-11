#!/usr/bin/env python3
"""
Steel Demand Model Visualization Module

Creates comprehensive visualizations for the 3-model ensemble system:
- Algorithm performance dashboards
- Feature importance heatmaps  
- Historical vs forecast comparisons
- Model accuracy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SteelDemandVisualizer:
    """Creates visualizations for steel demand forecasting results."""
    
    def __init__(self, output_dir: str):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up styling
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better quality
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 8
        plt.rcParams['ytick.labelsize'] = 8
        plt.rcParams['legend.fontsize'] = 9
        
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def create_algorithm_performance_dashboard(self, performance_df: pd.DataFrame) -> str:
        """Create comprehensive algorithm performance dashboard."""
        self.logger.info("Creating algorithm performance dashboard...")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Algorithm Performance Dashboard - 3-Model Ensemble', fontsize=16, fontweight='bold')
        
        # 1. Average MAPE by Algorithm
        avg_mape = performance_df.groupby('Algorithm')['MAPE'].mean().sort_values()
        colors = ['#2E8B57', '#FF6B35', '#4ECDC4']  # Green, Orange, Teal
        bars1 = ax1.bar(avg_mape.index, avg_mape.values, color=colors)
        ax1.set_title('Average MAPE by Algorithm', fontweight='bold')
        ax1.set_ylabel('MAPE (%)')
        ax1.set_ylim(0, max(avg_mape.values) * 1.2)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. MAPE by Category and Algorithm (grouped bar chart)
        categories = performance_df['Category'].unique()[:8]  # Top 8 categories for readability
        performance_subset = performance_df[performance_df['Category'].isin(categories)]
        
        # Pivot for grouped bar chart
        mape_pivot = performance_subset.pivot(index='Category', columns='Algorithm', values='MAPE')
        mape_pivot.plot(kind='bar', ax=ax2, width=0.8, color=colors)
        ax2.set_title('MAPE by Category and Algorithm', fontweight='bold')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_xlabel('')
        ax2.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. R² Score Distribution
        all_r2_scores = performance_df['R2'].values
        ax3.hist(all_r2_scores, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(all_r2_scores.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean R² = {all_r2_scores.mean():.3f}')
        ax3.set_title('R² Score Distribution', fontweight='bold')
        ax3.set_xlabel('R² Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Algorithm Performance Summary (normalized)
        avg_metrics = performance_df.groupby('Algorithm').agg({
            'MAPE': 'mean',
            'R2': 'mean'
        }).reset_index()
        
        # Normalize MAPE (lower is better, so invert)
        mape_normalized = 1 - (avg_metrics['MAPE'] / avg_metrics['MAPE'].max())
        
        x_pos = np.arange(len(avg_metrics))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, mape_normalized, width, 
                       label='MAPE (normalized)', color='lightcoral', alpha=0.8)
        bars2 = ax4.bar(x_pos + width/2, avg_metrics['R2'], width,
                       label='R² Score', color='lightblue', alpha=0.8)
        
        ax4.set_title('Algorithm Performance Summary', fontweight='bold')
        ax4.set_ylabel('Performance Score')
        ax4.set_xlabel('Algorithm')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(avg_metrics['Algorithm'])
        ax4.legend()
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "algorithm_accuracy_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Algorithm performance dashboard saved: {filename}")
        return str(filename)
    
    def create_performance_heatmaps(self, performance_df: pd.DataFrame) -> str:
        """Create performance heatmaps for MAPE and R² scores."""
        self.logger.info("Creating algorithm performance heatmaps...")
        
        # Create figure with side-by-side heatmaps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
        fig.suptitle('Algorithm Performance by Category (MAPE % and R²)', fontsize=16, fontweight='bold')
        
        # Prepare data for heatmaps
        mape_pivot = performance_df.pivot(index='Category', columns='Algorithm', values='MAPE')
        r2_pivot = performance_df.pivot(index='Category', columns='Algorithm', values='R2')
        
        # Sort categories by average performance
        category_order = mape_pivot.mean(axis=1).sort_values().index
        mape_pivot = mape_pivot.reindex(category_order)
        r2_pivot = r2_pivot.reindex(category_order)
        
        # 1. MAPE Heatmap (lower is better - use reversed colormap)
        sns.heatmap(mape_pivot, annot=True, fmt='.2f', ax=ax1, 
                   cmap='RdYlGn_r', cbar_kws={'label': 'MAPE (%)'})
        ax1.set_title('Algorithm Performance by Category (MAPE %)', fontweight='bold')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Steel Category')
        
        # 2. R² Heatmap (higher is better - use normal colormap)
        sns.heatmap(r2_pivot, annot=True, fmt='.3f', ax=ax2,
                   cmap='RdYlGn', cbar_kws={'label': 'R² Score'})
        ax2.set_title('Algorithm Performance by Category (R²)', fontweight='bold')
        ax2.set_xlabel('Algorithm')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "algorithm_performance_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Performance heatmaps saved: {filename}")
        return str(filename)
    
    def create_feature_importance_heatmap(self, results_dict: Dict) -> str:
        """Create feature importance heatmap for XGBoost and Random Forest."""
        self.logger.info("Creating feature importance heatmap...")
        
        # Collect feature importance data
        importance_data = []
        
        for category, algorithms in results_dict.items():
            for algo_name, result in algorithms.items():
                if 'feature_importance' in result and algo_name in ['XGBoost', 'RandomForest']:
                    importance_df = result['feature_importance']
                    
                    # Get top features for this category-algorithm combination
                    for _, row in importance_df.head(5).iterrows():  # Top 5 features
                        importance_data.append({
                            'Algorithm_Category': f"{algo_name} - {category}",
                            'Feature': row['feature'],
                            'Importance': row['importance'] if 'importance' in row else row.get('abs_coefficient', 0)
                        })
        
        if not importance_data:
            self.logger.warning("No feature importance data found")
            return ""
        
        # Create DataFrame and pivot for heatmap
        importance_df = pd.DataFrame(importance_data)
        
        # Create pivot table
        heatmap_data = importance_df.pivot_table(
            index='Algorithm_Category', 
            columns='Feature', 
            values='Importance', 
            fill_value=0
        )
        
        # Create the heatmap
        plt.figure(figsize=(16, 12))
        
        # Use a colormap that highlights important features
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Feature Importance'})
        
        plt.title('Feature Importance by Algorithm and Steel Category', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Economic Features', fontweight='bold')
        plt.ylabel('Algorithm - Steel Category', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "feature_importance_heatmap.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Feature importance heatmap saved: {filename}")
        return str(filename)
    
    def create_forecast_comparison_charts(self, historical_data: pd.DataFrame, 
                                        forecast_df: pd.DataFrame,
                                        ensemble_df: Optional[pd.DataFrame] = None) -> str:
        """Create historical vs forecast comparison charts."""
        self.logger.info("Creating historical vs forecast comparison charts...")
        
        # Get steel categories from historical data
        steel_columns = [col for col in historical_data.columns 
                        if col not in ['Year', 'Population', 'Urbanisation', 'GDP_AUD_Real2015', 
                                     'Iron_Ore_Production', 'Coal_Production']]
        
        # Select top categories with sufficient data
        categories_to_plot = []
        for col in steel_columns:
            if historical_data[col].notna().sum() >= 10:  # At least 10 data points
                categories_to_plot.append(col)
        
        # Limit to 12 categories for readability
        categories_to_plot = categories_to_plot[:12]
        
        # Create subplot grid
        n_categories = len(categories_to_plot)
        n_cols = 4
        n_rows = (n_categories + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        fig.suptitle('Historical Data, Individual Algorithms & Ensemble Forecasts (2004-2050)', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else list(axes)
        else:
            axes = axes.flatten()
        
        colors = {'XGBoost': '#2E8B57', 'RandomForest': '#FF6B35', 'LinearRegression': '#4ECDC4'}
        
        for idx, category in enumerate(categories_to_plot):
            ax = axes[idx]
            
            # Plot historical data
            hist_years = historical_data['Year']
            hist_values = historical_data[category]
            
            # Remove NaN values for plotting
            valid_hist = hist_values.notna()
            ax.plot(hist_years[valid_hist], hist_values[valid_hist], 
                   'ko-', linewidth=2, markersize=4, label='Historical Data')
            
            # Plot algorithm forecasts
            forecast_years = forecast_df['Year']
            
            for algo in ['XGBoost', 'RandomForest', 'LinearRegression']:
                forecast_col = f"{category}_{algo}"
                if forecast_col in forecast_df.columns:
                    ax.plot(forecast_years, forecast_df[forecast_col], 
                           '--', color=colors[algo], linewidth=1.5, 
                           alpha=0.7, label=f'{algo} Forecast')
            
            # Plot ensemble forecast if available
            if ensemble_df is not None:
                # Track A ensemble (primary ensemble)
                ensemble_col = f"{category}_Ensemble"
                if ensemble_col in ensemble_df.columns:
                    ax.plot(ensemble_df['Year'], ensemble_df[ensemble_col],
                           'purple', linestyle='-', linewidth=3, alpha=0.9, 
                           label='Track A Ensemble (XGB+RF)')
                
                # Legacy Track A column name (for backward compatibility)
                track_a_ensemble_col = f"{category}_Track_A_Ensemble"
                if track_a_ensemble_col in ensemble_df.columns and ensemble_col not in ensemble_df.columns:
                    ax.plot(ensemble_df['Year'], ensemble_df[track_a_ensemble_col],
                           'purple', linestyle='-', linewidth=3, alpha=0.9, 
                           label='Track A Ensemble (XGB+RF)')
            
            # Add vertical line at forecast start
            ax.axvline(x=2024, color='gray', linestyle=':', alpha=0.8)
            
            # Formatting
            ax.set_title(category.replace('_', ' '), fontweight='bold', fontsize=10)
            ax.set_xlabel('Year')
            ax.set_ylabel('Production (thousand tonnes)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Format y-axis for readability
            ax.ticklabel_format(style='plain', axis='y')
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))
        
        # Hide unused subplots
        for idx in range(n_categories, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "historical_ensemble_vs_algorithms.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Forecast comparison charts saved: {filename}")
        return str(filename)
    
    def create_track_a_comparison_chart(self, ensemble_df: pd.DataFrame) -> str:
        """Create Track A vs Full Ensemble comparison chart."""
        self.logger.info("Creating Track A vs Full Ensemble comparison chart...")
        
        if ensemble_df is None:
            self.logger.warning("No ensemble data available for Track A comparison")
            return ""
        
        # Check for both ensemble types
        track_a_cols = [col for col in ensemble_df.columns if 'Track_A_Ensemble' in col]
        full_cols = [col for col in ensemble_df.columns if 'Full_Ensemble' in col]
        
        if not track_a_cols or not full_cols:
            self.logger.warning("Missing Track A or Full ensemble data for comparison")
            return ""
        
        # Create comparison figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Track A vs Full Ensemble Comparison', fontsize=16, fontweight='bold')
        
        years = ensemble_df['Year']
        
        # 1. Total Steel Consumption Comparison
        if 'Total_Steel_Consumption_Track_A_Ensemble' in ensemble_df.columns and \
           'Total_Steel_Consumption_Full_Ensemble' in ensemble_df.columns:
            
            track_a_total = ensemble_df['Total_Steel_Consumption_Track_A_Ensemble']
            full_total = ensemble_df['Total_Steel_Consumption_Full_Ensemble']
            
            ax1.plot(years, track_a_total, 'purple', linewidth=3, alpha=0.9, 
                    label='Track A (XGBoost + RF)')
            ax1.plot(years, full_total, 'red', linewidth=3, alpha=0.8, 
                    label='Full Ensemble (XGB + RF + LR)')
            ax1.axvline(x=2024, color='gray', linestyle=':', alpha=0.8)
            ax1.set_title('Total Steel Consumption Forecast', fontweight='bold')
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Consumption (thousand tonnes)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 2. Percentage Difference Over Time
        if 'Total_Steel_Consumption_Track_A_Ensemble' in ensemble_df.columns and \
           'Total_Steel_Consumption_Full_Ensemble' in ensemble_df.columns:
            
            track_a_total = ensemble_df['Total_Steel_Consumption_Track_A_Ensemble']
            full_total = ensemble_df['Total_Steel_Consumption_Full_Ensemble']
            
            # Calculate percentage difference: (Track A - Full) / Full * 100
            pct_diff = ((track_a_total - full_total) / full_total) * 100
            
            ax2.plot(years, pct_diff, 'darkgreen', linewidth=2, marker='o', markersize=3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(x=2024, color='gray', linestyle=':', alpha=0.8)
            ax2.set_title('Track A vs Full Ensemble Difference', fontweight='bold')
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Difference (%)')
            ax2.grid(True, alpha=0.3)
            
            # Add annotation for average difference
            avg_diff = pct_diff.mean()
            ax2.text(0.02, 0.98, f'Avg Difference: {avg_diff:.2f}%', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 3. Category-wise Comparison (Top 6 categories)
        category_pairs = []
        for col in ensemble_df.columns:
            if col.endswith('_Track_A_Ensemble'):
                category = col.replace('_Track_A_Ensemble', '')
                full_col = f'{category}_Full_Ensemble'
                if full_col in ensemble_df.columns:
                    category_pairs.append((category, col, full_col))
        
        # Select top 6 categories by magnitude
        category_magnitudes = []
        for category, track_a_col, full_col in category_pairs:
            avg_magnitude = (ensemble_df[track_a_col].mean() + ensemble_df[full_col].mean()) / 2
            category_magnitudes.append((category, track_a_col, full_col, avg_magnitude))
        
        # Sort by magnitude and take top 6
        category_magnitudes.sort(key=lambda x: x[3], reverse=True)
        top_categories = category_magnitudes[:6]
        
        # Plot category comparison
        categories_names = [cat[0].replace('_', ' ')[:20] for cat in top_categories]  # Truncate names
        track_a_2050 = [ensemble_df[cat[1]].iloc[-1] for cat in top_categories]
        full_2050 = [ensemble_df[cat[2]].iloc[-1] for cat in top_categories]
        
        x_pos = np.arange(len(categories_names))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, track_a_2050, width, 
                       label='Track A 2050', color='purple', alpha=0.7)
        bars2 = ax3.bar(x_pos + width/2, full_2050, width,
                       label='Full Ensemble 2050', color='red', alpha=0.7)
        
        ax3.set_title('2050 Forecasts by Category', fontweight='bold')
        ax3.set_ylabel('Consumption (thousand tonnes)')
        ax3.set_xlabel('Steel Category')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(categories_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        # 4. Summary Statistics Table
        ax4.axis('off')
        
        # Calculate summary statistics
        if 'Total_Steel_Consumption_Track_A_Ensemble' in ensemble_df.columns and \
           'Total_Steel_Consumption_Full_Ensemble' in ensemble_df.columns:
            
            track_a_total = ensemble_df['Total_Steel_Consumption_Track_A_Ensemble']
            full_total = ensemble_df['Total_Steel_Consumption_Full_Ensemble']
            
            # Growth rates
            track_a_growth = ((track_a_total.iloc[-1] / track_a_total.iloc[0]) ** (1/25) - 1) * 100
            full_growth = ((full_total.iloc[-1] / full_total.iloc[0]) ** (1/25) - 1) * 100
            
            # Summary table data
            summary_data = [
                ['Metric', 'Track A Ensemble', 'Full Ensemble', 'Difference'],
                ['2025 Forecast (Mt)', f'{track_a_total.iloc[0]/1e6:.2f}', f'{full_total.iloc[0]/1e6:.2f}', 
                 f'{(track_a_total.iloc[0] - full_total.iloc[0])/1e6:.2f}'],
                ['2050 Forecast (Mt)', f'{track_a_total.iloc[-1]/1e6:.2f}', f'{full_total.iloc[-1]/1e6:.2f}', 
                 f'{(track_a_total.iloc[-1] - full_total.iloc[-1])/1e6:.2f}'],
                ['Annual Growth Rate', f'{track_a_growth:.2f}%', f'{full_growth:.2f}%', 
                 f'{track_a_growth - full_growth:.2f}pp'],
                ['Composition', 'XGBoost (60%) + RF (40%)', 'XGB (50%) + RF (35%) + LR (15%)', '-'],
                ['Model Count', '2 algorithms', '3 algorithms', '-1']
            ]
            
            # Create table
            table = ax4.table(cellText=summary_data[1:], colLabels=summary_data[0],
                             cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            
            # Style the header row
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#E6E6FA')
                table[(0, i)].set_text_props(weight='bold')
        
        ax4.set_title('Ensemble Comparison Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "track_a_vs_full_ensemble_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Track A comparison chart saved: {filename}")
        return str(filename)
    
    def create_model_summary_report(self, performance_df: pd.DataFrame,
                                  total_categories: int,
                                  training_time: float = None) -> str:
        """Create a summary report visualization."""
        self.logger.info("Creating model summary report...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Steel Demand Model - Training Summary Report', fontsize=16, fontweight='bold')
        
        # 1. Model Performance Summary Table
        summary_stats = performance_df.groupby('Algorithm').agg({
            'MAPE': ['mean', 'std', 'min', 'max'],
            'R2': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Flatten column names
        summary_stats.columns = [f"{metric}_{stat}" for metric, stat in summary_stats.columns]
        
        # Create table visualization
        ax1.axis('tight')
        ax1.axis('off')
        table_data = summary_stats.reset_index().values
        table_cols = ['Algorithm'] + list(summary_stats.columns)
        
        table = ax1.table(cellText=table_data, colLabels=table_cols,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax1.set_title('Performance Statistics Summary', fontweight='bold', pad=20)
        
        # 2. Algorithm Rankings
        avg_mape = performance_df.groupby('Algorithm')['MAPE'].mean().sort_values()
        avg_r2 = performance_df.groupby('Algorithm')['R2'].mean().sort_values(ascending=False)
        
        rankings = pd.DataFrame({
            'Algorithm': avg_mape.index,
            'MAPE_Rank': range(1, len(avg_mape) + 1),
            'R2_Rank': [list(avg_r2.index).index(algo) + 1 for algo in avg_mape.index]
        })
        rankings['Overall_Score'] = (rankings['MAPE_Rank'] + rankings['R2_Rank']) / 2
        
        colors = ['gold', 'silver', '#CD7F32']  # Gold, Silver, Bronze
        bars = ax2.bar(rankings['Algorithm'], rankings['Overall_Score'], color=colors)
        ax2.set_title('Algorithm Rankings (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Average Rank Score')
        ax2.set_ylim(0, max(rankings['Overall_Score']) * 1.2)
        
        # Add rank labels
        for bar, score in zip(bars, rankings['Overall_Score']):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Categories by Performance Level
        performance_levels = pd.cut(performance_df['MAPE'], 
                                  bins=[0, 2, 5, 10, float('inf')], 
                                  labels=['Excellent (<2%)', 'Good (2-5%)', 
                                         'Acceptable (5-10%)', 'Needs Improvement (>10%)'])
        
        level_counts = performance_levels.value_counts()
        colors_pie = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
        
        wedges, texts, autotexts = ax3.pie(level_counts.values, labels=level_counts.index,
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax3.set_title('Model Performance Distribution', fontweight='bold')
        
        # 4. System Information
        ax4.axis('off')
        info_text = f"""
System Configuration:
├─ Total Steel Categories: {total_categories}
├─ Training Categories: {len(performance_df['Category'].unique())}
├─ Algorithms: {', '.join(performance_df['Algorithm'].unique())}
├─ Ensemble: Track A (XGBoost 60% + Random Forest 40%)
├─ Forecast Period: 2025-2050
└─ Output Directory: {self.output_dir.name}

Performance Highlights:
├─ Best MAPE: {performance_df['MAPE'].min():.2f}% ({performance_df.loc[performance_df['MAPE'].idxmin(), 'Algorithm']})
├─ Best R²: {performance_df['R2'].max():.3f} ({performance_df.loc[performance_df['R2'].idxmax(), 'Algorithm']})
├─ Avg MAPE: {performance_df['MAPE'].mean():.2f}%
└─ Avg R²: {performance_df['R2'].mean():.3f}

Model Status: ✅ Training Complete
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        ax4.set_title('System Information', fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        filename = self.viz_dir / "model_summary_report.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Model summary report saved: {filename}")
        return str(filename)
    
    def generate_all_visualizations(self, performance_df: pd.DataFrame,
                                  results_dict: Dict,
                                  historical_data: pd.DataFrame,
                                  forecast_df: pd.DataFrame,
                                  ensemble_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """Generate all visualization types."""
        self.logger.info("Generating all visualizations for 3-model ensemble...")
        
        generated_files = {}
        
        try:
            # 1. Algorithm Performance Dashboard
            generated_files['dashboard'] = self.create_algorithm_performance_dashboard(performance_df)
            
            # 2. Performance Heatmaps
            generated_files['heatmaps'] = self.create_performance_heatmaps(performance_df)
            
            # 3. Feature Importance Heatmap
            generated_files['feature_importance'] = self.create_feature_importance_heatmap(results_dict)
            
            # 4. Historical vs Forecast Comparison
            generated_files['forecast_comparison'] = self.create_forecast_comparison_charts(
                historical_data, forecast_df, ensemble_df)
            
            # Note: Track A vs Full Ensemble comparison removed since only Track A ensemble remains
            
            # 5. Model Summary Report
            generated_files['summary_report'] = self.create_model_summary_report(
                performance_df, len(historical_data.columns) - 6)  # Exclude non-steel columns
            
            self.logger.info(f"All visualizations completed. Files saved to: {self.viz_dir}")
            
            # Create summary file
            summary_file = self.viz_dir / "visualization_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("Steel Demand Model - Visualization Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Output directory: {self.viz_dir}\n\n")
                f.write("Generated visualizations:\n")
                for viz_type, filepath in generated_files.items():
                    f.write(f"  • {viz_type}: {Path(filepath).name}\n")
            
            generated_files['summary'] = str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
        
        return generated_files

def create_visualizations_for_results(output_dir: str, 
                                    performance_file: str = "Algorithm_Performance_Comparison.csv",
                                    forecast_file: str = "ML_Algorithm_Forecasts_2025-2050.csv",
                                    ensemble_file: str = "Ensemble_Forecasts_2025-2050.csv") -> Dict[str, str]:
    """
    Standalone function to create visualizations from saved results.
    
    Args:
        output_dir: Directory containing the training results
        performance_file: Name of performance comparison CSV file
        forecast_file: Name of forecast results CSV file
        ensemble_file: Name of ensemble forecast CSV file
    
    Returns:
        Dictionary mapping visualization types to file paths
    """
    output_path = Path(output_dir)
    
    # Load data
    performance_df = pd.read_csv(output_path / performance_file)
    forecast_df = pd.read_csv(output_path / forecast_file)
    
    # Try to load ensemble file
    ensemble_df = None
    ensemble_path = output_path / ensemble_file
    if ensemble_path.exists():
        ensemble_df = pd.read_csv(ensemble_path)
    
    # Load historical data
    from data.data_loader import SteelDemandDataLoader
    loader = SteelDemandDataLoader("config/")
    loader.load_all_data()
    historical_data = loader.get_historical_data()
    
    # Create dummy results dict for feature importance (if needed)
    results_dict = {}
    
    # Initialize visualizer
    visualizer = SteelDemandVisualizer(output_dir)
    
    # Generate visualizations
    return visualizer.generate_all_visualizations(
        performance_df, results_dict, historical_data, forecast_df, ensemble_df
    )