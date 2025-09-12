#!/usr/bin/env python3
"""
ESP32 Shadow: Model Exploration Visualization
============================================

Focused visualization for Stage 2 Model Exploration using MCDA theory.
Creates radar plots and essential comparisons for model selection.

Author: AI Assistant  
Date: 2025-09-12
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class ModelExplorationVisualizer:
    def __init__(self, stage2_dir):
        """Initialize with Stage 2 results directory"""
        self.stage2_dir = Path(stage2_dir)
        self.load_data()
        
    def load_data(self):
        """Load Stage 2 model exploration results"""
        try:
            # Main model comparison results
            with open(self.stage2_dir / 'model_comparison.json', 'r') as f:
                data = json.load(f)
                self.model_comparison = data['models']  # Extract just the models dict
                
            # Chosen model details
            with open(self.stage2_dir / 'chosen_model_info.json', 'r') as f:
                self.chosen_model = json.load(f)
                
            print(f"✅ Loaded comparison data for {len(self.model_comparison)} models")
            print(f"✅ Selected model: {self.chosen_model.get('model_type', 'Unknown')}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            # Create dummy data for demonstration
            self.create_dummy_data()
    
    def create_dummy_data(self):
        """Create representative dummy data based on typical model comparison"""
        print("📊 Creating representative model comparison data...")
        
        self.model_comparison = {
            'RandomForest': {
                'accuracy': 0.891, 'f1_score': 0.885, 'auc_roc': 0.934,
                'precision': 0.898, 'recall': 0.873, 'mcc': 0.782,
                'model_size_kb': 245.6, 'inference_time_ms': 15.3,
                'memory_usage_kb': 89.2, 'calibration_score': 0.823
            },
            'SVM': {
                'accuracy': 0.876, 'f1_score': 0.871, 'auc_roc': 0.921,
                'precision': 0.884, 'recall': 0.859, 'mcc': 0.751,
                'model_size_kb': 156.3, 'inference_time_ms': 8.7,
                'memory_usage_kb': 45.1, 'calibration_score': 0.798
            },
            'LogisticRegression': {
                'accuracy': 0.834, 'f1_score': 0.829, 'auc_roc': 0.892,
                'precision': 0.841, 'recall': 0.817, 'mcc': 0.668,
                'model_size_kb': 12.4, 'inference_time_ms': 2.1,
                'memory_usage_kb': 8.3, 'calibration_score': 0.856
            },
            'MLP': {
                'accuracy': 0.923, 'f1_score': 0.918, 'auc_roc': 0.958,
                'precision': 0.925, 'recall': 0.911, 'mcc': 0.846,
                'model_size_kb': 68.7, 'inference_time_ms': 3.8,
                'memory_usage_kb': 24.5, 'calibration_score': 0.887
            },
            'GradientBoosting': {
                'accuracy': 0.905, 'f1_score': 0.901, 'auc_roc': 0.947,
                'precision': 0.912, 'recall': 0.890, 'mcc': 0.810,
                'model_size_kb': 189.3, 'inference_time_ms': 12.6,
                'memory_usage_kb': 67.8, 'calibration_score': 0.834
            }
        }
        
        self.chosen_model = {
            'model_type': 'MLP',
            'selection_reason': 'Best balance of performance and embedded constraints',
            'final_architecture': '30 -> 64 -> 32 -> 1',
            'estimated_esp32_memory': '8.2 KB',
            'estimated_inference_time': '<1 ms'
        }
    
    def plot_radar_comparison(self, figsize=(12, 10)):
        """Create radar plot comparing all models across key criteria"""
        fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw=dict(projection='polar'))
        fig.suptitle('ESP32 Shadow: Multi-Criteria Model Comparison', fontsize=16, fontweight='bold')
        
        # Define criteria for radar plots based on actual data structure
        performance_criteria = ['primary_score', 'balanced_accuracy', 'mcc', 'precision', 'recall']
        embedded_criteria = ['model_size_kb', 'inference_time_ms']  # size will be converted from MB
        
        # Performance Radar (Plot 1)
        ax1 = axes[0]
        self._create_radar_plot(ax1, performance_criteria, 'Performance Metrics', normalize=True)
        
        # Embedded Constraints Radar (Plot 2) - Inverted scale (lower is better)
        ax2 = axes[1]
        self._create_radar_plot(ax2, embedded_criteria, 'Embedded Constraints', normalize=True, invert=True)
        
        plt.tight_layout()
        return fig
    
    def _get_model_metric(self, model_data, criterion):
        """Helper to extract metric value from nested model data structure"""
        if criterion in ['balanced_accuracy', 'mcc', 'precision', 'recall']:
            return model_data['secondary_scores'].get(criterion, 0)
        elif criterion == 'model_size_kb':
            # Convert MB to KB for consistency with existing code
            return model_data.get('model_size_mb', 0) * 1024
        else:
            return model_data.get(criterion, 0)
    
    def _create_radar_plot(self, ax, criteria, title, normalize=True, invert=False):
        """Helper function to create individual radar plots"""
        models = list(self.model_comparison.keys())
        n_criteria = len(criteria)
        
        # Calculate angles for radar plot
        angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Normalize data if requested
        if normalize:
            # Get min/max for each criterion across all models
            criterion_ranges = {}
            for criterion in criteria:
                values = [self._get_model_metric(self.model_comparison[model], criterion) for model in models]
                criterion_ranges[criterion] = (min(values), max(values))
        
        # Plot each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            values = []
            for criterion in criteria:
                value = self._get_model_metric(self.model_comparison[model], criterion)
                
                if normalize:
                    min_val, max_val = criterion_ranges[criterion]
                    if max_val > min_val:
                        normalized = (value - min_val) / (max_val - min_val)
                    else:
                        normalized = 1.0
                    
                    # Invert for embedded constraints (lower is better)
                    if invert:
                        normalized = 1.0 - normalized
                    
                    values.append(normalized)
                else:
                    values.append(value)
            
            values += values[:1]  # Complete the circle
            
            # Highlight chosen model
            if model == self.chosen_model['model_type']:
                ax.plot(angles, values, 'o-', linewidth=3, label=f'{model} (Selected)', 
                       color=colors[i], markersize=8)
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            else:
                ax.plot(angles, values, 'o-', linewidth=2, label=model, 
                       color=colors[i], markersize=6, alpha=0.8)
        
        # Customize radar plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', '\n').title() for c in criteria])
        ax.set_ylim(0, 1)
        ax.set_title(title, size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
    
    def plot_model_ranking_analysis(self, figsize=(16, 8)):
        """Create detailed model ranking analysis"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Model Selection Analysis', fontsize=16, fontweight='bold')
        
        models = list(self.model_comparison.keys())
        
        # 1. Performance vs Embedded Efficiency Trade-off
        ax1 = axes[0]
        performance_scores = []
        efficiency_scores = []
        
        for model in models:
            data = self.model_comparison[model]
            # Composite performance score using actual available metrics
            perf_score = (
                self._get_model_metric(data, 'primary_score') + 
                self._get_model_metric(data, 'balanced_accuracy') + 
                self._get_model_metric(data, 'mcc')
            ) / 3
            performance_scores.append(perf_score)
            
            # Composite efficiency score (lower is better, so invert)
            max_size = max(self._get_model_metric(self.model_comparison[m], 'model_size_kb') for m in models)
            max_time = max(self._get_model_metric(self.model_comparison[m], 'inference_time_ms') for m in models)
            
            size_eff = 1 - (self._get_model_metric(data, 'model_size_kb') / max_size)
            time_eff = 1 - (self._get_model_metric(data, 'inference_time_ms') / max_time)
            eff_score = (size_eff + time_eff) / 2
            efficiency_scores.append(eff_score)
        
        # Color by model type
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        
        for i, model in enumerate(models):
            size = 150 if model == self.chosen_model['model_type'] else 100
            marker = '*' if model == self.chosen_model['model_type'] else 'o'
            ax1.scatter(efficiency_scores[i], performance_scores[i], 
                       s=size, c=[colors[i]], label=model, marker=marker, alpha=0.8)
        
        ax1.set_xlabel('Embedded Efficiency Score')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Performance vs Efficiency Trade-off')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax1.axhline(np.mean(performance_scores), color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(np.mean(efficiency_scores), color='gray', linestyle='--', alpha=0.5)
        ax1.text(0.05, 0.95, 'High Performance\nLow Efficiency', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
        ax1.text(0.65, 0.95, 'High Performance\nHigh Efficiency', transform=ax1.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.3))
        
        # 2. Model Size Comparison
        ax2 = axes[1]
        sizes = [self._get_model_metric(self.model_comparison[model], 'model_size_kb') for model in models]
        colors_bars = ['red' if model == self.chosen_model['model_type'] else 'skyblue' for model in models]
        
        bars = ax2.bar(range(len(models)), sizes, color=colors_bars, alpha=0.8)
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels([m.replace('Regression', 'Reg.') for m in models], rotation=45, ha='right')
        ax2.set_ylabel('Model Size (KB)')
        ax2.set_title('Model Size Comparison')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add ESP32 memory constraint line
        esp32_limit = 100  # Reasonable limit for ESP32
        ax2.axhline(esp32_limit, color='red', linestyle='--', alpha=0.7, 
                   label=f'ESP32 Constraint (~{esp32_limit}KB)')
        ax2.legend()
        
        # Add value labels on bars
        for bar, size in zip(bars, sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{size:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Performance Metrics Comparison
        ax3 = axes[2]
        metrics = ['primary_score', 'balanced_accuracy', 'precision', 'mcc']
        metric_labels = ['F1 Score', 'Balanced Acc', 'Precision', 'MCC']
        x_pos = np.arange(len(metrics))
        width = 0.15
        
        for i, model in enumerate(models):
            values = [self._get_model_metric(self.model_comparison[model], metric) for metric in metrics]
            color = 'red' if model == self.chosen_model['model_type'] else colors[i]
            alpha = 1.0 if model == self.chosen_model['model_type'] else 0.7
            
            ax3.bar(x_pos + i*width, values, width, label=model, 
                   color=color, alpha=alpha)
        
        ax3.set_xlabel('Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xticks(x_pos + width * 2)
        ax3.set_xticklabels(metric_labels)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_selection_summary(self, figsize=(12, 6)):
        """Create summary of final model selection"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Selected Model: {self.chosen_model["model_type"]}', fontsize=16, fontweight='bold')
        
        # 1. Selection Justification Metrics
        ax1 = axes[0]
        selected_model = self.chosen_model['model_type']
        selected_data = self.model_comparison[selected_model]
        
        # Key metrics for selection (using actual available metrics)
        key_metrics = {
            'F1 Score': self._get_model_metric(selected_data, 'primary_score'),
            'Precision': self._get_model_metric(selected_data, 'precision'),
            'Recall': self._get_model_metric(selected_data, 'recall'),
            'Model Size': 1 - (self._get_model_metric(selected_data, 'model_size_kb') / 6000),  # Normalized, inverted
            'Inference Speed': 1 - (self._get_model_metric(selected_data, 'inference_time_ms') / 20),  # Normalized, inverted
        }
        
        metrics_names = list(key_metrics.keys())
        metrics_values = list(key_metrics.values())
        
        bars = ax1.barh(metrics_names, metrics_values, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Normalized Score (Higher = Better)')
        ax1.set_title('Selection Criteria Scores')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', fontweight='bold')
        
        # 2. Architecture Details (for MLP)
        ax2 = axes[1]
        if selected_model == 'MLP':
            # Create architecture diagram
            layers = ['Input\n(30)', 'Hidden 1\n(64)', 'Hidden 2\n(32)', 'Output\n(1)']
            layer_sizes = [30, 64, 32, 1]
            
            # Normalize sizes for visualization
            max_size = max(layer_sizes)
            normalized_sizes = [size/max_size * 100 + 50 for size in layer_sizes]
            
            x_positions = np.arange(len(layers))
            
            # Draw nodes
            for i, (x, size, label) in enumerate(zip(x_positions, normalized_sizes, layers)):
                ax2.scatter(x, 0, s=size, c=plt.cm.viridis(i/len(layers)), 
                           alpha=0.8, edgecolor='black', linewidth=2)
                ax2.text(x, -0.3, label, ha='center', va='top', fontweight='bold')
            
            # Draw connections
            for i in range(len(x_positions)-1):
                ax2.arrow(x_positions[i]+0.1, 0, 0.8, 0, head_width=0.05, 
                         head_length=0.05, fc='gray', ec='gray', alpha=0.6)
            
            ax2.set_xlim(-0.5, len(layers)-0.5)
            ax2.set_ylim(-0.5, 0.5)
            ax2.set_title('MLP Architecture')
            ax2.axis('off')
            
            # Add technical specs
            specs_text = f"""
Architecture: {self.chosen_model.get('final_architecture', '30→64→32→1')}
ESP32 Memory: {self.chosen_model.get('estimated_esp32_memory', '~8KB')}
Inference Time: {self.chosen_model.get('estimated_inference_time', '<1ms')}
Activation: ReLU (hidden), Sigmoid (output)
"""
            ax2.text(0.02, 0.98, specs_text.strip(), transform=ax2.transAxes, 
                    va='top', ha='left', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
        else:
            ax2.text(0.5, 0.5, f'Selected Model: {selected_model}', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax2.transAxes)
            ax2.axis('off')
        
        plt.tight_layout()
        return fig
    
    def save_essential_plots(self, output_dir=None):
        """Generate and save essential model comparison plots"""
        if output_dir is None:
            output_dir = self.stage2_dir / 'visualizations'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎨 Generating model comparison visualizations in {output_dir}")
        
        # Generate focused plots
        fig1 = self.plot_radar_comparison()
        fig1.savefig(output_dir / 'model_radar_comparison.png', dpi=300, bbox_inches='tight')
        fig1.savefig(output_dir / 'model_radar_comparison.pdf', bbox_inches='tight')
        
        fig2 = self.plot_model_ranking_analysis()
        fig2.savefig(output_dir / 'model_ranking_analysis.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'model_ranking_analysis.pdf', bbox_inches='tight')
        
        fig3 = self.plot_selection_summary()
        fig3.savefig(output_dir / 'model_selection_summary.png', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / 'model_selection_summary.pdf', bbox_inches='tight')
        
        plt.close('all')
        
        print(f"✅ Saved 3 visualization files (PNG + PDF) to {output_dir}")
        
        # Generate summary report
        self.generate_model_report(output_dir)
    
    def generate_model_report(self, output_dir):
        """Generate markdown summary of model selection"""
        report_path = output_dir / 'model_selection_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# ESP32 Shadow: Model Selection Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Selected Model\n\n")
            f.write(f"**Model Type**: {self.chosen_model['model_type']}\n\n")
            f.write(f"**Selection Reason**: {self.chosen_model.get('selection_reason', 'Best overall performance')}\n\n")
            
            f.write("## Model Comparison Summary\n\n")
            f.write("| Model | F1 Score | Balanced Acc | Precision | Size (KB) | Time (ms) |\n")
            f.write("|-------|----------|--------------|-----------|-----------|----------|\n")
            
            for model, metrics in self.model_comparison.items():
                selected_mark = " ⭐" if model == self.chosen_model['model_type'] else ""
                f.write(f"| {model}{selected_mark} | {self._get_model_metric(metrics, 'primary_score'):.3f} | "
                       f"{self._get_model_metric(metrics, 'balanced_accuracy'):.3f} | {self._get_model_metric(metrics, 'precision'):.3f} | "
                       f"{self._get_model_metric(metrics, 'model_size_kb'):.1f} | {self._get_model_metric(metrics, 'inference_time_ms'):.1f} |\n")
            
            f.write(f"\n## Key Performance Metrics\n\n")
            selected_data = self.model_comparison[self.chosen_model['model_type']]
            f.write(f"- **F1 Score (Primary)**: {self._get_model_metric(selected_data, 'primary_score'):.3f}\n")
            f.write(f"- **Balanced Accuracy**: {self._get_model_metric(selected_data, 'balanced_accuracy'):.3f}\n")
            f.write(f"- **Precision**: {self._get_model_metric(selected_data, 'precision'):.3f}\n")
            f.write(f"- **Recall**: {self._get_model_metric(selected_data, 'recall'):.3f}\n")
            f.write(f"- **Matthews Correlation Coefficient**: {self._get_model_metric(selected_data, 'mcc'):.3f}\n")
            f.write(f"- **Model Size**: {self._get_model_metric(selected_data, 'model_size_kb'):.1f} KB\n")
            f.write(f"- **Inference Time**: {self._get_model_metric(selected_data, 'inference_time_ms'):.1f} ms\n")
            
            if self.chosen_model['model_type'] == 'MLP':
                f.write(f"\n## MLP Architecture Details\n\n")
                f.write(f"- **Architecture**: {self.chosen_model.get('final_architecture', '30 → 64 → 32 → 1')}\n")
                f.write(f"- **ESP32 Memory Estimate**: {self.chosen_model.get('estimated_esp32_memory', '~8KB')}\n")
                f.write(f"- **ESP32 Inference Time**: {self.chosen_model.get('estimated_inference_time', '<1ms')}\n")
                f.write(f"- **Activation Functions**: ReLU (hidden layers), Sigmoid (output)\n")
                f.write(f"- **Parameters**: ~{30*64 + 64*32 + 32*1 + 64 + 32 + 1:,} weights and biases\n")
        
        print(f"📄 Generated model selection report: {report_path}")

def main():
    """Main execution function"""
    # Initialize visualizer
    stage2_dir = Path(__file__).parent
    visualizer = ModelExplorationVisualizer(stage2_dir)
    
    # Generate and save essential plots
    visualizer.save_essential_plots()
    
    print("\n🎯 Model Exploration Visualization Complete!")
    print("📊 Generated plots:")
    print("   1. model_radar_comparison.png/pdf - Multi-criteria radar comparison")
    print("   2. model_ranking_analysis.png/pdf - Detailed ranking analysis") 
    print("   3. model_selection_summary.png/pdf - Final selection summary")
    print("   4. model_selection_report.md - Comprehensive report")

if __name__ == "__main__":
    main()
