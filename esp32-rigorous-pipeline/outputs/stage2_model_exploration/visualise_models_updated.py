#!/usr/bin/env python3
"""
ESP32 Shadow: Stage 2 Model Comparison Visualization (Updated)

This script creates publication-ready visualizations for the model comparison stage.
Uses the updated data structure from the latest pipeline output.

Author: Ashidu Dissanayake
Date: September 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ModelComparisonVisualizer:
    def __init__(self, stage2_dir=None):
        self.stage2_dir = Path(stage2_dir) if stage2_dir else Path('.')
        self.output_dir = self.stage2_dir / 'visualizations'
        self.output_dir.mkdir(exist_ok=True)
        
        # Load updated data structure
        self.load_data()
        
    def load_data(self):
        """Load model comparison data from updated structure"""
        try:
            # Model ranking
            with open(self.stage2_dir / 'model_ranking.json', 'r') as f:
                self.ranking_data = json.load(f)
                
            # Aggregated metrics
            with open(self.stage2_dir / 'aggregated_metrics.json', 'r') as f:
                self.metrics_data = json.load(f)
                
            # Final model info
            with open(self.stage2_dir / 'final_model_artifacts.json', 'r') as f:
                self.final_model = json.load(f)
                
            # Stage summary
            with open(self.stage2_dir / 'stage2_summary.json', 'r') as f:
                self.summary_data = json.load(f)
                
            print(f"✅ Loaded data for {len(self.metrics_data)} models")
            print(f"✅ Selected model: {self.final_model['model_type']}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def get_model_metric(self, model_name, metric_name):
        """Helper to extract specific metric for a model"""
        return self.metrics_data[model_name].get(metric_name, 0)
    
    def plot_model_ranking_radar(self):
        """Create radar plot showing model ranking comparison"""
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # Define key metrics for comparison
        metrics = ['mean_f1', 'mean_balanced_accuracy', 'mean_precision', 'mean_recall', 'mean_mcc']
        metric_labels = ['F1 Score', 'Balanced Accuracy', 'Precision', 'Recall', 'MCC']
        
        # Colors for each model
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.ranking_data['ranking'])))
        
        for i, model_name in enumerate(self.ranking_data['ranking']):
            if model_name in self.metrics_data:
                # Extract values
                values = [self.get_model_metric(model_name, metric) for metric in metrics]
                
                # Close the plot
                values += values[:1]
                
                # Angles for each metric
                angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
                angles += angles[:1]
                
                # Highlight selected model
                linewidth = 3 if model_name == self.final_model['model_type'] else 2
                alpha = 0.8 if model_name == self.final_model['model_type'] else 0.6
                
                # Plot
                ax.plot(angles, values, 'o-', linewidth=linewidth, 
                       label=model_name.replace('_', ' ').title(), 
                       color=colors[i], alpha=alpha)
                ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Customize plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        plt.title('ESP32 Shadow: Model Performance Comparison\n(Radar Chart)', 
                 size=16, fontweight='bold', pad=30)
        plt.tight_layout()
        
        return fig
    
    def plot_performance_metrics_comparison(self):
        """Create bar plot comparing key performance metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        models = list(self.metrics_data.keys())
        
        # 1. Primary Metrics Comparison
        metrics = ['mean_f1', 'mean_balanced_accuracy', 'mean_precision', 'mean_recall']
        metric_labels = ['F1 Score', 'Balanced Acc', 'Precision', 'Recall']
        
        x = np.arange(len(metric_labels))
        width = 0.12
        
        for i, model in enumerate(models):
            values = [self.get_model_metric(model, metric) for metric in metrics]
            color = 'red' if model == self.final_model['model_type'] else f'C{i}'
            alpha = 1.0 if model == self.final_model['model_type'] else 0.7
            
            ax1.bar(x + i*width, values, width, label=model.replace('_', ' ').title(),
                   color=color, alpha=alpha)
        
        ax1.set_xlabel('Performance Metrics', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Primary Performance Metrics', fontweight='bold', fontsize=14)
        ax1.set_xticks(x + width * 2.5)
        ax1.set_xticklabels(metric_labels)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Model Rankings
        rankings = self.ranking_data['ranking']
        y_pos = np.arange(len(rankings))
        
        # Get F1 scores for each model
        f1_scores = [self.get_model_metric(model, 'mean_f1') for model in rankings]
        
        colors = ['red' if model == self.final_model['model_type'] else 'skyblue' 
                 for model in rankings]
        
        bars = ax2.barh(y_pos, f1_scores, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, f1_scores)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([model.replace('_', ' ').title() for model in rankings])
        ax2.set_xlabel('F1 Score', fontweight='bold')
        ax2.set_title('Model Ranking by F1 Score', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('ESP32 Shadow: Model Performance Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_model_stability_analysis(self):
        """Create visualization showing model stability (std deviations)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        models = list(self.metrics_data.keys())
        
        # 1. Standard Deviation Comparison
        std_metrics = ['std_f1', 'std_balanced_accuracy', 'std_precision', 'std_recall']
        std_labels = ['F1 Std', 'Balanced Acc Std', 'Precision Std', 'Recall Std']
        
        std_data = []
        for model in models:
            model_stds = [self.get_model_metric(model, metric) for metric in std_metrics]
            std_data.append(model_stds)
        
        x = np.arange(len(models))
        width = 0.2
        
        for i, (metric, label) in enumerate(zip(std_metrics, std_labels)):
            values = [self.get_model_metric(model, metric) for model in models]
            ax1.bar(x + i*width, values, width, label=label, alpha=0.8)
        
        ax1.set_xlabel('Models', fontweight='bold')
        ax1.set_ylabel('Standard Deviation', fontweight='bold')
        ax1.set_title('Model Stability Analysis (Lower = More Stable)', 
                     fontweight='bold', fontsize=14)
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels([model.replace('_', ' ').title() for model in models], 
                           rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Mean vs Std Scatter Plot
        mean_f1 = [self.get_model_metric(model, 'mean_f1') for model in models]
        std_f1 = [self.get_model_metric(model, 'std_f1') for model in models]
        
        colors = ['red' if model == self.final_model['model_type'] else 'blue' 
                 for model in models]
        sizes = [150 if model == self.final_model['model_type'] else 100 
                for model in models]
        
        scatter = ax2.scatter(mean_f1, std_f1, c=colors, s=sizes, alpha=0.7, edgecolors='black')
        
        # Add model name labels
        for i, model in enumerate(models):
            ax2.annotate(model.replace('_', ' ').title(), 
                        (mean_f1[i], std_f1[i]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, ha='left')
        
        ax2.set_xlabel('Mean F1 Score', fontweight='bold')
        ax2.set_ylabel('F1 Standard Deviation', fontweight='bold')
        ax2.set_title('Performance vs Stability Trade-off', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add quadrant labels
        ax2.axhline(y=np.mean(std_f1), color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=np.mean(mean_f1), color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def plot_selected_model_summary(self):
        """Create summary visualization for the selected model"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        selected_model = self.final_model['model_type']
        
        # 1. Key Metrics Gauge Chart
        key_metrics = {
            'F1 Score': self.get_model_metric(selected_model, 'mean_f1'),
            'Balanced Accuracy': self.get_model_metric(selected_model, 'mean_balanced_accuracy'),
            'Precision': self.get_model_metric(selected_model, 'mean_precision'),
            'Recall': self.get_model_metric(selected_model, 'mean_recall'),
            'MCC': self.get_model_metric(selected_model, 'mean_mcc')
        }
        
        metrics_names = list(key_metrics.keys())
        metrics_values = list(key_metrics.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        bars = ax1.barh(metrics_names, metrics_values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, value in zip(bars, metrics_values):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        ax1.set_xlim(0, 1)
        ax1.set_xlabel('Score', fontweight='bold')
        ax1.set_title(f'Selected Model: {selected_model.replace("_", " ").title()}', 
                     fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature Importance (Top 10)
        features = self.final_model.get('features', [])[:10]
        if features:
            feature_names = [f.split('_')[-1] for f in features]  # Extract feature type
            y_pos = np.arange(len(feature_names))
            
            ax2.barh(y_pos, range(len(feature_names), 0, -1), 
                    color='steelblue', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names)
            ax2.set_xlabel('Feature Rank', fontweight='bold')
            ax2.set_title('Top 10 Selected Features', fontweight='bold', fontsize=14)
            ax2.grid(True, alpha=0.3)
        
        # 3. Model Configuration Summary
        config_text = f"""
        Model Type: {selected_model.upper()}
        Calibration: {self.final_model.get('calibrator_type', 'None')}
        Optimal Threshold: {self.final_model.get('optimal_threshold', 'N/A'):.3f}
        Total Features: {len(self.final_model.get('features', []))}
        
        Primary Metric: {self.summary_data['config']['primary_metric'].upper()}
        Secondary Metrics: {', '.join(self.summary_data['config']['secondary_metrics'])}
        """
        
        ax3.text(0.1, 0.5, config_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="lightblue", alpha=0.7))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        ax3.set_title('Model Configuration', fontweight='bold', fontsize=14)
        
        # 4. Performance Distribution
        if selected_model in self.metrics_data:
            model_data = self.metrics_data[selected_model]
            
            # Create distribution info
            metrics_for_dist = ['mean_f1', 'mean_precision', 'mean_recall', 'mean_balanced_accuracy']
            values = [model_data.get(metric, 0) for metric in metrics_for_dist]
            labels = ['F1', 'Precision', 'Recall', 'Bal. Acc']
            
            # Pie chart of metric contributions
            ax4.pie(values, labels=labels, autopct='%1.2f', startangle=90,
                   colors=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99'])
            ax4.set_title('Performance Metric Distribution', fontweight='bold', fontsize=14)
        
        plt.suptitle(f'ESP32 Shadow: Selected Model Analysis - {selected_model.upper()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def save_all_visualizations(self):
        """Generate and save all visualization plots"""
        print(f"🎨 Generating model comparison visualizations in {self.output_dir}")
        
        # 1. Radar Comparison
        print("📊 Creating radar comparison plot...")
        fig1 = self.plot_model_ranking_radar()
        fig1.savefig(self.output_dir / 'model_radar_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # 2. Performance Metrics
        print("📊 Creating performance metrics comparison...")
        fig2 = self.plot_performance_metrics_comparison()
        fig2.savefig(self.output_dir / 'performance_metrics_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # 3. Stability Analysis
        print("📊 Creating stability analysis...")
        fig3 = self.plot_model_stability_analysis()
        fig3.savefig(self.output_dir / 'model_stability_analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # 4. Selected Model Summary
        print("📊 Creating selected model summary...")
        fig4 = self.plot_selected_model_summary()
        fig4.savefig(self.output_dir / 'selected_model_summary.png', 
                    dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        # 5. Generate summary report
        self.generate_summary_report()
        
        print(f"✅ All visualizations saved to: {self.output_dir}")
        
    def generate_summary_report(self):
        """Generate a markdown summary report"""
        report_path = self.output_dir / 'model_comparison_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# ESP32 Shadow: Model Comparison Report\n\n")
            f.write(f"**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Selected Model\n\n")
            f.write(f"**Model Type**: {self.final_model['model_type'].upper()}\n\n")
            
            f.write(f"## Model Ranking\n\n")
            f.write(f"**Primary Metric**: {self.ranking_data['primary_metric'].upper()}\n\n")
            f.write(f"**Ranking Order**:\n")
            for i, model in enumerate(self.ranking_data['ranking'], 1):
                star = " ⭐" if model == self.final_model['model_type'] else ""
                f1_score = self.get_model_metric(model, 'mean_f1')
                f.write(f"{i}. {model.replace('_', ' ').title()}{star} (F1: {f1_score:.3f})\n")
            
            f.write(f"\n## Performance Summary\n\n")
            selected_model = self.final_model['model_type']
            f.write(f"**Selected Model Performance**:\n")
            f.write(f"- F1 Score: {self.get_model_metric(selected_model, 'mean_f1'):.3f} ± {self.get_model_metric(selected_model, 'std_f1'):.3f}\n")
            f.write(f"- Balanced Accuracy: {self.get_model_metric(selected_model, 'mean_balanced_accuracy'):.3f} ± {self.get_model_metric(selected_model, 'std_balanced_accuracy'):.3f}\n")
            f.write(f"- Precision: {self.get_model_metric(selected_model, 'mean_precision'):.3f} ± {self.get_model_metric(selected_model, 'std_precision'):.3f}\n")
            f.write(f"- Recall: {self.get_model_metric(selected_model, 'mean_recall'):.3f} ± {self.get_model_metric(selected_model, 'std_recall'):.3f}\n")
            f.write(f"- MCC: {self.get_model_metric(selected_model, 'mean_mcc'):.3f} ± {self.get_model_metric(selected_model, 'std_mcc'):.3f}\n")
            
            f.write(f"\n## Configuration\n\n")
            f.write(f"- **Calibration**: {self.final_model.get('calibrator_type', 'None')}\n")
            f.write(f"- **Optimal Threshold**: {self.final_model.get('optimal_threshold', 'N/A')}\n")
            f.write(f"- **Feature Count**: {len(self.final_model.get('features', []))}\n")
            
        print(f"✅ Summary report saved to: {report_path}")

def main():
    """Main execution function"""
    # Initialize visualizer
    visualizer = ModelComparisonVisualizer()
    
    # Generate all visualizations
    visualizer.save_all_visualizations()
    
    print("\n🎉 Model comparison visualization complete!")
    print(f"📁 Check the visualizations folder: {visualizer.output_dir}")

if __name__ == "__main__":
    main()
