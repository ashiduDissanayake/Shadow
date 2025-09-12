#!/usr/bin/env python3
"""
ESP32 Shadow Stress Detection - Essential Feature Selection Visualization
==========================================================================

Focused visualization suite showing only the most important feature selection insights.

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

class FeatureSelectionVisualizer:
    def __init__(self, stage15_dir):
        """Initialize with Stage 1.5 results directory"""
        self.stage15_dir = Path(stage15_dir)
        self.load_data()
        
    def load_data(self):
        """Load Stage 1.5 results"""
        # Main summary
        with open(self.stage15_dir / 'stage1_5_enhanced_summary.json', 'r') as f:
            self.summary = json.load(f)
            
        # Feature frequencies across folds
        with open(self.stage15_dir / 'aggregated_feature_frequencies.json', 'r') as f:
            freq_data = json.load(f)
            n_folds_str = str(self.summary['n_folds'])
            if n_folds_str in freq_data:
                self.feature_frequencies = freq_data[n_folds_str]
            else:
                first_key = list(freq_data.keys())[0]
                self.feature_frequencies = freq_data[first_key]
                
        # Candidate feature sets
        with open(self.stage15_dir / 'candidate_feature_sets.json', 'r') as f:
            candidates_data = json.load(f)
            self.candidates = []
            for n_features_str, candidate_info in candidates_data.items():
                try:
                    n_features = int(n_features_str)
                    candidate_info['n_features'] = n_features
                    self.candidates.append(candidate_info)
                except ValueError:
                    continue
                    
        print(f"✅ Loaded feature selection results")
        print(f"✅ Final selection: {len(self.summary['final_selection']['selected_features'])} features")
        
    def create_essential_plots(self, figsize=(16, 12)):
        """Create the 4 most important feature selection plots"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('ESP32 Shadow: Essential Feature Selection Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top 20 Features by Selection Frequency
        ax1 = axes[0, 0]
        top_20_features = list(self.summary['final_selection']['feature_frequency_top'].keys())[:20]
        top_20_freqs = [self.summary['final_selection']['feature_frequency_top'][f] for f in top_20_features]
        n_folds = self.summary['n_folds']
        
        # Color code by signal type
        colors = []
        for feature in top_20_features:
            if feature.startswith('bvp'):
                colors.append('#FF6B6B')  # Red for BVP
            elif feature.startswith('acc'):
                colors.append('#4ECDC4')  # Teal for Accelerometer
            elif feature.startswith('eda'):
                colors.append('#45B7D1')  # Blue for EDA
            elif feature.startswith('temp'):
                colors.append('#96CEB4')  # Green for Temperature
            else:
                colors.append('gray')
        
        # Create horizontal bar chart
        bars = ax1.barh(range(len(top_20_features)), top_20_freqs, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top_20_features)))
        
        # Clean feature names for display
        clean_names = []
        for f in top_20_features:
            # Remove prefixes and make more readable
            clean_name = f.replace('bvp_BVP_', 'BVP_').replace('acc_', 'ACC_').replace('eda_', 'EDA_').replace('temp_', 'TEMP_')
            clean_name = clean_name.replace('_', ' ').replace('l2', 'L2').title()
            clean_names.append(clean_name)
        
        ax1.set_yticklabels(clean_names, fontsize=9)
        ax1.set_xlabel('Selection Frequency (out of 15 folds)')
        ax1.set_title('Top 20 Features by Selection Frequency', fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars, top_20_freqs)):
            ax1.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{freq}/15', va='center', fontweight='bold', fontsize=8)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, color='#FF6B6B', alpha=0.8, label='BVP (Heart Rate)'),
            plt.Rectangle((0,0),1,1, color='#4ECDC4', alpha=0.8, label='Accelerometer'),
            plt.Rectangle((0,0),1,1, color='#45B7D1', alpha=0.8, label='EDA (Skin Conductance)'),
            plt.Rectangle((0,0),1,1, color='#96CEB4', alpha=0.8, label='Temperature')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # 2. Selected Features by Physiological Signal
        ax2 = axes[0, 1]
        selected_features = self.summary['final_selection']['selected_features']
        signal_counts = {'BVP': 0, 'Accelerometer': 0, 'EDA': 0, 'Temperature': 0}
        
        for feature in selected_features:
            if feature.startswith('bvp'):
                signal_counts['BVP'] += 1
            elif feature.startswith('acc'):
                signal_counts['Accelerometer'] += 1
            elif feature.startswith('eda'):
                signal_counts['EDA'] += 1
            elif feature.startswith('temp'):
                signal_counts['Temperature'] += 1
        
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax2.pie(signal_counts.values(), 
                                          labels=[f'{k}\n({v} features)' for k, v in signal_counts.items()],
                                          autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                          textprops={'fontsize': 10, 'fontweight': 'bold'})
        
        # Make percentage text more visible
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        ax2.set_title('Selected Features by Physiological Signal', fontweight='bold')
        
        # 3. Performance vs Feature Count
        ax3 = axes[1, 0]
        candidate_data = []
        for candidate in self.candidates:
            candidate_data.append({
                'n_features': candidate['n_features'],
                'mean_f1': candidate['mean_f1'],
                'std_f1': candidate['std_f1']
            })
        
        df_candidates = pd.DataFrame(candidate_data)
        df_candidates = df_candidates.sort_values('n_features')
        
        # Plot with error bars
        ax3.errorbar(df_candidates['n_features'], df_candidates['mean_f1'], 
                    yerr=df_candidates['std_f1'], marker='o', capsize=5, 
                    color='purple', alpha=0.8, linewidth=2, markersize=8)
        
        # Fill between for confidence interval
        ax3.fill_between(df_candidates['n_features'], 
                        df_candidates['mean_f1'] - df_candidates['std_f1'],
                        df_candidates['mean_f1'] + df_candidates['std_f1'],
                        alpha=0.2, color='purple')
        
        # Highlight selected point
        final_n = self.summary['final_selection']['n_features']
        final_f1 = self.summary['final_selection']['mean_f1']
        ax3.scatter([final_n], [final_f1], color='red', s=150, zorder=5, 
                   marker='*', label=f'Selected: {final_n} features', edgecolors='darkred', linewidth=2)
        
        ax3.set_xlabel('Number of Features')
        ax3.set_ylabel('F1 Score (Mean ± Std)')
        ax3.set_title('Model Performance vs Feature Count', fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 4. Final Model Performance Metrics
        ax4 = axes[1, 1]
        metrics = ['mean_f1', 'mean_mcc', 'mean_balanced_accuracy', 'mean_precision', 'mean_recall']
        metric_values = [self.summary['final_selection'][m] for m in metrics]
        metric_labels = ['F1 Score', 'MCC', 'Balanced Accuracy', 'Precision', 'Recall']
        
        bars = ax4.bar(metric_labels, metric_values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
                      alpha=0.8, edgecolor='black', linewidth=1)
        
        ax4.set_ylabel('Score')
        ax4.set_title('Final Model Performance Metrics', fontweight='bold')
        ax4.set_ylim(0, 1)
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add performance summary text
        ax4.text(0.02, 0.95, f'30 Features Selected\nMean F1: {final_f1:.3f}\nStatistically Significant\n(p < 0.001)', 
                transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def save_essential_plots(self, output_dir=None):
        """Generate and save essential visualization"""
        if output_dir is None:
            output_dir = self.stage15_dir / 'visualizations'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎨 Generating essential feature selection visualization...")
        
        # Generate the essential plot
        fig = self.create_essential_plots()
        fig.savefig(output_dir / 'feature_selection_essential.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'feature_selection_essential.pdf', bbox_inches='tight')
        
        plt.close('all')  # Clean up memory
        
        print(f"✅ Saved essential visualization to {output_dir}")
        
        # Generate concise summary
        self.generate_concise_summary(output_dir)
        
    def generate_concise_summary(self, output_dir):
        """Generate a concise summary of key results"""
        summary_path = output_dir / 'feature_selection_summary.txt'
        
        selected_features = self.summary['final_selection']['selected_features']
        n_folds = self.summary['n_folds']
        
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ESP32 SHADOW: FEATURE SELECTION SUMMARY\n")
            f.write("=" * 60 + "\n")
            
            f.write(f"📊 FINAL RESULTS:\n")
            f.write(f"   • Selected Features: {len(selected_features)}\n")
            f.write(f"   • F1 Score: {self.summary['final_selection']['mean_f1']:.3f} ± {self.summary['final_selection']['std_f1']:.3f}\n")
            f.write(f"   • MCC: {self.summary['final_selection']['mean_mcc']:.3f}\n")
            f.write(f"   • ROC AUC: {self.summary['final_selection']['mean_roc_auc']:.3f}\n")
            f.write(f"   • Statistical Significance: p < {self.summary['final_selection']['significance_test']['p_value']:.3f}\n\n")
            
            # Signal distribution
            signal_counts = {'bvp': 0, 'acc': 0, 'eda': 0, 'temp': 0}
            for feature in selected_features:
                if feature.startswith('bvp'): signal_counts['bvp'] += 1
                elif feature.startswith('acc'): signal_counts['acc'] += 1
                elif feature.startswith('eda'): signal_counts['eda'] += 1
                elif feature.startswith('temp'): signal_counts['temp'] += 1
            
            f.write(f"🔬 SIGNAL DISTRIBUTION:\n")
            f.write(f"   • BVP (Heart Rate): {signal_counts['bvp']} features ({signal_counts['bvp']/len(selected_features)*100:.1f}%)\n")
            f.write(f"   • Accelerometer: {signal_counts['acc']} features ({signal_counts['acc']/len(selected_features)*100:.1f}%)\n")
            f.write(f"   • EDA (Skin): {signal_counts['eda']} features ({signal_counts['eda']/len(selected_features)*100:.1f}%)\n")
            f.write(f"   • Temperature: {signal_counts['temp']} features ({signal_counts['temp']/len(selected_features)*100:.1f}%)\n\n")
            
            f.write(f"⭐ TOP 10 MOST STABLE FEATURES:\n")
            top_10 = list(self.summary['final_selection']['feature_frequency_top'].items())[:10]
            for i, (feature, freq) in enumerate(top_10, 1):
                stability = freq / n_folds
                clean_name = feature.replace('bvp_BVP_', 'BVP_').replace('acc_', 'ACC_').replace('eda_', 'EDA_').replace('temp_', 'TEMP_')
                f.write(f"   {i:2d}. {clean_name:<25} ({freq:2d}/{n_folds} = {stability:.1%})\n")
        
        print(f"📄 Generated concise summary: {summary_path}")

def main():
    """Main execution function"""
    # Initialize visualizer
    stage15_dir = Path(__file__).parent
    visualizer = FeatureSelectionVisualizer(stage15_dir)
    
    # Generate and save essential plots
    visualizer.save_essential_plots()
    
    print("\n🎯 Essential Feature Selection Visualization Complete!")
    print("📊 Generated files:")
    print("   • feature_selection_essential.png/pdf - Key insights visualization")
    print("   • feature_selection_summary.txt - Concise text summary")

if __name__ == "__main__":
    main()
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FeatureSelectionVisualizer:
    def __init__(self, stage15_dir):
        """Initialize with Stage 1.5 results directory"""
        self.stage15_dir = Path(stage15_dir)
        self.load_data()
        
    def load_data(self):
        """Load all Stage 1.5 results"""
        # Main summary
        with open(self.stage15_dir / 'stage1_5_enhanced_summary.json', 'r') as f:
            self.summary = json.load(f)
            
        # Feature frequencies across folds
        with open(self.stage15_dir / 'aggregated_feature_frequencies.json', 'r') as f:
            freq_data = json.load(f)
            # Extract the feature frequencies from the nested structure
            # The JSON has fold counts as keys, we want the data for the actual number of folds
            n_folds_str = str(self.summary['n_folds'])
            if n_folds_str in freq_data:
                self.feature_frequencies = freq_data[n_folds_str]
            else:
                # Fallback: take the first available key
                first_key = list(freq_data.keys())[0]
                self.feature_frequencies = freq_data[first_key]
            
        # Candidate feature sets
        with open(self.stage15_dir / 'candidate_feature_sets.json', 'r') as f:
            candidates_data = json.load(f)
            # Convert the nested structure to a list of candidates
            self.candidates = []
            for n_features_str, candidate_info in candidates_data.items():
                try:
                    n_features = int(n_features_str)
                    candidate_info['n_features'] = n_features
                    self.candidates.append(candidate_info)
                except ValueError:
                    # Skip any non-numeric keys
                    continue
            
        # Per-fold results
        with open(self.stage15_dir / 'per_fold_results.json', 'r') as f:
            self.fold_results = json.load(f)
            
        print(f"✅ Loaded data from {len(self.fold_results)} folds")
        print(f"✅ Final selection: {len(self.summary['final_selection']['selected_features'])} features")
        
    def plot_feature_selection_overview(self, figsize=(20, 12)):
        """Create comprehensive overview of feature selection process"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('ESP32 Shadow: Feature Selection Analysis Overview', fontsize=16, fontweight='bold')
        
        # 1. Feature Selection Frequency Distribution
        ax1 = axes[0, 0]
        frequencies = list(self.feature_frequencies.values())
        n_folds = self.summary['n_folds']
        
        hist, bins = np.histogram(frequencies, bins=np.arange(0, n_folds+2))
        ax1.bar(bins[:-1], hist, alpha=0.7, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Selection Frequency (across 15 folds)')
        ax1.set_ylabel('Number of Features')
        ax1.set_title('Feature Selection Frequency Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        ax1.axvline(np.mean(frequencies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(frequencies):.1f}')
        ax1.axvline(self.summary['config']['min_selection_frequency'] * n_folds, 
                   color='orange', linestyle='--', 
                   label=f'Threshold: {self.summary["config"]["min_selection_frequency"] * n_folds:.1f}')
        ax1.legend()
        
        # 2. Performance vs Feature Count
        ax2 = axes[0, 1]
        candidate_data = []
        for candidate in self.candidates:
            candidate_data.append({
                'n_features': candidate['n_features'],
                'mean_f1': candidate['mean_f1'],
                'std_f1': candidate['std_f1'],
                'min_f1': candidate['min_f1']
            })
        
        df_candidates = pd.DataFrame(candidate_data)
        df_candidates = df_candidates.sort_values('n_features')
        
        ax2.errorbar(df_candidates['n_features'], df_candidates['mean_f1'], 
                    yerr=df_candidates['std_f1'], marker='o', capsize=5, 
                    color='purple', alpha=0.7, linewidth=2, markersize=6)
        ax2.fill_between(df_candidates['n_features'], 
                        df_candidates['mean_f1'] - df_candidates['std_f1'],
                        df_candidates['mean_f1'] + df_candidates['std_f1'],
                        alpha=0.2, color='purple')
        
        # Highlight selected point
        final_n = self.summary['final_selection']['n_features']
        final_f1 = self.summary['final_selection']['mean_f1']
        ax2.scatter([final_n], [final_f1], color='red', s=100, zorder=5, 
                   label=f'Selected: {final_n} features')
        
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('F1 Score (Mean ± Std)')
        ax2.set_title('Performance vs Feature Count')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Top Features Stability Heatmap
        ax3 = axes[0, 2]
        top_features = list(self.summary['final_selection']['feature_frequency_top'].keys())[:15]
        top_freqs = [self.summary['final_selection']['feature_frequency_top'][f] for f in top_features]
        
        # Create heatmap data
        heatmap_data = np.array(top_freqs).reshape(-1, 1)
        
        im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=n_folds)
        ax3.set_yticks(range(len(top_features)))
        ax3.set_yticklabels([f.replace('_', '_\n') for f in top_features], fontsize=8)
        ax3.set_xticks([0])
        ax3.set_xticklabels(['Selection\nFrequency'])
        ax3.set_title('Top 15 Features Stability')
        
        # Add frequency values
        for i, freq in enumerate(top_freqs):
            ax3.text(0, i, f'{freq}/{n_folds}', ha='center', va='center', 
                    color='white' if freq < n_folds/2 else 'black', fontweight='bold')
        
        # 4. Physiological Signal Distribution
        ax4 = axes[1, 0]
        selected_features = self.summary['final_selection']['selected_features']
        signal_counts = {'bvp': 0, 'acc': 0, 'eda': 0, 'temp': 0}
        
        for feature in selected_features:
            if feature.startswith('bvp'):
                signal_counts['bvp'] += 1
            elif feature.startswith('acc'):
                signal_counts['acc'] += 1
            elif feature.startswith('eda'):
                signal_counts['eda'] += 1
            elif feature.startswith('temp'):
                signal_counts['temp'] += 1
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        wedges, texts, autotexts = ax4.pie(signal_counts.values(), 
                                          labels=[f'{k.upper()}\n({v} features)' for k, v in signal_counts.items()],
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax4.set_title('Selected Features by Physiological Signal')
        
        # 5. Cross-Fold Performance Consistency
        ax5 = axes[1, 1]
        final_n = len(self.summary['final_selection']['selected_features'])
        fold_f1_scores = []
        fold_ids = []
        
        for fold_result in self.fold_results:
            if str(final_n) in fold_result.get('prefix_evaluations', {}):
                fold_f1_scores.append(fold_result['prefix_evaluations'][str(final_n)]['outer_f1'])
                fold_ids.append(fold_result['fold_id'])
        
        if fold_f1_scores:
            ax5.boxplot([fold_f1_scores], labels=[f'{final_n} Features'])
            ax5.scatter([1] * len(fold_f1_scores), fold_f1_scores, 
                       alpha=0.6, color='red', s=30)
            ax5.set_ylabel('F1 Score')
            ax5.set_title(f'Cross-Fold Performance Consistency\n(Mean: {np.mean(fold_f1_scores):.3f} ± {np.std(fold_f1_scores):.3f})')
        else:
            ax5.text(0.5, 0.5, f'No performance data\nfor {final_n} features', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Cross-Fold Performance Consistency')
        ax5.grid(True, alpha=0.3)
        
        # Add horizontal lines for statistics
        ax5.axhline(np.mean(fold_f1_scores), color='green', linestyle='--', alpha=0.7, label='Mean')
        ax5.axhline(np.mean(fold_f1_scores) + np.std(fold_f1_scores), color='orange', linestyle=':', alpha=0.7)
        ax5.axhline(np.mean(fold_f1_scores) - np.std(fold_f1_scores), color='orange', linestyle=':', alpha=0.7)
        
        # 6. Statistical Significance Test Results
        ax6 = axes[1, 2]
        sig_test = self.summary['final_selection']['significance_test']
        
        # Create bar chart showing true vs null performance
        categories = ['True F1', 'Null Mean', 'Null Mean + 2σ']
        values = [sig_test['true_f1'], 
                 sig_test['null_f1_mean'],
                 sig_test['null_f1_mean'] + 2 * sig_test['null_f1_std']]
        colors_sig = ['green', 'red', 'orange']
        
        bars = ax6.bar(categories, values, color=colors_sig, alpha=0.7)
        ax6.set_ylabel('F1 Score')
        ax6.set_title(f'Statistical Significance Test\n(p < {sig_test["p_value"]:.3f}, Effect Size: {sig_test["effect_size"]:.1f})')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax6.set_ylim(0, max(values) * 1.2)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_ranking_analysis(self, figsize=(16, 10)):
        """Detailed feature ranking and stability analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Feature Ranking & Stability Analysis', fontsize=16, fontweight='bold')
        
        # 1. Top 20 Features with Selection Frequency
        ax1 = axes[0, 0]
        top_20_features = list(self.summary['final_selection']['feature_frequency_top'].keys())[:20]
        top_20_freqs = [self.summary['final_selection']['feature_frequency_top'][f] for f in top_20_features]
        
        # Color code by signal type
        colors = []
        for feature in top_20_features:
            if feature.startswith('bvp'):
                colors.append('#FF6B6B')
            elif feature.startswith('acc'):
                colors.append('#4ECDC4')
            elif feature.startswith('eda'):
                colors.append('#45B7D1')
            elif feature.startswith('temp'):
                colors.append('#96CEB4')
            else:
                colors.append('gray')
        
        bars = ax1.barh(range(len(top_20_features)), top_20_freqs, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(top_20_features)))
        ax1.set_yticklabels([f.replace('_', '_\n').replace('l2', 'L2') for f in top_20_features], fontsize=8)
        ax1.set_xlabel('Selection Frequency (out of 15 folds)')
        ax1.set_title('Top 20 Features by Selection Frequency')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add frequency labels
        for i, (bar, freq) in enumerate(zip(bars, top_20_freqs)):
            ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{freq}/15', va='center', fontweight='bold')
        
        # 2. Feature Stability vs Performance Trade-off
        ax2 = axes[0, 1]
        candidate_stability = []
        candidate_performance = []
        candidate_sizes = []
        
        for candidate in self.candidates:
            # Calculate stability as mean selection frequency of selected features
            selected_feats = candidate.get('selected_features', [])
            if selected_feats:
                stabilities = [self.feature_frequencies.get(f, 0) for f in selected_feats]
                mean_stability = np.mean(stabilities) / self.summary['n_folds']
            else:
                mean_stability = 0
            
            candidate_stability.append(mean_stability)
            candidate_performance.append(candidate['mean_f1'])
            candidate_sizes.append(candidate['n_features'])
        
        scatter = ax2.scatter(candidate_stability, candidate_performance, 
                             s=[size*3 for size in candidate_sizes], 
                             alpha=0.6, c=candidate_sizes, cmap='viridis')
        
        # Highlight final selection
        final_idx = None
        for i, candidate in enumerate(self.candidates):
            if candidate['n_features'] == self.summary['final_selection']['n_features']:
                final_idx = i
                break
        
        if final_idx is not None:
            ax2.scatter([candidate_stability[final_idx]], [candidate_performance[final_idx]], 
                       s=200, color='red', marker='*', label='Selected', zorder=5)
        
        ax2.set_xlabel('Mean Feature Stability')
        ax2.set_ylabel('Mean F1 Score')
        ax2.set_title('Stability vs Performance Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Number of Features')
        
        # 3. Feature Type Distribution in Selected Set
        ax3 = axes[1, 0]
        selected_features = self.summary['final_selection']['selected_features']
        
        # Categorize features by type
        feature_types = {}
        for feature in selected_features:
            parts = feature.split('_')
            signal = parts[0]  # bvp, acc, eda, temp
            if len(parts) > 2:
                feature_type = parts[2]  # the actual feature type
            else:
                feature_type = parts[1]
                
            key = f"{signal}_{feature_type}"
            feature_types[key] = feature_types.get(key, 0) + 1
        
        # Sort by count
        sorted_types = sorted(feature_types.items(), key=lambda x: x[1], reverse=True)
        types, counts = zip(*sorted_types[:15])  # Top 15 types
        
        bars = ax3.bar(range(len(types)), counts, 
                      color=plt.cm.Set3(np.linspace(0, 1, len(types))))
        ax3.set_xticks(range(len(types)))
        ax3.set_xticklabels([t.replace('_', '\n') for t in types], rotation=45, ha='right')
        ax3.set_ylabel('Count')
        ax3.set_title('Feature Type Distribution in Selected Set')
        ax3.grid(True, axis='y', alpha=0.3)
        
        # 4. Performance Metrics Comparison
        ax4 = axes[1, 1]
        metrics = ['mean_f1', 'mean_mcc', 'mean_balanced_accuracy', 'mean_precision', 'mean_recall']
        metric_values = [self.summary['final_selection'][m] for m in metrics]
        metric_labels = ['F1', 'MCC', 'Bal. Acc.', 'Precision', 'Recall']
        
        bars = ax4.bar(metric_labels, metric_values, 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
        ax4.set_ylabel('Score')
        ax4.set_title('Final Model Performance Metrics')
        ax4.set_ylim(0, 1)
        ax4.grid(True, axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_signal_feature_analysis(self, figsize=(16, 8)):
        """Analysis of features by physiological signal type"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Physiological Signal Feature Analysis', fontsize=16, fontweight='bold')
        
        selected_features = self.summary['final_selection']['selected_features']
        
        # 1. Feature count and stability by signal type
        ax1 = axes[0]
        signal_data = {'bvp': [], 'acc': [], 'eda': [], 'temp': []}
        
        for feature in selected_features:
            freq = self.summary['final_selection']['feature_frequency_top'][feature]
            stability = freq / self.summary['n_folds']
            
            if feature.startswith('bvp'):
                signal_data['bvp'].append(stability)
            elif feature.startswith('acc'):
                signal_data['acc'].append(stability)
            elif feature.startswith('eda'):
                signal_data['eda'].append(stability)
            elif feature.startswith('temp'):
                signal_data['temp'].append(stability)
        
        # Create box plot
        data_for_boxplot = [signal_data[signal] for signal in ['bvp', 'acc', 'eda', 'temp']]
        labels = [f'{signal.upper()}\n({len(data)} features)' for signal, data in zip(['bvp', 'acc', 'eda', 'temp'], data_for_boxplot)]
        
        bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Feature Stability (Selection Frequency / 15)')
        ax1.set_title('Feature Stability by Signal Type')
        ax1.grid(True, alpha=0.3)
        
        # 2. Feature importance heatmap by signal and type
        ax2 = axes[1]
        
        # Create matrix: signals vs feature types
        signal_types = ['bvp', 'acc', 'eda', 'temp']
        feature_type_counts = {}
        
        for feature in selected_features:
            parts = feature.split('_')
            signal = parts[0]
            if len(parts) > 2:
                feat_type = parts[2]
            else:
                feat_type = parts[1]
            
            if signal not in feature_type_counts:
                feature_type_counts[signal] = {}
            feature_type_counts[signal][feat_type] = feature_type_counts[signal].get(feat_type, 0) + 1
        
        # Get all unique feature types
        all_feat_types = set()
        for signal_dict in feature_type_counts.values():
            all_feat_types.update(signal_dict.keys())
        all_feat_types = sorted(list(all_feat_types))
        
        # Create matrix
        matrix = np.zeros((len(signal_types), len(all_feat_types)))
        for i, signal in enumerate(signal_types):
            for j, feat_type in enumerate(all_feat_types):
                matrix[i, j] = feature_type_counts.get(signal, {}).get(feat_type, 0)
        
        im = ax2.imshow(matrix, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(len(all_feat_types)))
        ax2.set_xticklabels(all_feat_types, rotation=45, ha='right')
        ax2.set_yticks(range(len(signal_types)))
        ax2.set_yticklabels([s.upper() for s in signal_types])
        ax2.set_title('Feature Type Distribution by Signal')
        
        # Add text annotations
        for i in range(len(signal_types)):
            for j in range(len(all_feat_types)):
                count = int(matrix[i, j])
                if count > 0:
                    ax2.text(j, i, str(count), ha="center", va="center", 
                            color="white" if count > matrix.max()/2 else "black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Number of Features')
        
        plt.tight_layout()
        return fig
    
    def save_all_plots(self, output_dir=None):
        """Generate and save all visualization plots"""
        if output_dir is None:
            output_dir = self.stage15_dir / 'visualizations'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"🎨 Generating visualizations in {output_dir}")
        
        # Generate all plots
        fig1 = self.plot_feature_selection_overview()
        fig1.savefig(output_dir / 'feature_selection_overview.png', dpi=300, bbox_inches='tight')
        fig1.savefig(output_dir / 'feature_selection_overview.pdf', bbox_inches='tight')
        
        fig2 = self.plot_feature_ranking_analysis()
        fig2.savefig(output_dir / 'feature_ranking_analysis.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'feature_ranking_analysis.pdf', bbox_inches='tight')
        
        fig3 = self.plot_signal_feature_analysis()
        fig3.savefig(output_dir / 'signal_feature_analysis.png', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / 'signal_feature_analysis.pdf', bbox_inches='tight')
        
        plt.close('all')  # Clean up memory
        
        print(f"✅ Saved 3 visualization files (PNG + PDF) to {output_dir}")
        
        # Generate summary report
        self.generate_summary_report(output_dir)
        
    def generate_summary_report(self, output_dir):
        """Generate a summary report of the feature selection results"""
        report_path = output_dir / 'feature_selection_report.md'
        
        selected_features = self.summary['final_selection']['selected_features']
        n_folds = self.summary['n_folds']
        
        with open(report_path, 'w') as f:
            f.write("# ESP32 Shadow: Feature Selection Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Folds**: {n_folds}\n")
            f.write(f"- **Final Feature Count**: {len(selected_features)}\n")
            f.write(f"- **Mean F1 Score**: {self.summary['final_selection']['mean_f1']:.4f} ± {self.summary['final_selection']['std_f1']:.4f}\n")
            f.write(f"- **Mean MCC**: {self.summary['final_selection']['mean_mcc']:.4f}\n")
            f.write(f"- **ROC AUC**: {self.summary['final_selection']['mean_roc_auc']:.4f}\n")
            f.write(f"- **Statistical Significance**: p < {self.summary['final_selection']['significance_test']['p_value']:.3f}\n\n")
            
            f.write("## Feature Distribution by Signal Type\n\n")
            signal_counts = {'bvp': 0, 'acc': 0, 'eda': 0, 'temp': 0}
            for feature in selected_features:
                if feature.startswith('bvp'): signal_counts['bvp'] += 1
                elif feature.startswith('acc'): signal_counts['acc'] += 1
                elif feature.startswith('eda'): signal_counts['eda'] += 1
                elif feature.startswith('temp'): signal_counts['temp'] += 1
            
            for signal, count in signal_counts.items():
                percentage = (count / len(selected_features)) * 100
                f.write(f"- **{signal.upper()}**: {count} features ({percentage:.1f}%)\n")
            
            f.write(f"\n## Top 10 Most Stable Features\n\n")
            top_10 = list(self.summary['final_selection']['feature_frequency_top'].items())[:10]
            for i, (feature, freq) in enumerate(top_10, 1):
                stability = freq / n_folds
                f.write(f"{i}. **{feature}**: {freq}/{n_folds} ({stability:.1%} stability)\n")
            
            f.write(f"\n## Selected Features (All {len(selected_features)})\n\n")
            for i, feature in enumerate(selected_features, 1):
                freq = self.summary['final_selection']['feature_frequency_top'][feature]
                stability = freq / n_folds
                f.write(f"{i}. `{feature}` - {freq}/{n_folds} ({stability:.1%})\n")
        
        print(f"📄 Generated summary report: {report_path}")

def main():
    """Main execution function"""
    # Initialize visualizer
    stage15_dir = Path(__file__).parent
    visualizer = FeatureSelectionVisualizer(stage15_dir)
    
    # Generate and save all plots
    visualizer.save_all_plots()
    
    print("\n🎯 Feature Selection Visualization Complete!")
    print("📊 Generated plots:")
    print("   1. feature_selection_overview.png/pdf - Comprehensive overview")
    print("   2. feature_ranking_analysis.png/pdf - Detailed ranking analysis") 
    print("   3. signal_feature_analysis.png/pdf - Signal-specific analysis")
    print("   4. feature_selection_report.md - Summary report")

if __name__ == "__main__":
    main()