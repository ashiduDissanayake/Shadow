"""
Dataset Plotting Module

Provides comprehensive dataset-level visualization including cross-subject
comparisons, dataset-wide statistics, and quality metrics.

Features:
- Full dataset label distributions
- Cross-subject comparisons
- Dataset-wide quality statistics
- Subject performance comparisons
- Comprehensive dataset overview

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class DatasetPlotter:
    """
    Dataset plotting class for comprehensive dataset analysis visualization.
    
    Provides visualization capabilities for full dataset analysis including
    cross-subject comparisons and dataset-wide statistics.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the dataset plotter.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup plotting parameters
        plt.style.use(config.visualization.style)
        self.figure_size = config.visualization.figure_size
        self.dpi = config.visualization.dpi
        self.condition_colors = config.visualization.condition_colors
        
        # Create output directory
        if config.visualization.save_plots:
            self.output_dir = Path(config.output.output_path) / config.output.plots_dir / "dataset"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        self.logger.info(f"Dataset plotter initialized (output: {self.output_dir})")
    
    def plot_dataset_overview(self, dataset_results: Dict[int, Dict],
                            title: str = "WESAD Dataset Overview",
                            save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive dataset overview plot.
        
        Args:
            dataset_results: Dictionary mapping subject IDs to their analysis results
            title: Plot title
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not dataset_results:
                self.logger.warning("No dataset results provided")
                return plt.figure()
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, 
                                                        figsize=(self.figure_size[0] * 1.5, 
                                                               self.figure_size[1] * 1.2), 
                                                        dpi=self.dpi)
            
            # Extract dataset statistics
            dataset_stats = self._extract_dataset_statistics(dataset_results)
            
            # 1. Subject data availability
            subject_ids = list(dataset_results.keys())
            data_durations = [stats.get('signal_duration', 0) for stats in dataset_stats.values()]
            
            bars1 = ax1.bar(range(len(subject_ids)), data_durations, 
                          color='lightblue', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Subject ID', fontsize=12)
            ax1.set_ylabel('Signal Duration (seconds)', fontsize=12)
            ax1.set_title('Data Duration by Subject', fontsize=12, fontweight='bold')
            ax1.set_xticks(range(len(subject_ids)))
            ax1.set_xticklabels([f'S{sid}' for sid in subject_ids], rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add duration labels on bars
            for bar, duration in zip(bars1, data_durations):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{duration:.0f}s', ha='center', va='bottom', fontsize=9)
            
            # 2. Dataset label distribution
            all_labels = []
            for subject_results in dataset_results.values():
                if 'windowing_result' in subject_results:
                    windows = subject_results['windowing_result'].get('windows', [])
                    all_labels.extend([w['label'] for w in windows])
            
            if all_labels:
                label_counts = Counter(all_labels)
                condition_names = [self.config.get_label_name(label_id) for label_id in label_counts.keys()]
                condition_counts = list(label_counts.values())
                condition_colors_list = [self.condition_colors.get(name, '#808080') for name in condition_names]
                
                ax2.pie(condition_counts, labels=condition_names, colors=condition_colors_list,
                       autopct='%1.1f%%', startangle=90)
                ax2.set_title('Dataset Label Distribution', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No windowing data available', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Dataset Label Distribution', fontsize=12, fontweight='bold')
            
            # 3. Quality scores by subject
            quality_data = []
            quality_subjects = []
            for subject_id, results in dataset_results.items():
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    if windows:
                        subject_qualities = [w['quality'] for w in windows]
                        quality_data.extend(subject_qualities)
                        quality_subjects.extend([f'S{subject_id}'] * len(subject_qualities))
            
            if quality_data:
                # Create violin plot for quality distribution
                quality_df = pd.DataFrame({'Subject': quality_subjects, 'Quality': quality_data})
                sns.violinplot(data=quality_df, x='Subject', y='Quality', ax=ax3, 
                             palette='viridis', alpha=0.7)
                ax3.axhline(y=self.config.analysis.min_window_quality, color='red', 
                           linestyle='--', alpha=0.7, label='Quality Threshold')
                ax3.set_xlabel('Subject', fontsize=12)
                ax3.set_ylabel('Window Quality Score', fontsize=12)
                ax3.set_title('Quality Distribution by Subject', fontsize=12, fontweight='bold')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.get_xticklabels(), rotation=45)
            else:
                ax3.text(0.5, 0.5, 'No quality data available', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Quality Distribution by Subject', fontsize=12, fontweight='bold')
            
            # 4. Heart rate analysis by subject
            hr_data = []
            hr_subjects = []
            for subject_id, results in dataset_results.items():
                if 'heart_rate_result' in results:
                    hr_result = results['heart_rate_result']
                    if hr_result.get('valid_estimates', 0) > 0:
                        mean_hr = hr_result.get('mean_hr', 0)
                        if mean_hr > 0:
                            hr_data.append(mean_hr)
                            hr_subjects.append(f'S{subject_id}')
            
            if hr_data:
                bars4 = ax4.bar(range(len(hr_subjects)), hr_data, 
                              color='lightcoral', alpha=0.7, edgecolor='black')
                ax4.axhspan(60, 100, alpha=0.2, color='green', label='Normal Resting Range')
                ax4.set_xlabel('Subject', fontsize=12)
                ax4.set_ylabel('Mean Heart Rate (BPM)', fontsize=12)
                ax4.set_title('Mean Heart Rate by Subject', fontsize=12, fontweight='bold')
                ax4.set_xticks(range(len(hr_subjects)))
                ax4.set_xticklabels(hr_subjects, rotation=45)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                # Add HR labels on bars
                for bar, hr in zip(bars4, hr_data):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{hr:.0f}', ha='center', va='bottom', fontsize=9)
            else:
                ax4.text(0.5, 0.5, 'No heart rate data available', 
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('Mean Heart Rate by Subject', fontsize=12, fontweight='bold')
            
            plt.suptitle(title, fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved dataset overview plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Dataset overview plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_subject_comparison(self, dataset_results: Dict[int, Dict],
                              metric: str = 'quality',
                              title: Optional[str] = None,
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Create subject comparison plot for specific metrics.
        
        Args:
            dataset_results: Dictionary mapping subject IDs to their analysis results
            metric: Metric to compare ('quality', 'heart_rate', 'confidence', 'windows_count')
            title: Plot title
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not dataset_results:
                self.logger.warning("No dataset results provided")
                return plt.figure()
            
            # Extract comparison data
            comparison_data = self._extract_comparison_data(dataset_results, metric)
            
            if not comparison_data:
                self.logger.warning(f"No data available for metric: {metric}")
                return plt.figure()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            subject_ids = list(comparison_data.keys())
            values = list(comparison_data.values())
            
            # Create bar plot
            bars = ax.bar(range(len(subject_ids)), values, 
                         color='skyblue', alpha=0.7, edgecolor='black')
            
            # Customize based on metric
            if metric == 'quality':
                ax.set_ylabel('Average Quality Score', fontsize=12)
                ax.axhline(y=self.config.analysis.min_window_quality, color='red', 
                          linestyle='--', alpha=0.7, label='Quality Threshold')
                ax.set_ylim(0, 1)
                default_title = "Quality Score Comparison Across Subjects"
            elif metric == 'heart_rate':
                ax.set_ylabel('Average Heart Rate (BPM)', fontsize=12)
                ax.axhspan(60, 100, alpha=0.2, color='green', label='Normal Resting Range')
                default_title = "Heart Rate Comparison Across Subjects"
            elif metric == 'confidence':
                ax.set_ylabel('Average Label Confidence', fontsize=12)
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                          label='50% Confidence')
                ax.set_ylim(0, 1)
                default_title = "Label Confidence Comparison Across Subjects"
            elif metric == 'windows_count':
                ax.set_ylabel('Number of Valid Windows', fontsize=12)
                default_title = "Window Count Comparison Across Subjects"
            else:
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
                default_title = f"{metric.replace('_', ' ').title()} Comparison Across Subjects"
            
            ax.set_xlabel('Subject ID', fontsize=12)
            ax.set_title(title or default_title, fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(subject_ids)))
            ax.set_xticklabels([f'S{sid}' for sid in subject_ids], rotation=45)
            ax.grid(True, alpha=0.3)
            
            if 'label' in ax.get_legend_handles_labels()[1]:
                ax.legend()
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if metric in ['quality', 'confidence']:
                    label_text = f'{value:.3f}'
                elif metric == 'heart_rate':
                    label_text = f'{value:.1f}'
                else:
                    label_text = f'{value:.0f}'
                
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       label_text, ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved subject comparison plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Subject comparison plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_condition_analysis(self, dataset_results: Dict[int, Dict],
                              title: str = "Condition Analysis Across Dataset",
                              save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive condition analysis plot.
        
        Args:
            dataset_results: Dictionary mapping subject IDs to their analysis results
            title: Plot title
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if not dataset_results:
                self.logger.warning("No dataset results provided")
                return plt.figure()
            
            # Extract condition data
            condition_data = self._extract_condition_data(dataset_results)
            
            if not condition_data:
                self.logger.warning("No condition data available")
                return plt.figure()
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
            
            conditions = list(condition_data.keys())
            condition_colors_list = [self.condition_colors.get(cond, '#808080') for cond in conditions]
            
            # 1. Window counts by condition
            window_counts = [condition_data[cond]['window_count'] for cond in conditions]
            bars1 = ax1.bar(conditions, window_counts, color=condition_colors_list, alpha=0.7)
            ax1.set_xlabel('Condition', fontsize=11)
            ax1.set_ylabel('Number of Windows', fontsize=11)
            ax1.set_title('Window Count by Condition', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars1, window_counts):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=9)
            
            # 2. Average quality by condition
            avg_qualities = [condition_data[cond]['avg_quality'] for cond in conditions]
            bars2 = ax2.bar(conditions, avg_qualities, color=condition_colors_list, alpha=0.7)
            ax2.axhline(y=self.config.analysis.min_window_quality, color='red', 
                       linestyle='--', alpha=0.7, label='Quality Threshold')
            ax2.set_xlabel('Condition', fontsize=11)
            ax2.set_ylabel('Average Quality Score', fontsize=11)
            ax2.set_title('Quality Score by Condition', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add quality labels
            for bar, quality in zip(bars2, avg_qualities):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{quality:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 3. Average confidence by condition
            avg_confidences = [condition_data[cond]['avg_confidence'] for cond in conditions]
            bars3 = ax3.bar(conditions, avg_confidences, color=condition_colors_list, alpha=0.7)
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                       label='50% Confidence')
            ax3.set_xlabel('Condition', fontsize=11)
            ax3.set_ylabel('Average Label Confidence', fontsize=11)
            ax3.set_title('Label Confidence by Condition', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add confidence labels
            for bar, confidence in zip(bars3, avg_confidences):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{confidence:.3f}', ha='center', va='bottom', fontsize=9)
            
            # 4. Subject participation by condition
            subject_counts = [condition_data[cond]['subject_count'] for cond in conditions]
            bars4 = ax4.bar(conditions, subject_counts, color=condition_colors_list, alpha=0.7)
            ax4.set_xlabel('Condition', fontsize=11)
            ax4.set_ylabel('Number of Subjects', fontsize=11)
            ax4.set_title('Subject Participation by Condition', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add subject count labels
            for bar, count in zip(bars4, subject_counts):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count}', ha='center', va='bottom', fontsize=9)
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved condition analysis plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Condition analysis plotting failed: {str(e)}")
            return plt.figure()
    
    def _extract_dataset_statistics(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """Extract statistics from dataset results."""
        stats = {}
        
        for subject_id, results in dataset_results.items():
            subject_stats = {}
            
            # Signal duration
            if 'processed_data' in results:
                timestamps = results['processed_data'].get('timestamps', np.array([]))
                if len(timestamps) > 0:
                    subject_stats['signal_duration'] = timestamps[-1] - timestamps[0]
                else:
                    subject_stats['signal_duration'] = 0
            else:
                subject_stats['signal_duration'] = 0
            
            # Window statistics
            if 'windowing_result' in results:
                windows = results['windowing_result'].get('windows', [])
                subject_stats['window_count'] = len(windows)
                if windows:
                    subject_stats['avg_quality'] = np.mean([w['quality'] for w in windows])
                    subject_stats['avg_confidence'] = np.mean([w['confidence'] for w in windows])
                else:
                    subject_stats['avg_quality'] = 0
                    subject_stats['avg_confidence'] = 0
            else:
                subject_stats['window_count'] = 0
                subject_stats['avg_quality'] = 0
                subject_stats['avg_confidence'] = 0
            
            # Heart rate statistics
            if 'heart_rate_result' in results:
                hr_result = results['heart_rate_result']
                subject_stats['mean_heart_rate'] = hr_result.get('mean_hr', 0)
                subject_stats['valid_hr_estimates'] = hr_result.get('valid_estimates', 0)
            else:
                subject_stats['mean_heart_rate'] = 0
                subject_stats['valid_hr_estimates'] = 0
            
            stats[subject_id] = subject_stats
        
        return stats
    
    def _extract_comparison_data(self, dataset_results: Dict[int, Dict], metric: str) -> Dict[int, float]:
        """Extract comparison data for specific metric."""
        comparison_data = {}
        
        for subject_id, results in dataset_results.items():
            if metric == 'quality':
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    if windows:
                        comparison_data[subject_id] = np.mean([w['quality'] for w in windows])
            
            elif metric == 'heart_rate':
                if 'heart_rate_result' in results:
                    hr_result = results['heart_rate_result']
                    if hr_result.get('valid_estimates', 0) > 0:
                        comparison_data[subject_id] = hr_result.get('mean_hr', 0)
            
            elif metric == 'confidence':
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    if windows:
                        comparison_data[subject_id] = np.mean([w['confidence'] for w in windows])
            
            elif metric == 'windows_count':
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    comparison_data[subject_id] = len(windows)
        
        return comparison_data
    
    def _extract_condition_data(self, dataset_results: Dict[int, Dict]) -> Dict[str, Dict]:
        """Extract condition-specific data across all subjects."""
        condition_data = defaultdict(lambda: {
            'window_count': 0,
            'quality_scores': [],
            'confidence_scores': [],
            'subjects': set()
        })
        
        for subject_id, results in dataset_results.items():
            if 'windowing_result' not in results:
                continue
            
            windows = results['windowing_result'].get('windows', [])
            for window in windows:
                condition_name = self.config.get_label_name(window['label'])
                
                condition_data[condition_name]['window_count'] += 1
                condition_data[condition_name]['quality_scores'].append(window['quality'])
                condition_data[condition_name]['confidence_scores'].append(window['confidence'])
                condition_data[condition_name]['subjects'].add(subject_id)
        
        # Calculate averages
        final_condition_data = {}
        for condition, data in condition_data.items():
            if data['window_count'] > 0:
                final_condition_data[condition] = {
                    'window_count': data['window_count'],
                    'avg_quality': np.mean(data['quality_scores']),
                    'avg_confidence': np.mean(data['confidence_scores']),
                    'subject_count': len(data['subjects'])
                }
        
        return final_condition_data