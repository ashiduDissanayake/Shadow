"""
Window Plotting Module

Provides visualization capabilities for windowing analysis including window
creation process, label distributions, and quality metrics.

Features:
- Visualize window creation process
- Window label distribution plots
- Window quality and confidence histograms
- Window-level analysis plots

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import Counter
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class WindowPlotter:
    """
    Window plotting class for windowing analysis visualization.
    
    Provides comprehensive visualization for window creation, label distributions,
    and quality metrics in the windowing analysis pipeline.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the window plotter.
        
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
            self.output_dir = Path(config.output.output_path) / config.output.plots_dir / "windows"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        self.logger.info(f"Window plotter initialized (output: {self.output_dir})")
    
    def plot_window_creation(self, bvp_signal: np.ndarray, labels: np.ndarray,
                           windows_result: Dict, timestamps: Optional[np.ndarray] = None,
                           max_windows_shown: int = 10,
                           title: str = "Window Creation Process",
                           subject_id: Optional[int] = None,
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualize the window creation process over the signal.
        
        Args:
            bvp_signal: Original BVP signal
            labels: Original labels
            windows_result: Result from window analysis
            timestamps: Optional timestamps
            max_windows_shown: Maximum number of windows to highlight
            title: Plot title
            subject_id: Subject ID for annotation
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if len(bvp_signal) == 0:
                self.logger.warning("Empty BVP signal provided")
                return plt.figure()
            
            # Prepare timestamps
            if timestamps is None:
                timestamps = np.arange(len(bvp_signal)) / self.config.dataset.bvp_sampling_rate
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size, 
                                         dpi=self.dpi, sharex=True, height_ratios=[3, 1])
            
            # Top subplot: BVP signal with window overlays
            self._plot_condition_backgrounds(ax1, timestamps, labels)
            ax1.plot(timestamps, bvp_signal, 'k-', linewidth=0.8, alpha=0.7, label='BVP Signal')
            
            # Overlay windows
            windows = windows_result.get('windows', [])
            window_positions = windows_result.get('window_positions', [])
            
            # Show subset of windows to avoid clutter
            windows_to_show = min(len(windows), max_windows_shown)
            step = max(1, len(windows) // windows_to_show)
            
            for i in range(0, len(windows), step):
                if i >= len(window_positions):
                    break
                    
                start_idx, end_idx = window_positions[i]
                start_time = timestamps[start_idx]
                end_time = timestamps[end_idx]
                
                # Color based on window quality
                window_quality = windows[i]['quality'] if i < len(windows) else 0
                color = self._get_quality_color(window_quality)
                
                # Add window rectangle
                height = ax1.get_ylim()[1] - ax1.get_ylim()[0]
                rect = Rectangle((start_time, ax1.get_ylim()[0]), 
                               end_time - start_time, height,
                               linewidth=1, edgecolor=color, facecolor=color, alpha=0.1)
                ax1.add_patch(rect)
            
            ax1.set_ylabel('BVP Amplitude', fontsize=12)
            ax1.set_title(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Bottom subplot: Window quality timeline
            if windows:
                window_times = [(w['start_time'] + w['end_time']) / 2 for w in windows]
                window_qualities = [w['quality'] for w in windows]
                
                # Create color-coded scatter plot
                colors = [self._get_quality_color(q) for q in window_qualities]
                ax2.scatter(window_times, window_qualities, c=colors, s=20, alpha=0.7)
                
                # Add quality threshold line
                ax2.axhline(y=self.config.analysis.min_window_quality, 
                           color='red', linestyle='--', alpha=0.7, 
                           label=f'Quality Threshold ({self.config.analysis.min_window_quality:.2f})')
            
            ax2.set_ylabel('Window Quality', fontsize=12)
            ax2.set_xlabel('Time (seconds)', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add window statistics annotation
            self._add_window_stats_annotation(ax1, windows_result)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved window creation plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Window creation plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_window_distributions(self, windows_result: Dict,
                                title: str = "Window Distributions",
                                subject_id: Optional[int] = None,
                                save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot window label distributions and quality metrics.
        
        Args:
            windows_result: Result from window analysis
            title: Plot title
            subject_id: Subject ID for annotation
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            windows = windows_result.get('windows', [])
            if not windows:
                self.logger.warning("No windows data provided")
                return plt.figure()
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figure_size, dpi=self.dpi)
            
            # Extract data
            labels = [w['label'] for w in windows]
            qualities = [w['quality'] for w in windows]
            confidences = [w['confidence'] for w in windows]
            
            # 1. Label distribution (pie chart)
            label_counts = Counter(labels)
            condition_names = [self.config.get_label_name(label_id) for label_id in label_counts.keys()]
            condition_counts = list(label_counts.values())
            condition_colors_list = [self.condition_colors.get(name, '#808080') for name in condition_names]
            
            ax1.pie(condition_counts, labels=condition_names, colors=condition_colors_list,
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Window Label Distribution', fontsize=12, fontweight='bold')
            
            # 2. Quality distribution (histogram)
            ax2.hist(qualities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax2.axvline(x=self.config.analysis.min_window_quality, color='red', 
                       linestyle='--', alpha=0.7, label='Quality Threshold')
            ax2.axvline(x=np.mean(qualities), color='green', 
                       linestyle='-', alpha=0.7, label=f'Mean ({np.mean(qualities):.3f})')
            ax2.set_xlabel('Quality Score', fontsize=11)
            ax2.set_ylabel('Number of Windows', fontsize=11)
            ax2.set_title('Window Quality Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Confidence distribution (histogram)
            ax3.hist(confidences, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            ax3.axvline(x=np.mean(confidences), color='green', 
                       linestyle='-', alpha=0.7, label=f'Mean ({np.mean(confidences):.3f})')
            ax3.set_xlabel('Label Confidence', fontsize=11)
            ax3.set_ylabel('Number of Windows', fontsize=11)
            ax3.set_title('Window Label Confidence Distribution', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Quality vs Confidence scatter plot
            colors_scatter = [self.condition_colors.get(self.config.get_label_name(label), '#808080') 
                            for label in labels]
            ax4.scatter(qualities, confidences, c=colors_scatter, alpha=0.6, s=30)
            ax4.set_xlabel('Quality Score', fontsize=11)
            ax4.set_ylabel('Label Confidence', fontsize=11)
            ax4.set_title('Quality vs Confidence', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add threshold lines
            ax4.axvline(x=self.config.analysis.min_window_quality, color='red', 
                       linestyle='--', alpha=0.5)
            ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, 
                       label='50% Confidence')
            ax4.legend()
            
            plt.suptitle(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved window distributions plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Window distributions plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_window_timeline(self, windows_result: Dict,
                           title: str = "Window Timeline Analysis",
                           subject_id: Optional[int] = None,
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot window analysis over time showing labels, quality, and confidence.
        
        Args:
            windows_result: Result from window analysis
            title: Plot title
            subject_id: Subject ID for annotation
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            windows = windows_result.get('windows', [])
            if not windows:
                self.logger.warning("No windows data provided")
                return plt.figure()
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figure_size, 
                                              dpi=self.dpi, sharex=True)
            
            # Extract timeline data
            window_times = [(w['start_time'] + w['end_time']) / 2 for w in windows]
            window_durations = [w['end_time'] - w['start_time'] for w in windows]
            labels = [w['label'] for w in windows]
            qualities = [w['quality'] for w in windows]
            confidences = [w['confidence'] for w in windows]
            
            # 1. Window labels over time
            unique_labels = sorted(set(labels))
            label_colors = [self.condition_colors.get(self.config.get_label_name(label), '#808080') 
                          for label in unique_labels]
            
            for i, label in enumerate(unique_labels):
                label_mask = np.array(labels) == label
                label_times = np.array(window_times)[label_mask]
                label_y = np.full_like(label_times, label)
                
                condition_name = self.config.get_label_name(label)
                color = self.condition_colors.get(condition_name, '#808080')
                ax1.scatter(label_times, label_y, c=color, s=40, alpha=0.7, 
                          label=condition_name.title())
            
            ax1.set_ylabel('Condition Label', fontsize=12)
            ax1.set_title('Window Labels Over Time', fontsize=12, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # 2. Window quality over time
            colors_quality = [self._get_quality_color(q) for q in qualities]
            ax2.scatter(window_times, qualities, c=colors_quality, s=30, alpha=0.7)
            ax2.axhline(y=self.config.analysis.min_window_quality, color='red', 
                       linestyle='--', alpha=0.7, label='Quality Threshold')
            ax2.set_ylabel('Quality Score', fontsize=12)
            ax2.set_title('Window Quality Over Time', fontsize=12, fontweight='bold')
            ax2.set_ylim(0, 1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Window confidence over time
            ax3.scatter(window_times, confidences, c='purple', s=30, alpha=0.7)
            ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, 
                       label='50% Confidence')
            ax3.axhline(y=np.mean(confidences), color='green', linestyle='-', alpha=0.7, 
                       label=f'Mean ({np.mean(confidences):.3f})')
            ax3.set_xlabel('Time (seconds)', fontsize=12)
            ax3.set_ylabel('Label Confidence', fontsize=12)
            ax3.set_title('Window Label Confidence Over Time', fontsize=12, fontweight='bold')
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            plt.suptitle(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved window timeline plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Window timeline plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_window_features(self, features_result: Dict,
                           max_features_shown: int = 16,
                           title: str = "Window Features Analysis",
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot window features analysis including feature distributions and correlations.
        
        Args:
            features_result: Result from window feature extraction
            max_features_shown: Maximum number of features to show
            title: Plot title
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            features = features_result.get('features', np.array([]))
            labels = features_result.get('labels', np.array([]))
            feature_names = features_result.get('feature_names', [])
            
            if features.size == 0:
                self.logger.warning("No features data provided")
                return plt.figure()
            
            # Limit number of features for visualization
            n_features = min(len(feature_names), max_features_shown)
            features_subset = features[:, :n_features]
            feature_names_subset = feature_names[:n_features]
            
            # Create figure
            fig = plt.figure(figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.2), 
                           dpi=self.dpi)
            
            # Calculate grid layout
            n_rows = int(np.ceil(np.sqrt(n_features)))
            n_cols = int(np.ceil(n_features / n_rows))
            
            # Plot feature distributions
            for i, feature_name in enumerate(feature_names_subset):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                
                feature_data = features_subset[:, i]
                
                # Create separate histograms for each condition
                unique_labels = np.unique(labels)
                colors = [self.condition_colors.get(self.config.get_label_name(label), '#808080') 
                         for label in unique_labels]
                
                for j, label in enumerate(unique_labels):
                    label_mask = labels == label
                    label_data = feature_data[label_mask]
                    
                    if len(label_data) > 0:
                        condition_name = self.config.get_label_name(label)
                        ax.hist(label_data, bins=10, alpha=0.6, 
                               color=colors[j], label=condition_name.title())
                
                ax.set_title(feature_name.replace('_', ' ').title(), fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add legend to first subplot
                if i == 0:
                    ax.legend(fontsize=8)
            
            # Hide empty subplots
            for i in range(n_features, n_rows * n_cols):
                ax = plt.subplot(n_rows, n_cols, i + 1)
                ax.set_visible(False)
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved window features plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Window features plotting failed: {str(e)}")
            return plt.figure()
    
    def _plot_condition_backgrounds(self, ax: plt.Axes, timestamps: np.ndarray, 
                                  labels: np.ndarray) -> None:
        """Plot condition backgrounds on axes."""
        if len(timestamps) == 0 or len(labels) == 0:
            return
        
        # Find condition segments
        condition_segments = self._find_condition_segments(timestamps, labels)
        
        for start_time, end_time, label_id in condition_segments:
            condition_name = self.config.get_label_name(label_id)
            color = self.condition_colors.get(condition_name, '#808080')
            
            # Add background rectangle
            rect = patches.Rectangle((start_time, ax.get_ylim()[0]), 
                                   end_time - start_time, 
                                   ax.get_ylim()[1] - ax.get_ylim()[0],
                                   linewidth=0, facecolor=color, alpha=0.2)
            ax.add_patch(rect)
    
    def _find_condition_segments(self, timestamps: np.ndarray, 
                               labels: np.ndarray) -> List[Tuple[float, float, int]]:
        """Find continuous condition segments."""
        if len(timestamps) == 0 or len(labels) == 0:
            return []
        
        segments = []
        current_label = labels[0]
        start_time = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_label:
                # End of current segment
                segments.append((start_time, timestamps[i-1], current_label))
                # Start of new segment
                current_label = labels[i]
                start_time = timestamps[i]
        
        # Add final segment
        segments.append((start_time, timestamps[-1], current_label))
        
        return segments
    
    def _get_quality_color(self, quality_score: float) -> str:
        """Get color for quality visualization."""
        if quality_score >= 0.8:
            return 'green'
        elif quality_score >= 0.6:
            return 'yellow'
        elif quality_score >= 0.4:
            return 'orange'
        else:
            return 'red'
    
    def _add_window_stats_annotation(self, ax: plt.Axes, windows_result: Dict) -> None:
        """Add window statistics annotation."""
        metadata = windows_result.get('metadata', {})
        
        stats_text = f"""Window Stats:
Total: {metadata.get('total_windows', 0)}
Accepted: {metadata.get('accepted_windows', 0)}
Rate: {metadata.get('acceptance_rate', 0):.1%}
Size: {metadata.get('window_size_seconds', 0)}s
Overlap: {metadata.get('overlap_seconds', 0)}s"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)