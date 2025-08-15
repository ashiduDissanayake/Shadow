"""
Signal Plotting Module

Provides comprehensive plotting capabilities for BVP signals, quality metrics,
and heart rate visualization with condition backgrounds and annotations.

Features:
- Plot complete BVP signal with condition backgrounds
- Plot signal quality over time
- Plot heart rate over time
- Subject-specific essential plots
- High-quality output with customizable styling

Author: Shadow AI Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

from wesad_pipeline.config import WESADConfig

logger = logging.getLogger(__name__)

class SignalPlotter:
    """
    Signal plotting class for BVP analysis visualization.
    
    Provides comprehensive visualization capabilities for BVP signals,
    quality metrics, and heart rate with proper condition annotation.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the signal plotter.
        
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
            self.output_dir = Path(config.output.output_path) / config.output.plots_dir / "signals"
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
        
        self.logger.info(f"Signal plotter initialized (output: {self.output_dir})")
    
    def plot_bvp_signal(self, bvp_signal: np.ndarray, labels: np.ndarray,
                       timestamps: Optional[np.ndarray] = None,
                       title: str = "BVP Signal",
                       subject_id: Optional[int] = None,
                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot BVP signal with condition backgrounds.
        
        Args:
            bvp_signal: BVP signal array
            labels: Corresponding labels array
            timestamps: Optional timestamps array
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
            
            # Ensure arrays have same length
            min_length = min(len(bvp_signal), len(labels), len(timestamps))
            bvp_signal = bvp_signal[:min_length]
            labels = labels[:min_length]
            timestamps = timestamps[:min_length]
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot condition backgrounds
            self._plot_condition_backgrounds(ax, timestamps, labels)
            
            # Plot BVP signal
            ax.plot(timestamps, bvp_signal, 'k-', linewidth=0.8, alpha=0.8, label='BVP Signal')
            
            # Customize plot
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('BVP Amplitude', fontsize=12)
            ax.set_title(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add condition legend
            self._add_condition_legend(ax)
            
            # Add signal statistics annotation
            self._add_signal_stats_annotation(ax, bvp_signal, timestamps)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved BVP signal plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"BVP signal plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_signal_quality(self, timestamps: np.ndarray, quality_scores: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           title: str = "Signal Quality",
                           subject_id: Optional[int] = None,
                           save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot signal quality over time.
        
        Args:
            timestamps: Time points array
            quality_scores: Quality scores array
            labels: Optional labels for condition backgrounds
            title: Plot title
            subject_id: Subject ID for annotation
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if len(quality_scores) == 0:
                self.logger.warning("Empty quality scores provided")
                return plt.figure()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot condition backgrounds if labels provided
            if labels is not None and len(labels) == len(timestamps):
                self._plot_condition_backgrounds(ax, timestamps, labels)
            
            # Create color-coded quality plot
            colors = self._get_quality_colors(quality_scores)
            
            # Use LineCollection for efficient color-coded plotting
            points = np.array([timestamps, quality_scores]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=colors[:-1], linewidths=2)
            ax.add_collection(lc)
            
            # Set axis limits
            ax.set_xlim(timestamps.min(), timestamps.max())
            ax.set_ylim(0, 1)
            
            # Add quality threshold line
            quality_threshold = self.config.analysis.quality_threshold
            ax.axhline(y=quality_threshold, color='red', linestyle='--', alpha=0.7,
                      label=f'Quality Threshold ({quality_threshold:.2f})')
            
            # Customize plot
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Signal Quality Score', fontsize=12)
            ax.set_title(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add quality statistics annotation
            self._add_quality_stats_annotation(ax, quality_scores)
            
            # Add condition legend if labels provided
            if labels is not None:
                self._add_condition_legend(ax)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved quality plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Quality plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_heart_rate(self, timestamps: np.ndarray, heart_rates: np.ndarray,
                       labels: Optional[np.ndarray] = None,
                       peak_times: Optional[np.ndarray] = None,
                       title: str = "Heart Rate",
                       subject_id: Optional[int] = None,
                       save_name: Optional[str] = None) -> plt.Figure:
        """
        Plot heart rate over time with physiological ranges.
        
        Args:
            timestamps: Time points array
            heart_rates: Heart rate values array (BPM)
            labels: Optional labels for condition backgrounds
            peak_times: Optional peak detection times
            title: Plot title
            subject_id: Subject ID for annotation
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            if len(heart_rates) == 0:
                self.logger.warning("Empty heart rate data provided")
                return plt.figure()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            
            # Plot condition backgrounds if labels provided
            if labels is not None and len(labels) == len(timestamps):
                self._plot_condition_backgrounds(ax, timestamps, labels)
            
            # Plot heart rate
            ax.plot(timestamps, heart_rates, 'b-', linewidth=2, alpha=0.8, label='Heart Rate')
            
            # Add physiological range bands
            ax.axhspan(self.config.analysis.min_heart_rate, self.config.analysis.max_heart_rate, 
                      alpha=0.1, color='green', label='Physiological Range')
            
            # Add normal resting HR range
            ax.axhspan(60, 100, alpha=0.1, color='blue', label='Normal Resting Range')
            
            # Mark peaks if provided
            if peak_times is not None and len(peak_times) > 0:
                # Interpolate HR values at peak times
                peak_hrs = np.interp(peak_times, timestamps, heart_rates)
                ax.scatter(peak_times, peak_hrs, c='red', s=20, alpha=0.6, 
                          label=f'Detected Peaks (n={len(peak_times)})', zorder=5)
            
            # Customize plot
            ax.set_xlabel('Time (seconds)', fontsize=12)
            ax.set_ylabel('Heart Rate (BPM)', fontsize=12)
            ax.set_title(f'{title}' + (f' - Subject {subject_id}' if subject_id else ''), 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Set reasonable y-axis limits
            hr_min = max(30, np.min(heart_rates) - 10)
            hr_max = min(220, np.max(heart_rates) + 10)
            ax.set_ylim(hr_min, hr_max)
            
            # Add heart rate statistics annotation
            self._add_hr_stats_annotation(ax, heart_rates)
            
            # Add condition legend if labels provided
            if labels is not None:
                self._add_condition_legend(ax)
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved heart rate plot: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Heart rate plotting failed: {str(e)}")
            return plt.figure()
    
    def plot_subject_overview(self, subject_data: Dict, subject_id: int,
                             save_name: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive overview plot for a subject.
        
        Args:
            subject_data: Processed subject data dictionary
            subject_id: Subject ID
            save_name: Optional save filename
            
        Returns:
            Matplotlib figure object
        """
        try:
            bvp = subject_data.get('bvp', np.array([]))
            labels = subject_data.get('labels', np.array([]))
            timestamps = subject_data.get('timestamps', np.array([]))
            
            if len(bvp) == 0:
                self.logger.warning(f"No data for subject {subject_id}")
                return plt.figure()
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 1, figsize=(self.figure_size[0], self.figure_size[1] * 1.5), 
                                   dpi=self.dpi, sharex=True)
            
            # 1. BVP Signal
            self._plot_condition_backgrounds(axes[0], timestamps, labels)
            axes[0].plot(timestamps, bvp, 'k-', linewidth=0.8, alpha=0.8)
            axes[0].set_ylabel('BVP Amplitude', fontsize=11)
            axes[0].set_title(f'Subject {subject_id} - BVP Signal Overview', 
                            fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # 2. Signal Quality (if available)
            if 'quality_scores' in subject_data:
                quality_scores = subject_data['quality_scores']
                if len(quality_scores) == len(timestamps):
                    self._plot_condition_backgrounds(axes[1], timestamps, labels)
                    colors = self._get_quality_colors(quality_scores)
                    points = np.array([timestamps, quality_scores]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    lc = LineCollection(segments, colors=colors[:-1], linewidths=1.5)
                    axes[1].add_collection(lc)
                    axes[1].set_xlim(timestamps.min(), timestamps.max())
                    axes[1].set_ylim(0, 1)
                    axes[1].axhline(y=self.config.analysis.quality_threshold, 
                                  color='red', linestyle='--', alpha=0.7)
                else:
                    axes[1].text(0.5, 0.5, 'Quality data unavailable', 
                               transform=axes[1].transAxes, ha='center', va='center')
            else:
                axes[1].text(0.5, 0.5, 'Quality assessment not performed', 
                           transform=axes[1].transAxes, ha='center', va='center')
            
            axes[1].set_ylabel('Signal Quality', fontsize=11)
            axes[1].grid(True, alpha=0.3)
            
            # 3. Heart Rate (if available)
            if 'heart_rates' in subject_data:
                heart_rates = subject_data['heart_rates']
                if len(heart_rates) > 0:
                    # Create corresponding timestamps for HR data
                    hr_timestamps = np.linspace(timestamps[0], timestamps[-1], len(heart_rates))
                    hr_labels = np.interp(hr_timestamps, timestamps, labels).astype(int)
                    
                    self._plot_condition_backgrounds(axes[2], hr_timestamps, hr_labels)
                    axes[2].plot(hr_timestamps, heart_rates, 'b-', linewidth=1.5, alpha=0.8)
                    axes[2].axhspan(60, 100, alpha=0.1, color='blue')
                    axes[2].set_ylim(max(30, np.min(heart_rates) - 10), 
                                   min(220, np.max(heart_rates) + 10))
                else:
                    axes[2].text(0.5, 0.5, 'Heart rate data unavailable', 
                               transform=axes[2].transAxes, ha='center', va='center')
            else:
                axes[2].text(0.5, 0.5, 'Heart rate analysis not performed', 
                           transform=axes[2].transAxes, ha='center', va='center')
            
            axes[2].set_ylabel('Heart Rate (BPM)', fontsize=11)
            axes[2].set_xlabel('Time (seconds)', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            
            # Add condition legend to the first subplot
            self._add_condition_legend(axes[0])
            
            plt.tight_layout()
            
            # Save if requested
            if save_name and self.output_dir:
                save_path = self.output_dir / f"{save_name}.{self.config.visualization.plot_format}"
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                self.logger.debug(f"Saved subject overview: {save_path}")
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Subject overview plotting failed: {str(e)}")
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
    
    def _get_quality_colors(self, quality_scores: np.ndarray) -> List[str]:
        """Get color array for quality visualization."""
        colors = []
        for score in quality_scores:
            if score >= 0.8:
                colors.append('green')
            elif score >= 0.6:
                colors.append('yellow')
            elif score >= 0.4:
                colors.append('orange')
            else:
                colors.append('red')
        return colors
    
    def _add_condition_legend(self, ax: plt.Axes) -> None:
        """Add condition legend to plot."""
        legend_elements = []
        for condition_name, color in self.condition_colors.items():
            if condition_name in self.config.dataset.target_conditions:
                legend_elements.append(patches.Patch(color=color, alpha=0.2, label=condition_name.title()))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    
    def _add_signal_stats_annotation(self, ax: plt.Axes, bvp_signal: np.ndarray, 
                                   timestamps: np.ndarray) -> None:
        """Add signal statistics annotation."""
        stats_text = f"""Signal Stats:
Duration: {timestamps[-1] - timestamps[0]:.1f}s
Mean: {np.mean(bvp_signal):.3f}
Std: {np.std(bvp_signal):.3f}
Range: {np.ptp(bvp_signal):.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    def _add_quality_stats_annotation(self, ax: plt.Axes, quality_scores: np.ndarray) -> None:
        """Add quality statistics annotation."""
        stats_text = f"""Quality Stats:
Mean: {np.mean(quality_scores):.3f}
Min: {np.min(quality_scores):.3f}
Max: {np.max(quality_scores):.3f}
Std: {np.std(quality_scores):.3f}"""
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)
    
    def _add_hr_stats_annotation(self, ax: plt.Axes, heart_rates: np.ndarray) -> None:
        """Add heart rate statistics annotation."""
        valid_hrs = heart_rates[(heart_rates >= self.config.analysis.min_heart_rate) & 
                               (heart_rates <= self.config.analysis.max_heart_rate)]
        
        if len(valid_hrs) > 0:
            stats_text = f"""HR Stats:
Mean: {np.mean(valid_hrs):.1f} BPM
Min: {np.min(valid_hrs):.1f} BPM
Max: {np.max(valid_hrs):.1f} BPM
Std: {np.std(valid_hrs):.1f} BPM"""
        else:
            stats_text = "HR Stats:\nNo valid data"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               fontsize=9)