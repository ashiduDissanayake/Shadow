"""
Documentation Generation Module

Generates comprehensive reports and documentation for WESAD analysis results
including JSON reports, summary statistics, and data export functionality.

Features:
- Generate JSON reports for subjects and dataset
- Create summary statistics and performance metrics
- Export processed data in multiple formats
- Auto-generate analysis documentation
- Performance and quality assessment reports

Author: Shadow AI Team
License: MIT
"""

import json
import csv
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.utils.helpers import WESADHelpers

logger = logging.getLogger(__name__)

class DocumentationGenerator:
    """
    Documentation generator for WESAD analysis pipeline.
    
    Creates comprehensive reports, exports data, and generates
    documentation for analysis results.
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the documentation generator.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.helpers = WESADHelpers(config)
        
        # Create output directories
        self.reports_dir = self.helpers.ensure_output_directory(config.output.reports_dir)
        self.data_dir = self.helpers.ensure_output_directory(config.output.data_dir)
        
        self.logger.info(f"Documentation generator initialized")
        self.logger.info(f"Reports: {self.reports_dir}")
        self.logger.info(f"Data: {self.data_dir}")
    
    def generate_subject_report(self, subject_id: int, subject_results: Dict,
                              save_to_file: bool = True) -> Dict:
        """
        Generate comprehensive subject analysis report.
        
        Args:
            subject_id: Subject ID
            subject_results: Complete subject analysis results
            save_to_file: Whether to save report to file
            
        Returns:
            Subject report dictionary
        """
        try:
            self.logger.info(f"Generating report for subject {subject_id}")
            
            # Create report structure
            report = {
                'metadata': {
                    'subject_id': subject_id,
                    'generated_at': datetime.now().isoformat(),
                    'pipeline_version': '1.0.0',
                    'config_version': self.config.to_dict()
                },
                'data_summary': {},
                'signal_analysis': {},
                'quality_assessment': {},
                'heart_rate_analysis': {},
                'windowing_analysis': {},
                'performance_metrics': {}
            }
            
            # Data summary
            if 'processed_data' in subject_results:
                data = subject_results['processed_data']
                report['data_summary'] = self._generate_data_summary(data)
            
            # Signal analysis
            if 'signal_quality_result' in subject_results:
                quality_result = subject_results['signal_quality_result']
                report['quality_assessment'] = self._generate_quality_summary(quality_result)
            
            # Heart rate analysis
            if 'heart_rate_result' in subject_results:
                hr_result = subject_results['heart_rate_result']
                report['heart_rate_analysis'] = self._generate_hr_summary(hr_result)
            
            # Windowing analysis
            if 'windowing_result' in subject_results:
                window_result = subject_results['windowing_result']
                report['windowing_analysis'] = self._generate_windowing_summary(window_result)
            
            # Performance metrics
            report['performance_metrics'] = self._generate_performance_summary(subject_results)
            
            # Save to file if requested
            if save_to_file:
                timestamp = self.helpers.create_timestamp_string()
                filename = f"subject_{subject_id}_report_{timestamp}.json"
                filepath = self.reports_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=self._json_serializer)
                
                self.logger.info(f"Subject {subject_id} report saved to {filepath}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate subject {subject_id} report: {str(e)}")
            return {}
    
    def generate_dataset_report(self, dataset_results: Dict[int, Dict],
                              save_to_file: bool = True) -> Dict:
        """
        Generate comprehensive dataset analysis report.
        
        Args:
            dataset_results: Complete dataset analysis results
            save_to_file: Whether to save report to file
            
        Returns:
            Dataset report dictionary
        """
        try:
            self.logger.info("Generating dataset-wide report")
            
            # Create report structure
            report = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'pipeline_version': '1.0.0',
                    'total_subjects': len(dataset_results),
                    'processed_subjects': list(dataset_results.keys()),
                    'config': self.config.to_dict()
                },
                'dataset_overview': {},
                'subject_summaries': {},
                'cross_subject_analysis': {},
                'condition_analysis': {},
                'quality_metrics': {},
                'performance_summary': {}
            }
            
            # Dataset overview
            report['dataset_overview'] = self._generate_dataset_overview(dataset_results)
            
            # Subject summaries
            for subject_id, results in dataset_results.items():
                report['subject_summaries'][f'subject_{subject_id}'] = self._generate_subject_summary(results)
            
            # Cross-subject analysis
            report['cross_subject_analysis'] = self._generate_cross_subject_analysis(dataset_results)
            
            # Condition analysis
            report['condition_analysis'] = self._generate_dataset_condition_analysis(dataset_results)
            
            # Quality metrics
            report['quality_metrics'] = self._generate_dataset_quality_metrics(dataset_results)
            
            # Performance summary
            report['performance_summary'] = self._generate_dataset_performance_summary(dataset_results)
            
            # Save to file if requested
            if save_to_file:
                timestamp = self.helpers.create_timestamp_string()
                filename = f"dataset_report_{timestamp}.json"
                filepath = self.reports_dir / filename
                
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=self._json_serializer)
                
                self.logger.info(f"Dataset report saved to {filepath}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate dataset report: {str(e)}")
            return {}
    
    def export_processed_data(self, dataset_results: Dict[int, Dict],
                            formats: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Export processed data in multiple formats.
        
        Args:
            dataset_results: Complete dataset analysis results
            formats: List of export formats ('numpy', 'csv', 'json')
            
        Returns:
            Dictionary mapping formats to lists of created files
        """
        if formats is None:
            formats = []
            if self.config.output.export_numpy:
                formats.append('numpy')
            if self.config.output.export_csv:
                formats.append('csv')
            if self.config.output.export_json:
                formats.append('json')
        
        exported_files = {format: [] for format in formats}
        
        try:
            timestamp = self.helpers.create_timestamp_string()
            
            for subject_id, results in dataset_results.items():
                # Export processed BVP data
                if 'processed_data' in results:
                    data = results['processed_data']
                    
                    for format in formats:
                        filename = f"subject_{subject_id}_processed_data_{timestamp}"
                        filepath = self.data_dir / filename
                        
                        if self.helpers.save_array_data(data, filepath, format):
                            exported_files[format].append(str(filepath))
                
                # Export windowing results
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    if windows:
                        # Prepare windowing data for export
                        window_data = self._prepare_window_data_for_export(windows)
                        
                        for format in formats:
                            filename = f"subject_{subject_id}_windows_{timestamp}"
                            filepath = self.data_dir / filename
                            
                            if self.helpers.save_array_data(window_data, filepath, format):
                                exported_files[format].append(str(filepath))
                
                # Export features if available
                if 'features_result' in results:
                    features_data = results['features_result']
                    
                    for format in formats:
                        filename = f"subject_{subject_id}_features_{timestamp}"
                        filepath = self.data_dir / filename
                        
                        if self.helpers.save_array_data(features_data, filepath, format):
                            exported_files[format].append(str(filepath))
            
            # Export dataset-wide summary
            dataset_summary = self._create_dataset_summary_for_export(dataset_results)
            
            for format in formats:
                filename = f"dataset_summary_{timestamp}"
                filepath = self.data_dir / filename
                
                if self.helpers.save_array_data(dataset_summary, filepath, format):
                    exported_files[format].append(str(filepath))
            
            self.logger.info(f"Data export completed: {sum(len(files) for files in exported_files.values())} files")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Data export failed: {str(e)}")
            return exported_files
    
    def create_analysis_summary(self, dataset_results: Dict[int, Dict]) -> str:
        """
        Create a human-readable analysis summary.
        
        Args:
            dataset_results: Complete dataset analysis results
            
        Returns:
            Formatted summary string
        """
        try:
            summary_lines = []
            summary_lines.append("WESAD Dataset Analysis Summary")
            summary_lines.append("=" * 50)
            summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append("")
            
            # Dataset overview
            total_subjects = len(dataset_results)
            successful_subjects = sum(1 for results in dataset_results.values() 
                                    if 'processed_data' in results)
            
            summary_lines.append(f"Dataset Overview:")
            summary_lines.append(f"  Total subjects: {total_subjects}")
            summary_lines.append(f"  Successfully processed: {successful_subjects}")
            summary_lines.append(f"  Success rate: {successful_subjects/total_subjects*100:.1f}%")
            summary_lines.append("")
            
            # Signal quality overview
            all_qualities = []
            for results in dataset_results.values():
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    all_qualities.extend([w['quality'] for w in windows])
            
            if all_qualities:
                summary_lines.append(f"Signal Quality Overview:")
                summary_lines.append(f"  Total windows: {len(all_qualities)}")
                summary_lines.append(f"  Average quality: {np.mean(all_qualities):.3f}")
                summary_lines.append(f"  Quality std: {np.std(all_qualities):.3f}")
                above_threshold = sum(1 for q in all_qualities if q >= self.config.analysis.quality_threshold)
                summary_lines.append(f"  Above threshold: {above_threshold}/{len(all_qualities)} ({above_threshold/len(all_qualities)*100:.1f}%)")
                summary_lines.append("")
            
            # Condition distribution
            all_labels = []
            for results in dataset_results.values():
                if 'windowing_result' in results:
                    windows = results['windowing_result'].get('windows', [])
                    all_labels.extend([w['label'] for w in windows])
            
            if all_labels:
                from collections import Counter
                label_counts = Counter(all_labels)
                
                summary_lines.append(f"Condition Distribution:")
                for label_id, count in label_counts.items():
                    condition_name = self.config.get_label_name(label_id)
                    percentage = count / len(all_labels) * 100
                    summary_lines.append(f"  {condition_name.title()}: {count} windows ({percentage:.1f}%)")
                summary_lines.append("")
            
            # Heart rate overview
            valid_hrs = []
            for results in dataset_results.values():
                if 'heart_rate_result' in results:
                    hr_result = results['heart_rate_result']
                    if hr_result.get('valid_estimates', 0) > 0:
                        valid_hrs.append(hr_result.get('mean_hr', 0))
            
            if valid_hrs:
                summary_lines.append(f"Heart Rate Overview:")
                summary_lines.append(f"  Subjects with valid HR: {len(valid_hrs)}")
                summary_lines.append(f"  Average HR: {np.mean(valid_hrs):.1f} BPM")
                summary_lines.append(f"  HR range: {np.min(valid_hrs):.1f} - {np.max(valid_hrs):.1f} BPM")
                summary_lines.append("")
            
            # Configuration summary
            summary_lines.append(f"Analysis Configuration:")
            summary_lines.append(f"  Window size: {self.config.analysis.window_size_seconds}s")
            summary_lines.append(f"  Window overlap: {self.config.analysis.overlap_seconds}s")
            summary_lines.append(f"  Quality threshold: {self.config.analysis.quality_threshold}")
            summary_lines.append(f"  HR range: {self.config.analysis.min_heart_rate}-{self.config.analysis.max_heart_rate} BPM")
            summary_lines.append("")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            self.logger.error(f"Failed to create analysis summary: {str(e)}")
            return "Analysis summary generation failed"
    
    def _generate_data_summary(self, data: Dict) -> Dict:
        """Generate data summary section."""
        summary = {}
        
        if 'bvp' in data:
            bvp = data['bvp']
            summary['bvp_signal'] = {
                'length': len(bvp),
                'duration_seconds': len(bvp) / self.config.dataset.bvp_sampling_rate,
                'sampling_rate': self.config.dataset.bvp_sampling_rate,
                'statistics': self.helpers.calculate_basic_statistics(bvp)
            }
        
        if 'labels' in data:
            labels = data['labels']
            from collections import Counter
            label_counts = Counter(labels)
            condition_counts = {}
            for label_id, count in label_counts.items():
                condition_name = self.helpers.safe_label_lookup(label_id)
                condition_counts[condition_name] = count
            
            summary['labels'] = {
                'length': len(labels),
                'unique_conditions': len(label_counts),
                'condition_distribution': condition_counts
            }
        
        if 'timestamps' in data:
            timestamps = data['timestamps']
            summary['timestamps'] = {
                'start': float(timestamps[0]) if len(timestamps) > 0 else 0,
                'end': float(timestamps[-1]) if len(timestamps) > 0 else 0,
                'duration': float(timestamps[-1] - timestamps[0]) if len(timestamps) > 0 else 0
            }
        
        return summary
    
    def _generate_quality_summary(self, quality_result: Dict) -> Dict:
        """Generate quality assessment summary."""
        summary = {
            'overall_score': quality_result.get('overall_score', 0),
            'quality_level': quality_result.get('quality_level', 'unknown'),
            'metrics': quality_result.get('metrics', {}),
            'signal_length': quality_result.get('signal_length', 0)
        }
        return summary
    
    def _generate_hr_summary(self, hr_result: Dict) -> Dict:
        """Generate heart rate analysis summary."""
        summary = {
            'mean_hr': hr_result.get('mean_hr', 0),
            'median_hr': hr_result.get('median_hr', 0),
            'std_hr': hr_result.get('std_hr', 0),
            'hr_range': [hr_result.get('min_hr', 0), hr_result.get('max_hr', 0)],
            'valid_estimates': hr_result.get('valid_estimates', 0),
            'total_estimates': hr_result.get('total_estimates', 0),
            'validity_ratio': hr_result.get('validity_ratio', 0),
            'hrv_metrics': hr_result.get('hrv_metrics', {}),
            'analysis_duration': hr_result.get('analysis_duration', 0)
        }
        return summary
    
    def _generate_windowing_summary(self, window_result: Dict) -> Dict:
        """Generate windowing analysis summary."""
        metadata = window_result.get('metadata', {})
        summary_stats = window_result.get('summary_stats', {})
        
        summary = {
            'total_windows': metadata.get('total_windows', 0),
            'accepted_windows': metadata.get('accepted_windows', 0),
            'rejected_windows': metadata.get('rejected_windows', 0),
            'acceptance_rate': metadata.get('acceptance_rate', 0),
            'window_size_seconds': metadata.get('window_size_seconds', 0),
            'overlap_seconds': metadata.get('overlap_seconds', 0),
            'signal_duration': metadata.get('signal_duration', 0),
            'statistics': summary_stats
        }
        return summary
    
    def _generate_performance_summary(self, subject_results: Dict) -> Dict:
        """Generate performance metrics summary."""
        summary = {
            'processing_successful': 'processed_data' in subject_results,
            'quality_assessment_performed': 'signal_quality_result' in subject_results,
            'heart_rate_analysis_performed': 'heart_rate_result' in subject_results,
            'windowing_analysis_performed': 'windowing_result' in subject_results
        }
        
        # Calculate overall success score
        success_count = sum(1 for v in summary.values() if v)
        summary['overall_success_score'] = success_count / len(summary)
        
        return summary
    
    def _generate_dataset_overview(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Generate dataset overview section."""
        overview = {
            'total_subjects': len(dataset_results),
            'processed_subjects': len([s for s in dataset_results.values() if 'processed_data' in s]),
            'subjects_with_windows': len([s for s in dataset_results.values() if 'windowing_result' in s]),
            'subjects_with_hr': len([s for s in dataset_results.values() if 'heart_rate_result' in s])
        }
        
        overview['processing_success_rate'] = overview['processed_subjects'] / overview['total_subjects']
        overview['windowing_success_rate'] = overview['subjects_with_windows'] / overview['total_subjects']
        overview['hr_success_rate'] = overview['subjects_with_hr'] / overview['total_subjects']
        
        return overview
    
    def _generate_subject_summary(self, results: Dict) -> Dict:
        """Generate summary for a single subject."""
        summary = {
            'has_processed_data': 'processed_data' in results,
            'has_quality_assessment': 'signal_quality_result' in results,
            'has_heart_rate_analysis': 'heart_rate_result' in results,
            'has_windowing_analysis': 'windowing_result' in results
        }
        
        if 'windowing_result' in results:
            metadata = results['windowing_result'].get('metadata', {})
            summary['window_count'] = metadata.get('accepted_windows', 0)
            summary['acceptance_rate'] = metadata.get('acceptance_rate', 0)
        
        if 'heart_rate_result' in results:
            hr_result = results['heart_rate_result']
            summary['mean_heart_rate'] = hr_result.get('mean_hr', 0)
            summary['hr_validity_ratio'] = hr_result.get('validity_ratio', 0)
        
        return summary
    
    def _generate_cross_subject_analysis(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Generate cross-subject analysis."""
        # Collect metrics across all subjects
        quality_scores = []
        heart_rates = []
        acceptance_rates = []
        window_counts = []
        
        for results in dataset_results.values():
            if 'windowing_result' in results:
                windows = results['windowing_result'].get('windows', [])
                if windows:
                    subject_qualities = [w['quality'] for w in windows]
                    quality_scores.extend(subject_qualities)
                    
                    metadata = results['windowing_result'].get('metadata', {})
                    acceptance_rates.append(metadata.get('acceptance_rate', 0))
                    window_counts.append(metadata.get('accepted_windows', 0))
            
            if 'heart_rate_result' in results:
                hr_result = results['heart_rate_result']
                if hr_result.get('valid_estimates', 0) > 0:
                    heart_rates.append(hr_result.get('mean_hr', 0))
        
        analysis = {}
        
        if quality_scores:
            analysis['quality_statistics'] = self.helpers.calculate_basic_statistics(np.array(quality_scores))
        
        if heart_rates:
            analysis['heart_rate_statistics'] = self.helpers.calculate_basic_statistics(np.array(heart_rates))
        
        if acceptance_rates:
            analysis['acceptance_rate_statistics'] = self.helpers.calculate_basic_statistics(np.array(acceptance_rates))
        
        if window_counts:
            analysis['window_count_statistics'] = self.helpers.calculate_basic_statistics(np.array(window_counts))
        
        return analysis
    
    def _generate_dataset_condition_analysis(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Generate dataset-wide condition analysis."""
        condition_data = {}
        
        for results in dataset_results.values():
            if 'windowing_result' not in results:
                continue
            
            windows = results['windowing_result'].get('windows', [])
            for window in windows:
                condition_name = self.helpers.safe_label_lookup(window['label'])
                
                if condition_name not in condition_data:
                    condition_data[condition_name] = {
                        'window_count': 0,
                        'quality_scores': [],
                        'confidence_scores': []
                    }
                
                condition_data[condition_name]['window_count'] += 1
                condition_data[condition_name]['quality_scores'].append(window['quality'])
                condition_data[condition_name]['confidence_scores'].append(window['confidence'])
        
        # Calculate statistics for each condition
        analysis = {}
        for condition, data in condition_data.items():
            if data['window_count'] > 0:
                analysis[condition] = {
                    'window_count': data['window_count'],
                    'quality_statistics': self.helpers.calculate_basic_statistics(np.array(data['quality_scores'])),
                    'confidence_statistics': self.helpers.calculate_basic_statistics(np.array(data['confidence_scores']))
                }
        
        return analysis
    
    def _generate_dataset_quality_metrics(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Generate dataset quality metrics."""
        all_qualities = []
        subject_qualities = {}
        
        for subject_id, results in dataset_results.items():
            if 'windowing_result' in results:
                windows = results['windowing_result'].get('windows', [])
                if windows:
                    subject_quality_scores = [w['quality'] for w in windows]
                    all_qualities.extend(subject_quality_scores)
                    subject_qualities[f'subject_{subject_id}'] = {
                        'mean_quality': np.mean(subject_quality_scores),
                        'window_count': len(subject_quality_scores)
                    }
        
        metrics = {}
        
        if all_qualities:
            metrics['overall_quality'] = self.helpers.calculate_basic_statistics(np.array(all_qualities))
            metrics['quality_threshold'] = self.config.analysis.quality_threshold
            
            above_threshold = sum(1 for q in all_qualities if q >= self.config.analysis.quality_threshold)
            metrics['threshold_compliance'] = {
                'above_threshold': above_threshold,
                'total_windows': len(all_qualities),
                'compliance_rate': above_threshold / len(all_qualities)
            }
        
        metrics['subject_quality_summary'] = subject_qualities
        
        return metrics
    
    def _generate_dataset_performance_summary(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Generate dataset performance summary."""
        summary = {
            'total_subjects': len(dataset_results),
            'processing_stages': {
                'data_loading': 0,
                'signal_quality': 0,
                'heart_rate_analysis': 0,
                'windowing_analysis': 0
            }
        }
        
        for results in dataset_results.values():
            if 'processed_data' in results:
                summary['processing_stages']['data_loading'] += 1
            if 'signal_quality_result' in results:
                summary['processing_stages']['signal_quality'] += 1
            if 'heart_rate_result' in results:
                summary['processing_stages']['heart_rate_analysis'] += 1
            if 'windowing_result' in results:
                summary['processing_stages']['windowing_analysis'] += 1
        
        # Calculate success rates
        summary['success_rates'] = {}
        for stage, count in summary['processing_stages'].items():
            summary['success_rates'][stage] = count / summary['total_subjects']
        
        return summary
    
    def _prepare_window_data_for_export(self, windows: List[Dict]) -> Dict:
        """Prepare window data for export."""
        if not windows:
            return {}
        
        export_data = {
            'window_labels': [w['label'] for w in windows],
            'window_qualities': [w['quality'] for w in windows],
            'window_confidences': [w['confidence'] for w in windows],
            'start_times': [w['start_time'] for w in windows],
            'end_times': [w['end_time'] for w in windows],
            'window_ids': [w['window_id'] for w in windows]
        }
        
        # Add BVP data if available
        if 'bvp' in windows[0]:
            bvp_data = np.array([w['bvp'] for w in windows])
            export_data['bvp_windows'] = bvp_data
        
        return export_data
    
    def _create_dataset_summary_for_export(self, dataset_results: Dict[int, Dict]) -> Dict:
        """Create dataset summary for export."""
        summary = {
            'subject_ids': list(dataset_results.keys()),
            'total_subjects': len(dataset_results)
        }
        
        # Aggregate statistics
        all_window_counts = []
        all_quality_scores = []
        all_heart_rates = []
        
        for subject_id, results in dataset_results.items():
            if 'windowing_result' in results:
                metadata = results['windowing_result'].get('metadata', {})
                all_window_counts.append(metadata.get('accepted_windows', 0))
                
                windows = results['windowing_result'].get('windows', [])
                if windows:
                    all_quality_scores.extend([w['quality'] for w in windows])
            
            if 'heart_rate_result' in results:
                hr_result = results['heart_rate_result']
                if hr_result.get('valid_estimates', 0) > 0:
                    all_heart_rates.append(hr_result.get('mean_hr', 0))
        
        if all_window_counts:
            summary['window_counts'] = np.array(all_window_counts)
        
        if all_quality_scores:
            summary['quality_scores'] = np.array(all_quality_scores)
        
        if all_heart_rates:
            summary['heart_rates'] = np.array(all_heart_rates)
        
        return summary
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")