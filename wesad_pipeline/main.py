"""
WESAD Analysis Pipeline Main Module

Orchestrates the complete WESAD analysis pipeline with the new processing flow:
1. Signal Quality Assessment (whole signal)
2. Windowing (if signal is good)
3. Windowed Quality Assessment
4. Heart Rate Analysis

Author: Shadow AI Team
License: MIT
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import numpy as np

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.data.loader import WESADLoader
from wesad_pipeline.data.preprocessor import WESADPreprocessor
from wesad_pipeline.analysis.signal_quality import SignalQuality
from wesad_pipeline.analysis.windowed_quality import WindowedQuality
from wesad_pipeline.analysis.windowing import WindowAnalyzer
from wesad_pipeline.analysis.heart_rate import HeartRateAnalyzer
from wesad_pipeline.visualization.signal_plots import SignalPlotter
from wesad_pipeline.visualization.window_plots import WindowPlotter
from wesad_pipeline.visualization.dataset_plots import DatasetPlotter
from wesad_pipeline.utils.report_generator import ReportGenerator
from wesad_pipeline.utils.data_exporter import DataExporter

logger = logging.getLogger(__name__)

class WESADPipeline:
    """
    Main WESAD Analysis Pipeline with improved processing flow.
    
    Features:
    - Sequential quality-based processing
    - Modular analysis components
    - Comprehensive visualization
    - Detailed reporting and export
    """
    
    def __init__(self, config: WESADConfig):
        """
        Initialize the WESAD pipeline.
        
        Args:
            config: Pipeline configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Results storage
        self.results = {}
        
        self.logger.info("WESAD Pipeline initialized with new processing flow")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Data components
            self.loader = WESADLoader(self.config)
            self.preprocessor = WESADPreprocessor(self.config)
            
            # Analysis components (new order)
            self.signal_quality = SignalQuality(self.config)
            self.windowed_quality = WindowedQuality(self.config)
            self.window_analyzer = WindowAnalyzer(self.config)
            self.heart_rate = HeartRateAnalyzer(self.config)
            
            # Visualization components
            if self.config.visualization.enable_plotting:
                self.signal_plotter = SignalPlotter(self.config)
                self.window_plotter = WindowPlotter(self.config)
                self.dataset_plotter = DatasetPlotter(self.config)
            
            # Export components
            self.report_generator = ReportGenerator(self.config)
            self.data_exporter = DataExporter(self.config)
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise
    
    def run_full_pipeline(self, subject_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run the complete WESAD analysis pipeline with new processing flow.
        
        Args:
            subject_ids: Optional list of subject IDs to process
            
        Returns:
            Dictionary containing complete pipeline results
        """
        try:
            self.logger.info("Starting WESAD Analysis Pipeline")
            
            # Phase 1: Data loading and preprocessing
            dataset_results = self.run_data_processing(subject_ids)
            
            # Phase 2: Signal quality assessment (whole signal)
            dataset_results = self.run_signal_quality_assessment(dataset_results)
            
            # Phase 3: Windowing (for good quality signals)
            dataset_results = self.run_windowing_analysis(dataset_results)
            
            # Phase 4: Windowed quality assessment
            dataset_results = self.run_windowed_quality_assessment(dataset_results)
            
            # Phase 5: Heart rate analysis
            dataset_results = self.run_heart_rate_analysis(dataset_results)
            
            # Phase 6: Visualization
            if self.config.visualization.enable_plotting:
                self.run_visualization(dataset_results)
            
            # Phase 7: Generate reports and export
            final_results = self.finalize_results(dataset_results)
            
            self.logger.info("WESAD Analysis Pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
    
    def run_data_processing(self, subject_ids: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        Run data loading and preprocessing phase.
        
        Args:
            subject_ids: Optional list of subject IDs to process
            
        Returns:
            Dictionary with processed data for each subject
        """
        try:
            self.logger.info("Starting data processing phase")
            
            # Load raw data
            raw_data = self.loader.load_subjects(subject_ids)
            
            # Process each subject
            dataset_results = {}
            for subject_id, subject_data in tqdm(raw_data.items(), desc="Processing subjects"):
                try:
                    processed_data = self.preprocessor.preprocess_subject(subject_data)
                    dataset_results[subject_id] = {
                        'raw_data': subject_data,
                        'processed_data': processed_data,
                        'processing_status': 'completed'
                    }
                except Exception as e:
                    self.logger.error(f"Processing failed for subject {subject_id}: {str(e)}")
                    dataset_results[subject_id] = {
                        'processing_status': 'failed',
                        'error': str(e)
                    }
            
            self.logger.info(f"Data processing completed for {len(dataset_results)} subjects")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Data processing phase failed: {str(e)}")
            raise
    
    def run_signal_quality_assessment(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run signal quality assessment phase (whole signal).
        
        Args:
            dataset_results: Results from data processing
            
        Returns:
            Updated dataset results with signal quality assessment
        """
        try:
            self.logger.info("Starting signal quality assessment phase")
            
            for subject_id, results in tqdm(dataset_results.items(), desc="Signal quality assessment"):
                if results.get('processing_status') != 'completed':
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                
                if len(bvp_signal) == 0:
                    self.logger.warning(f"No BVP data for subject {subject_id}")
                    continue
                
                try:
                    # Assess overall signal quality
                    quality_result = self.signal_quality.assess_signal_quality(bvp_signal)
                    results['signal_quality_result'] = quality_result
                    
                    # Check if signal meets quality threshold
                    quality_threshold = self.config.analysis.quality_threshold
                    meets_threshold = quality_result['overall_score'] >= quality_threshold
                    results['signal_quality_passed'] = meets_threshold
                    
                    if meets_threshold:
                        self.logger.debug(f"Subject {subject_id}: Signal quality passed "
                                        f"(score: {quality_result['overall_score']:.3f})")
                    else:
                        self.logger.warning(f"Subject {subject_id}: Signal quality failed "
                                          f"(score: {quality_result['overall_score']:.3f} < {quality_threshold})")
                    
                except Exception as e:
                    self.logger.error(f"Signal quality assessment failed for subject {subject_id}: {str(e)}")
                    results['signal_quality_result'] = None
                    results['signal_quality_passed'] = False
            
            # Count subjects that passed quality assessment
            passed_subjects = sum(1 for r in dataset_results.values() 
                                if r.get('signal_quality_passed', False))
            total_subjects = len([r for r in dataset_results.values() 
                                if r.get('processing_status') == 'completed'])
            
            self.logger.info(f"Signal quality assessment completed: {passed_subjects}/{total_subjects} subjects passed")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Signal quality assessment phase failed: {str(e)}")
            raise
    
    def run_windowing_analysis(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run windowing analysis phase (only for good quality signals).
        
        Args:
            dataset_results: Results from signal quality assessment
            
        Returns:
            Updated dataset results with windowing analysis
        """
        try:
            self.logger.info("Starting windowing analysis phase")
            
            processed_subjects = 0
            for subject_id, results in tqdm(dataset_results.items(), desc="Windowing analysis"):
                # Only process subjects that passed signal quality assessment
                if not results.get('signal_quality_passed', False):
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                labels = processed_data.get('labels', np.array([]))
                
                if len(bvp_signal) == 0 or len(labels) == 0:
                    self.logger.warning(f"Insufficient data for subject {subject_id}")
                    continue
                
                try:
                    # Create windows
                    windows_result = self.window_analyzer.create_windows(bvp_signal, labels)
                    results['windows_result'] = windows_result
                    processed_subjects += 1
                    
                    self.logger.debug(f"Subject {subject_id}: Created {len(windows_result['windows'])} windows")
                    
                except Exception as e:
                    self.logger.error(f"Windowing failed for subject {subject_id}: {str(e)}")
                    results['windows_result'] = None
            
            self.logger.info(f"Windowing analysis completed for {processed_subjects} subjects")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Windowing analysis phase failed: {str(e)}")
            raise
    
    def run_windowed_quality_assessment(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run windowed quality assessment phase.
        
        Args:
            dataset_results: Results from windowing analysis
            
        Returns:
            Updated dataset results with windowed quality assessment
        """
        try:
            self.logger.info("Starting windowed quality assessment phase")
            
            processed_subjects = 0
            for subject_id, results in tqdm(dataset_results.items(), desc="Windowed quality assessment"):
                # Only process subjects that have windows
                if 'windows_result' not in results or results['windows_result'] is None:
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                
                if len(bvp_signal) == 0:
                    continue
                
                try:
                    # Assess windowed quality
                    windowed_quality_result = self.windowed_quality.assess_windowed_quality(bvp_signal)
                    results['windowed_quality_result'] = windowed_quality_result
                    
                    # Validate windowed quality
                    validation_result = self.windowed_quality.validate_windowed_quality(bvp_signal)
                    results['windowed_quality_validation'] = validation_result
                    
                    processed_subjects += 1
                    
                    threshold_ratio = windowed_quality_result['threshold_ratio']
                    self.logger.debug(f"Subject {subject_id}: {threshold_ratio:.1%} windows above quality threshold")
                    
                except Exception as e:
                    self.logger.error(f"Windowed quality assessment failed for subject {subject_id}: {str(e)}")
                    results['windowed_quality_result'] = None
                    results['windowed_quality_validation'] = None
            
            self.logger.info(f"Windowed quality assessment completed for {processed_subjects} subjects")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Windowed quality assessment phase failed: {str(e)}")
            raise
    
    def run_heart_rate_analysis(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run heart rate analysis phase.
        
        Args:
            dataset_results: Results from windowed quality assessment
            
        Returns:
            Updated dataset results with heart rate analysis
        """
        try:
            self.logger.info("Starting heart rate analysis phase")
            
            processed_subjects = 0
            for subject_id, results in tqdm(dataset_results.items(), desc="Heart rate analysis"):
                # Only process subjects with windowed quality results
                if 'windowed_quality_result' not in results:
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                
                if len(bvp_signal) == 0:
                    continue
                
                try:
                    # Heart rate analysis
                    hr_result = self.heart_rate.estimate_heart_rate(bvp_signal)
                    results['heart_rate_result'] = hr_result
                    processed_subjects += 1
                    
                    mean_hr = hr_result.get('mean_hr', 0)
                    self.logger.debug(f"Subject {subject_id}: Mean heart rate {mean_hr:.1f} BPM")
                    
                except Exception as e:
                    self.logger.error(f"Heart rate analysis failed for subject {subject_id}: {str(e)}")
                    results['heart_rate_result'] = None
            
            self.logger.info(f"Heart rate analysis completed for {processed_subjects} subjects")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Heart rate analysis phase failed: {str(e)}")
            raise
    
    def run_visualization(self, dataset_results: Dict[int, Dict]):
        """
        Run visualization phase.
        
        Args:
            dataset_results: Complete analysis results
        """
        try:
            self.logger.info("Starting visualization phase")
            
            # Generate visualizations for each subject
            for subject_id, results in dataset_results.items():
                if results.get('processing_status') != 'completed':
                    continue
                
                try:
                    # Signal plots
                    if 'processed_data' in results:
                        processed_data = results['processed_data']
                        bvp_signal = processed_data.get('bvp', np.array([]))
                        labels = processed_data.get('labels', np.array([]))
                        
                        if len(bvp_signal) > 0:
                            self.signal_plotter.plot_bvp_signal(
                                bvp_signal, labels, subject_id=subject_id,
                                save_name=f"bvp_signal_subject_{subject_id}"
                            )
                    
                    # Quality plots
                    if 'signal_quality_result' in results and results['signal_quality_result']:
                        quality_result = results['signal_quality_result']
                        # Add quality visualization here
                    
                    # Window plots
                    if 'windows_result' in results and results['windows_result']:
                        windows_result = results['windows_result']
                        # Add window visualization here
                    
                    # Heart rate plots
                    if 'heart_rate_result' in results and results['heart_rate_result']:
                        hr_result = results['heart_rate_result']
                        # Add heart rate visualization here
                        
                except Exception as e:
                    self.logger.error(f"Visualization failed for subject {subject_id}: {str(e)}")
            
            # Generate dataset-wide visualizations
            try:
                self.dataset_plotter.plot_dataset_overview(dataset_results)
            except Exception as e:
                self.logger.error(f"Dataset overview visualization failed: {str(e)}")
            
            self.logger.info("Visualization phase completed")
            
        except Exception as e:
            self.logger.error(f"Visualization phase failed: {str(e)}")
    
    def finalize_results(self, dataset_results: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Generate final reports and export results.
        
        Args:
            dataset_results: Complete analysis results
            
        Returns:
            Final pipeline results
        """
        try:
            self.logger.info("Finalizing pipeline results")
            
            # Generate comprehensive report
            report = self.report_generator.generate_pipeline_report(dataset_results)
            
            # Export data
            export_summary = self.data_exporter.export_dataset_results(dataset_results)
            
            # Create final results summary
            final_results = {
                'pipeline_config': self.config.to_dict(),
                'dataset_results': dataset_results,
                'pipeline_report': report,
                'export_summary': export_summary,
                'pipeline_statistics': self._calculate_pipeline_statistics(dataset_results)
            }
            
            # Save final results
            if self.config.export.save_results:
                results_path = Path(self.config.export.output_dir) / "final_results.json"
                with open(results_path, 'w') as f:
                    json.dump(final_results, f, indent=2, default=str)
                self.logger.info(f"Final results saved to {results_path}")
            
            self.logger.info("Pipeline finalization completed")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Pipeline finalization failed: {str(e)}")
            raise
    
    def _calculate_pipeline_statistics(self, dataset_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Calculate overall pipeline statistics."""
        stats = {
            'total_subjects': len(dataset_results),
            'successfully_processed': 0,
            'quality_passed': 0,
            'windowed': 0,
            'heart_rate_analyzed': 0,
            'processing_success_rate': 0.0,
            'quality_pass_rate': 0.0
        }
        
        for results in dataset_results.values():
            if results.get('processing_status') == 'completed':
                stats['successfully_processed'] += 1
                
                if results.get('signal_quality_passed', False):
                    stats['quality_passed'] += 1
                    
                    if 'windows_result' in results:
                        stats['windowed'] += 1
                        
                        if 'heart_rate_result' in results:
                            stats['heart_rate_analyzed'] += 1
        
        if stats['total_subjects'] > 0:
            stats['processing_success_rate'] = stats['successfully_processed'] / stats['total_subjects']
            
        if stats['successfully_processed'] > 0:
            stats['quality_pass_rate'] = stats['quality_passed'] / stats['successfully_processed']
        
        return stats

def main():
    """Main entry point for the WESAD pipeline."""
    # Load configuration
    config = WESADConfig()
    
    # Initialize and run pipeline
    pipeline = WESADPipeline(config)
    results = pipeline.run_full_pipeline()
    
    print(f"Pipeline completed successfully!")
    print(f"Processed {results['pipeline_statistics']['total_subjects']} subjects")
    print(f"Quality pass rate: {results['pipeline_statistics']['quality_pass_rate']:.1%}")

if __name__ == "__main__":
    main()