"""
Main WESAD Analysis Pipeline

Main pipeline orchestrator that coordinates all components of the WESAD analysis
pipeline including data loading, processing, analysis, visualization, and reporting.

Features:
- Pipeline orchestration and component integration
- Command-line interface with configurable options
- Progress monitoring and comprehensive error handling
- Flexible execution modes (full analysis, specific components)
- Automated report generation and data export

Author: Shadow AI Team
License: MIT
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import traceback
from tqdm import tqdm
import numpy as np

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from wesad_pipeline.config import WESADConfig
from wesad_pipeline.data import WESADDataLoader, WESADPreprocessor
from wesad_pipeline.analysis import SignalQuality, HeartRateAnalyzer, WindowAnalyzer
from wesad_pipeline.visualization import SignalPlotter, WindowPlotter, DatasetPlotter
from wesad_pipeline.utils import WESADHelpers, DocumentationGenerator

class WESADPipeline:
    """
    Main WESAD Analysis Pipeline orchestrator.
    
    Coordinates all pipeline components to provide comprehensive analysis
    of WESAD dataset including signal processing, quality assessment,
    windowing analysis, and visualization.
    """
    
    def __init__(self, 
                 wesad_path: Optional[str] = None,
                 output_path: Optional[str] = None,
                 subjects: Optional[List[int]] = None,
                 config: Optional[WESADConfig] = None,
                 log_level: str = "INFO"):
        """
        Initialize the WESAD Analysis Pipeline.
        
        Args:
            wesad_path: Path to WESAD dataset directory
            output_path: Path for output files and reports
            subjects: List of subject IDs to process (if None, uses config subjects)
            config: Custom configuration object (if None, creates default)
            log_level: Logging level
        """
        # Setup configuration
        if config is None:
            config = WESADConfig()
        
        # Override config parameters if provided
        if wesad_path is not None:
            config.dataset.wesad_path = wesad_path
        if output_path is not None:
            config.output.output_path = output_path
        if subjects is not None:
            config.dataset.subjects = subjects
        
        self.config = config
        
        # Initialize helpers first (needed for logging setup)
        self.helpers = WESADHelpers(config)
        
        # Setup logging
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.config.create_output_directories()
        
        # Initialize components
        self.data_loader = WESADDataLoader(config)
        self.preprocessor = WESADPreprocessor(config)
        self.signal_quality = SignalQuality(config)
        self.heart_rate_analyzer = HeartRateAnalyzer(config)
        self.window_analyzer = WindowAnalyzer(config)
        self.signal_plotter = SignalPlotter(config)
        self.window_plotter = WindowPlotter(config)
        self.dataset_plotter = DatasetPlotter(config)
        self.doc_generator = DocumentationGenerator(config)
        
        # Pipeline state
        self.pipeline_results = {}
        self.pipeline_stats = {
            'subjects_processed': 0,
            'subjects_failed': 0,
            'total_windows': 0,
            'total_duration': 0.0,
            'processing_time': 0.0
        }
        
        self.logger.info(f"WESAD Pipeline initialized")
        self.logger.info(f"WESAD path: {config.dataset.wesad_path}")
        self.logger.info(f"Output path: {config.output.output_path}")
        self.logger.info(f"Target subjects: {len(config.dataset.subjects)}")
    
    def run_analysis(self, 
                    subjects: Optional[List[int]] = None,
                    enable_plots: bool = True,
                    enable_reports: bool = True,
                    enable_export: bool = True) -> Dict:
        """
        Run complete WESAD analysis pipeline.
        
        Args:
            subjects: Specific subjects to process (if None, uses config subjects)
            enable_plots: Whether to generate visualization plots
            enable_reports: Whether to generate analysis reports
            enable_export: Whether to export processed data
            
        Returns:
            Dictionary containing complete analysis results
        """
        try:
            self.logger.info("Starting complete WESAD analysis pipeline")
            
            import time
            start_time = time.time()
            
            # Step 1: Load and preprocess data
            dataset_results = self.run_data_processing(subjects)
            
            if not dataset_results:
                self.logger.error("No data was successfully processed")
                return {}
            
            # Step 2: Run signal analysis
            dataset_results = self.run_signal_analysis(dataset_results)
            
            # Step 3: Run windowing analysis
            dataset_results = self.run_windowing_analysis(dataset_results)
            
            # Step 4: Generate visualizations
            if enable_plots:
                self.generate_visualizations(dataset_results)
            
            # Step 5: Generate reports
            if enable_reports:
                self.generate_reports(dataset_results)
            
            # Step 6: Export data
            if enable_export:
                self.export_data(dataset_results)
            
            # Calculate final statistics
            self.pipeline_stats['processing_time'] = time.time() - start_time
            
            self.logger.info(f"Pipeline completed successfully in {self.pipeline_stats['processing_time']:.2f} seconds")
            self.logger.info(f"Processed {self.pipeline_stats['subjects_processed']} subjects")
            self.logger.info(f"Generated {self.pipeline_stats['total_windows']} windows")
            
            # Store results
            self.pipeline_results = dataset_results
            
            return {
                'results': dataset_results,
                'statistics': self.pipeline_stats,
                'config': self.config.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def run_data_processing(self, subjects: Optional[List[int]] = None) -> Dict[int, Dict]:
        """
        Run data loading and preprocessing steps.
        
        Args:
            subjects: Specific subjects to process
            
        Returns:
            Dictionary mapping subject IDs to processed data
        """
        try:
            self.logger.info("Starting data processing phase")
            
            # Validate and load subjects
            if subjects is None:
                subjects = self.config.dataset.subjects
            
            valid_subjects = self.data_loader.validate_subjects(subjects)
            
            if not valid_subjects:
                self.logger.error("No valid subjects found for processing")
                return {}
            
            # Load raw data
            self.logger.info(f"Loading data for {len(valid_subjects)} subjects")
            raw_data = self.data_loader.load_multiple_subjects(valid_subjects)
            
            if not raw_data:
                self.logger.error("No data was successfully loaded")
                return {}
            
            # Preprocess data
            self.logger.info("Preprocessing loaded data")
            processed_data = {}
            
            with tqdm(raw_data.items(), desc="Preprocessing subjects") as pbar:
                for subject_id, subject_raw_data in pbar:
                    pbar.set_description(f"Preprocessing subject {subject_id}")
                    
                    try:
                        processed_subject_data = self.preprocessor.process_subject_data(subject_raw_data)
                        processed_data[subject_id] = {
                            'processed_data': processed_subject_data,
                            'raw_data': subject_raw_data
                        }
                        self.pipeline_stats['subjects_processed'] += 1
                        
                        # Update duration statistics
                        if 'timestamps' in processed_subject_data:
                            timestamps = processed_subject_data['timestamps']
                            if len(timestamps) > 0:
                                duration = timestamps[-1] - timestamps[0]
                                self.pipeline_stats['total_duration'] += duration
                        
                    except Exception as e:
                        self.logger.error(f"Failed to preprocess subject {subject_id}: {str(e)}")
                        self.pipeline_stats['subjects_failed'] += 1
                        continue
            
            self.logger.info(f"Data processing completed: {len(processed_data)} subjects")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {str(e)}")
            return {}
    
    def run_signal_analysis(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run signal quality assessment and heart rate analysis.
        
        Args:
            dataset_results: Results from data processing
            
        Returns:
            Updated dataset results with signal analysis
        """
        try:
            self.logger.info("Starting signal analysis phase")
            
            for subject_id, results in tqdm(dataset_results.items(), desc="Signal analysis"):
                if 'processed_data' not in results:
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                
                if len(bvp_signal) == 0:
                    self.logger.warning(f"No BVP data for subject {subject_id}")
                    continue
                
                try:
                    # Signal quality assessment
                    if self.config.analysis.enable_quality_assessment:
                        quality_result = self.signal_quality.assess_signal_quality(bvp_signal)
                        results['signal_quality_result'] = quality_result
                    
                    # Heart rate analysis
                    heart_rate_result = self.heart_rate_analyzer.estimate_heart_rate(bvp_signal)
                    results['heart_rate_result'] = heart_rate_result
                    
                except Exception as e:
                    self.logger.error(f"Signal analysis failed for subject {subject_id}: {str(e)}")
                    continue
            
            self.logger.info("Signal analysis completed")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Signal analysis phase failed: {str(e)}")
            return dataset_results
    
    def run_windowing_analysis(self, dataset_results: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Run windowing analysis and feature extraction.
        
        Args:
            dataset_results: Results from signal analysis
            
        Returns:
            Updated dataset results with windowing analysis
        """
        try:
            self.logger.info("Starting windowing analysis phase")
            
            for subject_id, results in tqdm(dataset_results.items(), desc="Windowing analysis"):
                if 'processed_data' not in results:
                    continue
                
                processed_data = results['processed_data']
                bvp_signal = processed_data.get('bvp', np.array([]))
                labels = processed_data.get('labels', np.array([]))
                timestamps = processed_data.get('timestamps', np.array([]))
                
                if len(bvp_signal) == 0 or len(labels) == 0:
                    self.logger.warning(f"Insufficient data for windowing analysis: subject {subject_id}")
                    continue
                
                try:
                    # Create windows
                    windowing_result = self.window_analyzer.create_windows(
                        bvp_signal, labels, timestamps
                    )
                    results['windowing_result'] = windowing_result
                    
                    # Update statistics
                    metadata = windowing_result.get('metadata', {})
                    self.pipeline_stats['total_windows'] += metadata.get('accepted_windows', 0)
                    
                    # Extract features if requested
                    if self.config.analysis.enable_time_domain or self.config.analysis.enable_frequency_domain:
                        features_result = self.window_analyzer.extract_window_features(windowing_result)
                        results['features_result'] = features_result
                    
                except Exception as e:
                    self.logger.error(f"Windowing analysis failed for subject {subject_id}: {str(e)}")
                    continue
            
            self.logger.info("Windowing analysis completed")
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Windowing analysis phase failed: {str(e)}")
            return dataset_results
    
    def generate_visualizations(self, dataset_results: Dict[int, Dict]) -> None:
        """
        Generate comprehensive visualizations.
        
        Args:
            dataset_results: Complete analysis results
        """
        try:
            self.logger.info("Generating visualizations")
            
            # Subject-specific plots
            for subject_id, results in tqdm(dataset_results.items(), desc="Subject plots"):
                if 'processed_data' not in results:
                    continue
                
                processed_data = results['processed_data']
                
                try:
                    # Subject overview plot
                    fig = self.signal_plotter.plot_subject_overview(
                        processed_data, subject_id, 
                        save_name=f"subject_{subject_id}_overview"
                    )
                    
                    # Windowing visualization
                    if 'windowing_result' in results:
                        windowing_result = results['windowing_result']
                        
                        # Window creation plot
                        fig = self.window_plotter.plot_window_creation(
                            processed_data['bvp'], processed_data['labels'],
                            windowing_result, processed_data.get('timestamps'),
                            subject_id=subject_id,
                            save_name=f"subject_{subject_id}_windowing"
                        )
                        
                        # Window distributions
                        fig = self.window_plotter.plot_window_distributions(
                            windowing_result, subject_id=subject_id,
                            save_name=f"subject_{subject_id}_window_distributions"
                        )
                
                except Exception as e:
                    self.logger.error(f"Failed to generate plots for subject {subject_id}: {str(e)}")
                    continue
            
            # Dataset-wide plots
            try:
                # Dataset overview
                fig = self.dataset_plotter.plot_dataset_overview(
                    dataset_results, save_name="dataset_overview"
                )
                
                # Subject comparisons
                for metric in ['quality', 'heart_rate', 'windows_count']:
                    fig = self.dataset_plotter.plot_subject_comparison(
                        dataset_results, metric=metric,
                        save_name=f"subject_comparison_{metric}"
                    )
                
                # Condition analysis
                fig = self.dataset_plotter.plot_condition_analysis(
                    dataset_results, save_name="condition_analysis"
                )
            
            except Exception as e:
                self.logger.error(f"Failed to generate dataset plots: {str(e)}")
            
            self.logger.info("Visualization generation completed")
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")
    
    def generate_reports(self, dataset_results: Dict[int, Dict]) -> None:
        """
        Generate comprehensive analysis reports.
        
        Args:
            dataset_results: Complete analysis results
        """
        try:
            self.logger.info("Generating analysis reports")
            
            # Subject reports
            for subject_id, results in tqdm(dataset_results.items(), desc="Subject reports"):
                try:
                    self.doc_generator.generate_subject_report(subject_id, results)
                except Exception as e:
                    self.logger.error(f"Failed to generate report for subject {subject_id}: {str(e)}")
            
            # Dataset report
            try:
                self.doc_generator.generate_dataset_report(dataset_results)
            except Exception as e:
                self.logger.error(f"Failed to generate dataset report: {str(e)}")
            
            # Summary report
            try:
                summary = self.doc_generator.create_analysis_summary(dataset_results)
                summary_path = self.helpers.ensure_output_directory("reports") / "analysis_summary.txt"
                with open(summary_path, 'w') as f:
                    f.write(summary)
                self.logger.info(f"Analysis summary saved to {summary_path}")
            except Exception as e:
                self.logger.error(f"Failed to generate summary report: {str(e)}")
            
            self.logger.info("Report generation completed")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
    
    def export_data(self, dataset_results: Dict[int, Dict]) -> None:
        """
        Export processed data in multiple formats.
        
        Args:
            dataset_results: Complete analysis results
        """
        try:
            self.logger.info("Exporting processed data")
            
            exported_files = self.doc_generator.export_processed_data(dataset_results)
            
            total_files = sum(len(files) for files in exported_files.values())
            self.logger.info(f"Data export completed: {total_files} files generated")
            
            for format, files in exported_files.items():
                if files:
                    self.logger.info(f"{format.upper()}: {len(files)} files")
            
        except Exception as e:
            self.logger.error(f"Data export failed: {str(e)}")
    
    def get_pipeline_statistics(self) -> Dict:
        """Get comprehensive pipeline statistics."""
        # Combine pipeline stats with component stats
        stats = self.pipeline_stats.copy()
        
        stats['component_statistics'] = {
            'data_loader': self.data_loader.get_dataset_statistics(),
            'preprocessor': self.preprocessor.get_processing_statistics(),
            'signal_quality': self.signal_quality.get_quality_statistics(),
            'heart_rate_analyzer': self.heart_rate_analyzer.get_hr_statistics(),
            'window_analyzer': self.window_analyzer.get_windowing_statistics()
        }
        
        return stats
    
    def _setup_logging(self, log_level: str) -> None:
        """Setup logging configuration."""
        # Create logs directory
        logs_dir = Path(self.config.output.output_path) / self.config.output.logs_dir
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = logs_dir / f"wesad_pipeline_{self.helpers.create_timestamp_string()}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WESAD Analysis Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--wesad-path', 
        type=str, 
        default="data/raw/wesad/",
        help="Path to WESAD dataset directory"
    )
    
    parser.add_argument(
        '--output-path', 
        type=str, 
        default="wesad_analysis/",
        help="Output directory for results"
    )
    
    parser.add_argument(
        '--subjects', 
        type=int, 
        nargs='+',
        help="Specific subject IDs to process (e.g., 2 3 4 5)"
    )
    
    parser.add_argument(
        '--log-level', 
        type=str, 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help="Disable visualization generation"
    )
    
    parser.add_argument(
        '--no-reports', 
        action='store_true',
        help="Disable report generation"
    )
    
    parser.add_argument(
        '--no-export', 
        action='store_true',
        help="Disable data export"
    )
    
    parser.add_argument(
        '--window-size', 
        type=int, 
        default=60,
        help="Window size in seconds"
    )
    
    parser.add_argument(
        '--overlap', 
        type=int, 
        default=5,
        help="Window overlap in seconds"
    )
    
    parser.add_argument(
        '--quality-threshold', 
        type=float, 
        default=0.6,
        help="Minimum quality threshold for windows"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the WESAD Analysis Pipeline."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Create custom configuration if needed
        config = WESADConfig()
        
        # Update configuration with command line arguments
        if args.window_size != 60:
            config.analysis.window_size_seconds = args.window_size
        if args.overlap != 5:
            config.analysis.overlap_seconds = args.overlap
        if args.quality_threshold != 0.6:
            config.analysis.quality_threshold = args.quality_threshold
        
        # Initialize pipeline
        pipeline = WESADPipeline(
            wesad_path=args.wesad_path,
            output_path=args.output_path,
            subjects=args.subjects,
            config=config,
            log_level=args.log_level
        )
        
        # Run analysis
        results = pipeline.run_analysis(
            enable_plots=not args.no_plots,
            enable_reports=not args.no_reports,
            enable_export=not args.no_export
        )
        
        if results:
            print("\n" + "="*60)
            print("WESAD ANALYSIS PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            
            stats = results['statistics']
            print(f"Subjects processed: {stats['subjects_processed']}")
            print(f"Total windows generated: {stats['total_windows']}")
            print(f"Total signal duration: {stats['total_duration']:.1f} seconds")
            print(f"Processing time: {stats['processing_time']:.2f} seconds")
            print(f"Output directory: {args.output_path}")
            
        else:
            print("\nPipeline execution failed. Check logs for details.")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()