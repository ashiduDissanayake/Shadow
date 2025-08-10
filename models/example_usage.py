#!/usr/bin/env python3
"""
Example usage of Shadow AI Models

This script demonstrates how to use the new modular AI model structure.
"""

import os
import sys
import numpy as np
import logging

# Add the models directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logging, load_config, get_model_paths
from preprocessing.data_loader import WESADLoader
from preprocessing.bvp_processor import BVPProcessor
from training.hybrid_cnn import create_hybrid_cnn

def main():
    """Main example function."""
    
    # Setup logging
    paths = get_model_paths()
    logger = setup_logging(
        log_level="INFO",
        log_file=os.path.join(paths['logs_dir'], 'example.log')
    )
    
    logger.info("Starting Shadow AI Models example")
    
    # Load configuration
    config_path = os.path.join(paths['configs_dir'], 'model_config.yaml')
    try:
        config = load_config(config_path)
        logger.info("Configuration loaded successfully")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please ensure the configuration file exists")
        return
    
    # Example 1: Data Loading
    logger.info("Example 1: Loading WESAD dataset")
    try:
        # Initialize data loader
        data_loader = WESADLoader(data_path=os.path.join(paths['data_dir'], 'raw', 'wesad'))
        
        # Get dataset information
        dataset_info = data_loader.get_subject_info()
        logger.info(f"Dataset info: {dataset_info}")
        
        # Load BVP data for first few subjects (if available)
        subjects = dataset_info['subjects'][:3] if dataset_info['subjects'] else []
        if subjects:
            bvp_data = data_loader.load_bvp_data(subjects=subjects)
            logger.info(f"Loaded BVP data for {len(bvp_data)} subjects")
        else:
            logger.warning("No subjects found in dataset")
            # Create dummy data for demonstration
            bvp_data = create_dummy_data()
            
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        logger.info("Using dummy data for demonstration")
        bvp_data = create_dummy_data()
    
    # Example 2: BVP Processing
    logger.info("Example 2: Processing BVP signals")
    try:
        # Initialize BVP processor
        processor = BVPProcessor(
            sampling_rate=config['data']['bvp']['sampling_rate'],
            window_size=config['data']['bvp']['window_size'],
            overlap=config['data']['bvp']['overlap'],
            filter_low=config['data']['bvp']['filter_low'],
            filter_high=config['data']['bvp']['filter_high'],
            filter_order=config['data']['bvp']['filter_order']
        )
        
        # Process BVP data
        processed_data = processor.process_batch(
            bvp_data,
            normalize=config['data']['normalization']['fit_on_train'],
            normalize_method=config['data']['normalization']['method']
        )
        
        logger.info(f"Processed data for {len(processed_data)} subjects")
        
        # Show processing info
        segment_info = processor.get_segment_info()
        logger.info(f"Segment info: {segment_info}")
        
    except Exception as e:
        logger.error(f"Error processing BVP data: {str(e)}")
        return
    
    # Example 3: Model Creation
    logger.info("Example 3: Creating Hybrid CNN model")
    try:
        # Create model from configuration
        model = create_hybrid_cnn(config)
        
        # Get model information
        model_info = model.get_model_info()
        logger.info(f"Model created successfully: {model_info}")
        
        # Print model summary
        model_summary = model.get_model_summary()
        logger.info("Model summary:")
        print(model_summary)
        
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        return
    
    # Example 4: Model Training Preparation
    logger.info("Example 4: Preparing for training")
    try:
        # Prepare training data (simplified example)
        if processed_data:
            # Get first subject's data for demonstration
            subject_id = list(processed_data.keys())[0]
            subject_data = processed_data[subject_id]
            
            segments = subject_data['segments']
            labels = subject_data['labels']
            
            logger.info(f"Training data prepared: {len(segments)} segments, {len(labels)} labels")
            logger.info(f"Segment shape: {segments[0].shape if segments else 'No segments'}")
            
            # Show label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            logger.info(f"Label distribution: {dict(zip(unique_labels, counts))}")
            
    except Exception as e:
        logger.error(f"Error preparing training data: {str(e)}")
    
    logger.info("Example completed successfully!")
    logger.info("Next steps:")
    logger.info("1. Add your WESAD dataset to models/data/raw/wesad/")
    logger.info("2. Run the training notebook: models/training/Shadow_AI.ipynb")
    logger.info("3. Check results in models/results/")

def create_dummy_data():
    """Create dummy data for demonstration when real data is not available."""
    dummy_data = {}
    
    # Create dummy data for 3 subjects
    for subject_id in [1, 2, 3]:
        # Generate dummy BVP signal (10 minutes at 64Hz)
        bvp_signal = np.random.randn(10 * 60 * 64)
        
        # Generate dummy labels (random stress states)
        labels = np.random.choice([0, 1, 2, 3], size=len(bvp_signal), p=[0.4, 0.3, 0.2, 0.1])
        
        dummy_data[subject_id] = {
            'bvp': bvp_signal,
            'labels': labels,
            'sampling_rate': 64
        }
    
    return dummy_data

if __name__ == "__main__":
    main()
