from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import numpy as np
import pandas as pd
import flirt
from collections import deque
from typing import List, Optional, Dict, Any, Union
import logging
from datetime import datetime
import traceback

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Debug Circular Buffer API", version="1.0.0")

# Required features in exact order (67 total)
REQUIRED_FEATURES = [
    'bvp_BVP_mean', 'bvp_BVP_std', 'bvp_BVP_skewness', 'bvp_BVP_kurtosis',
    'bvp_BVP_peaks', 'bvp_BVP_n_above_mean', 'bvp_BVP_n_below_mean',
    'bvp_BVP_n_sign_changes', 'bvp_BVP_perm_entropy', 'bvp_BVP_svd_entropy',
    'bvp_l2_min', 'bvp_l2_n_above_mean', 'bvp_l2_n_below_mean',
    'bvp_l2_n_sign_changes', 'bvp_l2_perm_entropy', 'acc_x_mean',
    'acc_x_std', 'acc_x_energy', 'acc_x_skewness', 'acc_x_kurtosis',
    'acc_x_peaks', 'acc_x_lineintegral', 'acc_x_n_above_mean',
    'acc_x_n_sign_changes', 'acc_x_iqr', 'acc_y_mean', 'acc_y_std',
    'acc_y_max', 'acc_y_energy', 'acc_y_skewness', 'acc_y_kurtosis',
    'acc_y_peaks', 'acc_y_n_above_mean', 'acc_y_n_sign_changes',
    'acc_y_iqr', 'acc_y_svd_entropy', 'acc_z_mean', 'acc_z_std',
    'acc_z_min', 'acc_z_max', 'acc_z_energy', 'acc_z_skewness',
    'acc_z_kurtosis', 'acc_z_peaks', 'acc_z_n_above_mean',
    'acc_z_n_sign_changes', 'acc_z_svd_entropy', 'acc_l2_skewness',
    'acc_l2_kurtosis', 'acc_l2_n_above_mean', 'acc_l2_n_below_mean',
    'eda_EDA_mean', 'eda_EDA_std', 'eda_EDA_skewness', 'eda_EDA_kurtosis',
    'eda_EDA_peaks', 'eda_EDA_lineintegral', 'eda_EDA_n_above_mean',
    'eda_EDA_n_below_mean', 'eda_EDA_iqr', 'eda_EDA_perm_entropy',
    'eda_EDA_svd_entropy', 'temp_TEMP_mean', 'temp_TEMP_std',
    'temp_TEMP_skewness', 'temp_TEMP_kurtosis', 'temp_TEMP_lineintegral',
    'temp_TEMP_n_above_mean', 'temp_TEMP_n_below_mean', 'temp_TEMP_iqr',
    'temp_TEMP_perm_entropy', 'temp_TEMP_svd_entropy',
    'temp_l2_svd_entropy'
]

WINDOW_SIZE = 30

class SensorData(BaseModel):
    acc_x: Union[float, List[float]]
    acc_y: Union[float, List[float]] 
    acc_z: Union[float, List[float]]
    bvp: Union[float, List[float]]
    eda: Union[float, List[float]]
    temp: Union[float, List[float]]

class CircularBufferManager:
    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size
        self.reset_buffers()
        
    def reset_buffers(self):
        self.acc_x_buffer = deque(maxlen=self.window_size)
        self.acc_y_buffer = deque(maxlen=self.window_size)
        self.acc_z_buffer = deque(maxlen=self.window_size)
        self.bvp_buffer = deque(maxlen=self.window_size)
        self.eda_buffer = deque(maxlen=self.window_size)
        self.temp_buffer = deque(maxlen=self.window_size)
        self.window_count = 0
    
    def extract_features_with_detailed_debug(self) -> Dict[str, Any]:
        """Enhanced debug version with step-by-step error catching"""
        if len(self.acc_x_buffer) < self.window_size:
            return {"error": "insufficient_data", "buffer_size": len(self.acc_x_buffer)}
        
        debug_info = {
            "steps_completed": [],
            "errors": [],
            "feature_shapes": {},
            "available_columns": [],
            "missing_features": [],
            "final_features": None
        }
        
        try:
            # Step 1: Convert buffers to lists
            debug_info["steps_completed"].append("step_1_buffer_conversion")
            acc_x = list(self.acc_x_buffer)
            acc_y = list(self.acc_y_buffer)
            acc_z = list(self.acc_z_buffer)
            bvp = list(self.bvp_buffer)
            eda = list(self.eda_buffer)
            temp = list(self.temp_buffer)
            
            logger.info(f"Step 1 ✅: Buffer conversion complete")
            
            # Step 2: Create dataframes
            debug_info["steps_completed"].append("step_2_dataframe_creation")
            df_acc = pd.DataFrame(list(zip(acc_x, acc_y, acc_z)), columns=['x', 'y', 'z'])
            df_bvp = pd.DataFrame(bvp, columns=['BVP'])
            df_eda = pd.DataFrame(eda, columns=['EDA'])
            df_temp = pd.DataFrame(temp, columns=['TEMP'])
            
            logger.info(f"Step 2 ✅: DataFrames created")
            
            # Step 3: Extract ACC features
            debug_info["steps_completed"].append("step_3_acc_features")
            try:
                acc_features = self._get_features_inner_debug(df_acc, ['x', 'y', 'z'], 'acc_', 30, 1, 32)
                debug_info["feature_shapes"]["acc"] = acc_features.shape
                logger.info(f"Step 3 ✅: ACC features {acc_features.shape}")
            except Exception as e:
                debug_info["errors"].append(f"ACC feature extraction failed: {str(e)}")
                logger.error(f"Step 3 ❌: ACC feature extraction failed: {e}")
                return debug_info
            
            # Step 4: Extract BVP features
            debug_info["steps_completed"].append("step_4_bvp_features")
            try:
                bvp_features = self._get_features_inner_debug(df_bvp, ['BVP'], 'bvp_', 30, 1, 64)
                debug_info["feature_shapes"]["bvp"] = bvp_features.shape
                logger.info(f"Step 4 ✅: BVP features {bvp_features.shape}")
            except Exception as e:
                debug_info["errors"].append(f"BVP feature extraction failed: {str(e)}")
                logger.error(f"Step 4 ❌: BVP feature extraction failed: {e}")
                return debug_info
            
            # Step 5: Extract EDA features
            debug_info["steps_completed"].append("step_5_eda_features")
            try:
                eda_features = self._get_features_inner_debug(df_eda, ['EDA'], 'eda_', 30, 1, 4)
                debug_info["feature_shapes"]["eda"] = eda_features.shape
                logger.info(f"Step 5 ✅: EDA features {eda_features.shape}")
            except Exception as e:
                debug_info["errors"].append(f"EDA feature extraction failed: {str(e)}")
                logger.error(f"Step 5 ❌: EDA feature extraction failed: {e}")
                return debug_info
            
            # Step 6: Extract TEMP features
            debug_info["steps_completed"].append("step_6_temp_features")
            try:
                temp_features = self._get_features_inner_debug(df_temp, ['TEMP'], 'temp_', 30, 1, 4)
                debug_info["feature_shapes"]["temp"] = temp_features.shape
                logger.info(f"Step 6 ✅: TEMP features {temp_features.shape}")
            except Exception as e:
                debug_info["errors"].append(f"TEMP feature extraction failed: {str(e)}")
                logger.error(f"Step 6 ❌: TEMP feature extraction failed: {e}")
                return debug_info
            
            # Step 7: Fix dimensions (take first row if multi-window)
            debug_info["steps_completed"].append("step_7_dimension_fix")
            try:
                if acc_features.shape[0] > 1:
                    acc_features = acc_features.iloc[[0]]
                    logger.info(f"ACC features reduced to {acc_features.shape}")
                
                if bvp_features.shape[0] > 1:
                    bvp_features = bvp_features.iloc[[0]]
                    logger.info(f"BVP features reduced to {bvp_features.shape}")
                    
                if eda_features.shape[0] > 1:
                    eda_features = eda_features.iloc[[0]]
                    logger.info(f"EDA features reduced to {eda_features.shape}")
                    
                if temp_features.shape[0] > 1:
                    temp_features = temp_features.iloc[[0]]
                    logger.info(f"TEMP features reduced to {temp_features.shape}")
                
                logger.info(f"Step 7 ✅: Dimensions fixed")
            except Exception as e:
                debug_info["errors"].append(f"Dimension fixing failed: {str(e)}")
                logger.error(f"Step 7 ❌: Dimension fixing failed: {e}")
                return debug_info
            
            # Step 8: Merge features
            debug_info["steps_completed"].append("step_8_merge_features")
            try:
                res = pd.concat([bvp_features, acc_features, eda_features, temp_features], axis=1)
                debug_info["merged_shape"] = res.shape
                debug_info["available_columns"] = list(res.columns)
                logger.info(f"Step 8 ✅: Features merged to {res.shape}")
                logger.info(f"Available columns: {len(res.columns)} total")
            except Exception as e:
                debug_info["errors"].append(f"Feature merging failed: {str(e)}")
                logger.error(f"Step 8 ❌: Feature merging failed: {e}")
                return debug_info
            
            # Step 9: Check for missing required features
            debug_info["steps_completed"].append("step_9_check_required_features")
            try:
                available_features = set(res.columns)
                required_features = set(REQUIRED_FEATURES)
                missing_features = required_features - available_features
                debug_info["missing_features"] = list(missing_features)
                
                if missing_features:
                    logger.error(f"Step 9 ❌: Missing {len(missing_features)} required features")
                    logger.error(f"Missing features: {list(missing_features)[:10]}...")
                    debug_info["errors"].append(f"Missing {len(missing_features)} required features")
                    return debug_info
                else:
                    logger.info(f"Step 9 ✅: All required features available")
            except Exception as e:
                debug_info["errors"].append(f"Feature checking failed: {str(e)}")
                logger.error(f"Step 9 ❌: Feature checking failed: {e}")
                return debug_info
            
            # Step 10: Filter to required features
            debug_info["steps_completed"].append("step_10_filter_features")
            try:
                res_filtered = res[REQUIRED_FEATURES]
                debug_info["filtered_shape"] = res_filtered.shape
                logger.info(f"Step 10 ✅: Features filtered to {res_filtered.shape}")
            except Exception as e:
                debug_info["errors"].append(f"Feature filtering failed: {str(e)}")
                logger.error(f"Step 10 ❌: Feature filtering failed: {e}")
                return debug_info
            
            # Step 11: Extract feature values
            debug_info["steps_completed"].append("step_11_extract_values")
            try:
                feature_values = res_filtered.iloc[0].to_list()
                debug_info["feature_count"] = len(feature_values)
                
                # Check for NaN
                nan_count = sum(1 for x in feature_values if np.isnan(x))
                debug_info["nan_count"] = nan_count
                
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values, replacing with 0.0")
                    feature_values = [0.0 if np.isnan(x) else x for x in feature_values]
                
                debug_info["final_features"] = feature_values
                self.window_count += 1
                
                logger.info(f"Step 11 ✅: Extracted {len(feature_values)} features successfully")
                
                debug_info["success"] = True
                return debug_info
                
            except Exception as e:
                debug_info["errors"].append(f"Value extraction failed: {str(e)}")
                logger.error(f"Step 11 ❌: Value extraction failed: {e}")
                return debug_info
                
        except Exception as e:
            debug_info["errors"].append(f"Unexpected error: {str(e)}")
            debug_info["traceback"] = traceback.format_exc()
            logger.error(f"Unexpected error in feature extraction: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return debug_info
    
    def _get_features_inner_debug(self, df, columns_list, prefix, window_length, window_step_size, frequency):
        """Debug version of feature extraction"""
        try:
            logger.debug(f"Starting {prefix} feature extraction with frequency {frequency}")
            
            # Set correct datetime index
            ns = '250000000N'
            if frequency == 64:
                ns = '15625000N'
            elif frequency == 32:
                ns = '31250000N'
            
            logger.debug(f"Using time step: {ns}")
            
            time_index = pd.date_range(start=0, periods=len(df), freq=ns)
            df = df.set_index(time_index)
            
            df = df[columns_list]
            df = df.dropna()
            
            logger.debug(f"Input DataFrame shape for {prefix}: {df.shape}")
            logger.debug(f"Input DataFrame head:\n{df.head()}")
            
            features = flirt.get_acc_features(df,
                                              window_length=window_length, 
                                              window_step_size=window_step_size,
                                              data_frequency=frequency)
            
            logger.debug(f"Raw features shape for {prefix}: {features.shape}")
            logger.debug(f"Raw features columns: {list(features.columns)[:5]}...")
            
            features = features.add_prefix(prefix)
            
            logger.debug(f"Final features shape for {prefix}: {features.shape}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error in {prefix} feature extraction: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

buffer_manager = CircularBufferManager()

@app.post("/debug_detailed/")
async def debug_detailed_extraction(sensor_data: SensorData):
    """Detailed debug endpoint with step-by-step error tracking"""
    try:
        # Reset and fill buffer
        buffer_manager.reset_buffers()
        
        if isinstance(sensor_data.acc_x, list):
            for i in range(len(sensor_data.acc_x)):
                buffer_manager.acc_x_buffer.append(sensor_data.acc_x[i])
                buffer_manager.acc_y_buffer.append(sensor_data.acc_y[i])
                buffer_manager.acc_z_buffer.append(sensor_data.acc_z[i])
                buffer_manager.bvp_buffer.append(sensor_data.bvp[i])
                buffer_manager.eda_buffer.append(sensor_data.eda[i])
                buffer_manager.temp_buffer.append(sensor_data.temp[i])
        
        # Run detailed debug extraction
        debug_result = buffer_manager.extract_features_with_detailed_debug()
        
        return {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "buffer_size": len(buffer_manager.acc_x_buffer),
            "debug_info": debug_result
        }
        
    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)