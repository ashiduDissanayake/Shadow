#!/usr/bin/env python3
"""
Offline utility: derive empirical feature means and stds from WESAD-like parquet dataset
so embedded feature scaling (feature_means / feature_scales in simple_mlp.c) can be
re-validated or regenerated when feature definitions change.

Usage:
  python tools/extract_feature_stats.py ../flirt-wesad-acc-bvp-eda-temp-60-10.parquet

Outputs:
  JSON with per-feature mean/std & optional min/max written to stdout and a .json file.

NOTE: This assumes parquet columns matching raw sensor channel names used in firmware.
"""
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

FEATURE_ORDER = [
    # BVP block (example placeholders matching current extractor semantics)
    'BVP_MEAN','BVP_STD','BVP_MIN','BVP_MAX','BVP_MEDIAN','BVP_RANGE','BVP_IQR','BVP_ENERGY',
    # ACC X/Y/Z (5 each)
    'ACC_X_MEAN','ACC_X_STD','ACC_X_MIN','ACC_X_MAX','ACC_X_ENERGY',
    'ACC_Y_MEAN','ACC_Y_STD','ACC_Y_MIN','ACC_Y_MAX','ACC_Y_ENERGY',
    'ACC_Z_MEAN','ACC_Z_STD','ACC_Z_MIN','ACC_Z_MAX','ACC_Z_ENERGY',
    # EDA
    'EDA_MEAN','EDA_STD','EDA_MIN','EDA_MAX',
    # TEMP
    'TEMP_MEAN','TEMP_STD','TEMP_RANGE'
]

# Placeholder mapping from raw sensors to feature computations; replicate embedded logic
SENSOR_COLUMNS = {
    'bvp':'bvp',
    'acc_x':'acc_x',
    'acc_y':'acc_y',
    'acc_z':'acc_z',
    'eda':'eda',
    'temp':'temp'
}

WINDOW_SECONDS = 60
STEP_SECONDS = 10
RATES = { 'bvp':64,'acc_x':32,'acc_y':32,'acc_z':32,'eda':4,'temp':4 }


def compute_basic_stats(arr: np.ndarray):
    mean = arr.mean()
    std = arr.std(ddof=0)
    min_v = arr.min()
    max_v = arr.max()
    median = np.median(arr)
    q1 = np.quantile(arr,0.25)
    q3 = np.quantile(arr,0.75)
    iqr = q3 - q1
    rng = max_v - min_v
    energy = np.mean(arr*arr)
    return dict(mean=mean,std=std,min=min_v,max=max_v,median=median,iqr=iqr,range=rng,energy=energy)

def slice_windows(df: pd.DataFrame):
    # assumes monotonic index or time column; just chunk sequentially per full windows then slide
    total_rows = len(df)
    # derive samples per sensor for 60s window by rate * 60; but dataframe expected pre-aligned per sensor sampling? If stored row-wise synchronized, we simplify.
    # For offline approximation we aggregate consecutively for WINDOW_SECONDS then step STEP_SECONDS.
    window_size = WINDOW_SECONDS * RATES['bvp']  # approximate using highest rate timeline
    step_size = STEP_SECONDS * RATES['bvp']
    for start in range(0, total_rows - window_size + 1, step_size):
        yield df.iloc[start:start+window_size]

def features_from_window(win: pd.DataFrame):
    feats = {}
    # BVP stats
    b = win[SENSOR_COLUMNS['bvp']].to_numpy()
    bs = compute_basic_stats(b)
    feats.update({
        'BVP_MEAN':bs['mean'],'BVP_STD':bs['std'],'BVP_MIN':bs['min'],'BVP_MAX':bs['max'],
        'BVP_MEDIAN':bs['median'],'BVP_RANGE':bs['range'],'BVP_IQR':bs['iqr'],'BVP_ENERGY':bs['energy']})
    for axis,label in [('acc_x','ACC_X'),('acc_y','ACC_Y'),('acc_z','ACC_Z')]:
        a = win[SENSOR_COLUMNS[axis]].to_numpy()
        s = compute_basic_stats(a)
        feats[f'{label}_MEAN']=s['mean']
        feats[f'{label}_STD']=s['std']
        feats[f'{label}_MIN']=s['min']
        feats[f'{label}_MAX']=s['max']
        feats[f'{label}_ENERGY']=s['energy']
    e = win[SENSOR_COLUMNS['eda']].to_numpy(); es = compute_basic_stats(e)
    feats.update({'EDA_MEAN':es['mean'],'EDA_STD':es['std'],'EDA_MIN':es['min'],'EDA_MAX':es['max']})
    t = win[SENSOR_COLUMNS['temp']].to_numpy(); ts = compute_basic_stats(t)
    feats.update({'TEMP_MEAN':ts['mean'],'TEMP_STD':ts['std'],'TEMP_RANGE':ts['range']})
    return feats

def main():
    if len(sys.argv) < 2:
        print('Usage: extract_feature_stats.py <parquet_file>')
        sys.exit(1)
    path = Path(sys.argv[1])
    df = pd.read_parquet(path)
    feat_rows = []
    for win in slice_windows(df):
        feat_rows.append(features_from_window(win))
    feats_df = pd.DataFrame(feat_rows)
    stats = {}
    for col in FEATURE_ORDER:
        series = feats_df[col]
        stats[col] = {
            'mean': float(series.mean()),
            'std': float(series.std(ddof=0)),
            'min': float(series.min()),
            'max': float(series.max())
        }
    out_path = path.with_suffix('.feature_stats.json')
    with open(out_path,'w') as f:
        json.dump({'order':FEATURE_ORDER,'stats':stats},f,indent=2)
    json.dump({'order':FEATURE_ORDER,'stats':stats},sys.stdout,indent=2)

if __name__ == '__main__':
    main()
