import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
import joblib
import utils
import os

def find_optimal_threshold(model, X_val, y_val):
    """Find threshold that maximizes F1-score"""
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, y_proba)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Find threshold with best F1
    best_threshold_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_threshold_idx]
    best_f1 = f1_scores[best_threshold_idx]
    
    return best_threshold, best_f1


def train():
    '''
    Trains the ExtraTreesClassifier model with the training data.
    Dumps the model + optimal threshold to /model/model_with_threshold.joblib
    Stores the test data in /test-data/ as parquet files.
    Evaluates the model and prints performance metrics.
    '''

    # read data
    windowsize = 30
    stepsize = 1
    X_train, y_train, groups_train, X_test, y_test, groups_test = utils.read_data(windowsize, stepsize)

    # store test data
    df_test = X_test.copy()
    df_test['subject'] = groups_test
    df_test['label'] = y_test

    df_test = df_test.reset_index()
    df_test = df_test.set_index(['datetime', 'subject'])

    if not os.path.isdir('../web-app/test-data'):
        os.makedirs('../web-app/test-data')

    df_test_stress = df_test[df_test['label'] == 1]
    df_test_nostress = df_test[df_test['label'] == 0]

    df_test_stress = df_test_stress.drop(columns={'label'}).sample(100)
    df_test_nostress = df_test_nostress.drop(columns={'label'}).sample(100)

    df_test_stress.to_parquet('../web-app/test-data/test-data-stress.parquet', compression='ZSTD')
    df_test_nostress.to_parquet('../web-app/test-data/test-data-nostress.parquet', compression='ZSTD')

    # Define model hyperparameters
    hyperparameters = {
        'bootstrap': False,
        'criterion': "entropy",
        'max_features': 0.8,
        'min_samples_leaf': 4,
        'min_samples_split': 4,
        'n_estimators': 100,
        'random_state': 0,
        'class_weight': {0: 1.0, 1: 4.0}  # 4x weight for stress
    }
    
    print("=== Model Hyperparameters ===")
    for param, value in hyperparameters.items():
        print(f"{param}: {value}")
    print()

    # Train model
    model = ExtraTreesClassifier(**hyperparameters)
    model.fit(X_train, y_train.values.ravel())

    # Find optimal threshold
    optimal_threshold, best_f1_threshold = find_optimal_threshold(model, X_test, y_test)

    # Predictions with default threshold (0.5)
    y_test_predict_default = model.predict(X_test)

    # Predictions with optimal threshold
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_predict_optimal = (y_test_proba >= optimal_threshold).astype(int)

    # Calculate metrics for default
    print("=== Default Threshold (0.5) Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_test_predict_default):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_predict_default):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_predict_default):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_test_predict_default):.4f}")
    print()

    # Calculate metrics for optimal threshold
    print(f"=== Optimal Threshold ({optimal_threshold:.3f}) Metrics ===")
    print(f"Accuracy:  {accuracy_score(y_test, y_test_predict_optimal):.4f}")
    print(f"Precision: {precision_score(y_test, y_test_predict_optimal):.4f}")
    print(f"Recall:    {recall_score(y_test, y_test_predict_optimal):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_test_predict_optimal):.4f}")
    print()

    # Class distribution info
    print("=== Class Distribution in Test Set ===")
    class_distribution = y_test['label'].value_counts(normalize=True)
    print(f"No Stress (0): {class_distribution[0]:.4f}")
    print(f"Stress (1):    {class_distribution[1]:.4f}")
    print()

    print("=== Data Information ===")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples:     {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Window size: {windowsize} seconds")
    print(f"Step size:   {stepsize} second(s)")

    # store model + threshold
    if not os.path.isdir('model'):
        os.makedirs('model')

    model_data = {
        'model': model,
        'optimal_threshold': optimal_threshold,
        'feature_names': list(X_train.columns)
    }

    joblib.dump(model_data, 'model/model_with_threshold.joblib')
    print(f"\nModel and optimal threshold saved to: model/model_with_threshold.joblib")

train()
