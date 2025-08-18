import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split

def create_train_test(df, nsplits, subject_column, label_column):
    '''
    Creates a train, test split. Each subject is in one of the sets with all his/her data.
    Takes a df, number of splits, names of the subject and label column.
    Returns 2 dataframes with train and test set.
    '''
    X = df.drop(columns=[subject_column, label_column])
    y = df[[label_column]]
    groups = df[subject_column]
    
    gkf = GroupKFold(n_splits=nsplits)
    train_idx, test_idx = next(gkf.split(X, y, groups))

    X_train = X.iloc[train_idx.tolist(), :]
    y_train = y.iloc[train_idx.tolist(), :]
    groups_train = groups.iloc[train_idx.tolist()]
    
    X_test = X.iloc[test_idx.tolist(), :]
    y_test = y.iloc[test_idx.tolist(), :]
    groups_test = groups.iloc[test_idx.tolist()]
    
    res_train = pd.concat([X_train, groups_train, y_train], axis=1)
    res_test = pd.concat([X_test, groups_test, y_test], axis=1)
    
    return res_train, res_test

def split_df(df, subject_column, label_column):
    '''
    Takes a df and splits it into three dfs, containing the features, labels, and groups.
    '''
    X = df.drop(columns=['subject', 'label'])
    y = df[['label']]
    groups = df['subject']
    return X, y, groups

def remove_correlated_features(df, correlation):
    '''
    Takes a df and removed the correlated features with a correlation value equal or higher to the given value.
    Returns the resulting df and a list of the retained features.
    '''
    cor = df.corr(numeric_only=True)
    keep_columns = np.full(cor.shape[0], True)
    for i in range(cor.shape[0] - 1):
        for j in range(i + 1, cor.shape[0] - 1):
            if (np.abs(cor.iloc[i, j]) >= correlation):
                keep_columns[j] = False
    selected_columns = df.columns[keep_columns]
    df_reduced = df[selected_columns]
    
    return df_reduced, selected_columns

def read_data(windowsize, stepsize):
    '''
    Takes the window size and steps size of the calculates features.
    Returns X_train, y_train, groups_train, X_test, y_test, groups_test.
    Creates the dataset if it doesn't exist. Prints out info during creation.
    '''
    
    filename = f'data-input/flirt-wesad-acc-bvp-eda-temp-{windowsize}-{stepsize}.parquet'
    
    # Check if dataset exists, if not create it
    if not os.path.exists(filename):
        print(f"Dataset for window_size={windowsize}, step_size={stepsize} not found.")
        print("Creating dataset... This may take some time.")
        
        # Create the dataset
        create_dataset(windowsize, stepsize)
        print(f"Dataset created and saved as {filename}")
    
    # Load the dataset
    df = pd.read_parquet(filename)
    
    # Remove columns (see EDA)
    columns_to_drop = ['eda_EDA_n_sign_changes',
                       'temp_TEMP_peaks',
                       'acc_y_entropy',
                       'acc_l2_n_sign_changes',
                       'acc_x_entropy',
                       'acc_z_entropy',
                       'temp_l2_n_sign_changes',
                       'bvp_BVP_entropy',
                       'temp_TEMP_n_sign_changes',
                       'temp_l2_peaks',
                       'eda_l2_n_sign_changes']

    df = df.drop(columns=columns_to_drop)
    
    # Split into train and test
    df_train, df_test = create_train_test(df, 5, 'subject', 'label')

    X_train, y_train, groups_train = split_df(df_train, 'subject', 'label')
    X_test, y_test, groups_test = split_df(df_test, 'subject', 'label')

    # Remove correlated features from train
    X_train, selected_features = remove_correlated_features(X_train, 0.8)

    # Remove the same columns from test
    X_test = X_test[selected_features]
    
    # Print info
    print('Window Size: ', windowsize, '  Stepsize: ', stepsize)
    print('train shape: ', X_train.shape)
    print('test shape: ', X_test.shape)
    print('test columns: ', X_train.columns)
    
    # Check train and test set sizes
    print('Percentage train set:', len(y_train)/(len(y_train)+len(y_test)))
    print('Percentage test set:', len(y_test)/(len(y_train)+len(y_test)))

    print('\nClass distribution in train set: \n', y_train['label'].value_counts(normalize=True), '\n')
    print('Class distribution in test set: \n', y_test['label'].value_counts(normalize=True), '\n')
    
    return X_train, y_train, groups_train, X_test, y_test, groups_test

def create_dataset(window_length, window_step_size):
    '''
    Creates a dataset with specified window_length and window_step_size.
    Saves it as a parquet file in the data-input directory.
    '''
    
    # Ensure data-input directory exists
    if not os.path.isdir('data-input'):
        os.makedirs('data-input')
    
    # Load base datasets
    df_acc, df_bvp, df_eda, df_temp = load_base_data()
    
    # Create iterlist
    iterlist = [(i, j)
        for i in df_acc.subject.unique()
        for j in df_acc.label.unique()]
    
    result_dfs = []
    
    # Loop over all subject-label combinations
    for (subject, label) in iterlist:
        df_acc_chunk, df_bvp_chunk, df_eda_chunk, df_temp_chunk = get_all_chunks(
            df_acc, df_bvp, df_eda, df_temp, subject, label)
        
        res_df_chunks = get_features(subject, label, df_acc_chunk, df_bvp_chunk, 
                                   df_eda_chunk, df_temp_chunk, window_length, window_step_size)
        result_dfs.append(res_df_chunks)

    res = pd.concat(result_dfs)
    
    # Save as parquet
    filename = f'data-input/flirt-wesad-acc-bvp-eda-temp-{window_length}-{window_step_size}.parquet'
    res.to_parquet(filename)
    
    return filename

def load_base_data():
    '''
    Loads the base datasets (df_acc, df_bvp, df_eda, df_temp).
    '''
    df_acc = pd.read_parquet('data-input/dataset_wesad_wrist_acc.parquet')
    df_bvp = pd.read_parquet('data-input/dataset_wesad_wrist_bvp.parquet')
    df_eda = pd.read_parquet('data-input/dataset_wesad_wrist_eda.parquet')
    df_temp = pd.read_parquet('data-input/dataset_wesad_wrist_temp.parquet')
    
    return df_acc, df_bvp, df_eda, df_temp

def get_all_chunks(df_acc, df_bvp, df_eda, df_temp, subject, label):
    df_acc_chunk = get_chunks(df_acc, subject, label)
    df_bvp_chunk = get_chunks(df_bvp, subject, label)
    df_eda_chunk = get_chunks(df_eda, subject, label)
    df_temp_chunk = get_chunks(df_temp, subject, label)
    return df_acc_chunk, df_bvp_chunk, df_eda_chunk, df_temp_chunk

def get_chunks(df, subject, label):
    df_chunk = df[df['subject'] == subject]
    df_chunk = df_chunk[df_chunk['label'] == label]
    df_chunk = df_chunk.drop(columns=['session', 'subject', 'label'])
    return df_chunk

# You'll need to import the get_features function from notebook 02
# or implement it based on the FLIRT library usage shown there