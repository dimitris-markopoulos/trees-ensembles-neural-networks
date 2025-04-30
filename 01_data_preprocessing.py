# ===== import libraries ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json

columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class'
]

df_tr = pd.read_csv('split-data/original/adult.data', names=columns).replace(' ?', np.nan)
df_ts = pd.read_csv('split-data/original/adult.test', names=columns, skiprows=1).replace(' ?', np.nan)

print(f'Training Data Dimensions: {df_tr.shape} >> (missing = {(df_tr.isna().sum().sum()/(df_tr.shape[0] * df_tr.shape[1]))*100:.3f}%)')
print(f'Testing Data Dimensions: {df_ts.shape} >> (missing = {(df_ts.isna().sum().sum()/(df_ts.shape[0] * df_ts.shape[1]))*100:.3f}%)')

df_tr['set'] = 'train'
df_ts['set'] = 'test'
full_data = pd.concat([df_tr, df_ts]).reset_index(drop=True)
full_data['target'] = full_data['class'].apply(lambda x: 1 if x in [' <=50K', ' <=50K.'] else 0)

# ===== viz and clean missing data ======
perc_missing_df = pd.DataFrame(full_data.isna().sum() / full_data.shape[0]).apply(lambda x: round(100*x,2)).rename(columns={0:'percent_missing'})
nonzero_missing = perc_missing_df[perc_missing_df['percent_missing'] > 0]

os.makedirs('media', exist_ok=True)
def plot():
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    plt.figure(figsize=(8, 5))
    bars = plt.bar(nonzero_missing.index, nonzero_missing['percent_missing'], color=colors)
    plt.ylabel('Percent Missing (%)')
    plt.title('Missing Data by Column')
    plt.ylim(0, 10)
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('media/01_viz_missing_data');

# Want to drop all observations with missing data!
full_data_clean = full_data.dropna(axis=0).reset_index(drop=True)
dropped_n = full_data.shape[0] - full_data_clean.shape[0]
dropped_perc = dropped_n / full_data.shape[0]
print(f'Dropped observations: {dropped_n}\nObservations dropped: {dropped_perc*100:.3f}%')

# ===== one-hot encode ======
def one_hot_encode(raw_data, col_name_to_encode):
    n_obs = raw_data.shape[0]
    cols = sorted([f'{col_name_to_encode.upper()}_{col}' for col in raw_data[col_name_to_encode].unique()])
    encoded_df = pd.DataFrame(np.zeros(shape=(n_obs, len(cols))), columns=cols)
    for i in range(n_obs):
        value = raw_data.at[i, col_name_to_encode]
        col_name = f'{col_name_to_encode.upper()}_{value}'
        encoded_df.at[i, col_name] = 1
        
    return encoded_df

engineered_df_d = {}

# Define relevant columns to one-hot encode
list_col_names_to_encode = [
'age', 'workclass', 'education', 'marital-status', 'occupation', 
'relationship', 'race', 'sex','native-country'
]

# Define columns that are not to be one-hot encoded
valid_cols = [
'fnlwgt', 'education-num', 'capital-gain',
'capital-loss', 'hours-per-week', 'target', 'set'
]

assert len(list_col_names_to_encode) + len(valid_cols) == len(full_data_clean.columns) - 1, 'not true' # - 1 -> target

# Loop function over relevant columns
for col in list_col_names_to_encode:
    params = {'raw_data':full_data_clean, 'col_name_to_encode':col}
    engineered_df = one_hot_encode(**params)
    engineered_df_d[col] = engineered_df # Store df in dictionary to combine later

engineered_df = pd.concat(list(engineered_df_d.values()),axis=1)

# ===== split into tr, ts ======
df = pd.concat([engineered_df,full_data_clean[valid_cols]],axis=1)
X = df.drop(columns='target')
y = df['target']
label_map = {0: [' <=50K', ' <=50K.'], 1: [' >50K', ' >50K.']}
y = y.apply(lambda x: 0 if x == '<=50K' else 1).reset_index(drop=True)
os.makedirs('split-data/processed', exist_ok=True)
with open('split-data/processed/label_map.json', 'w') as f:
    json.dump(label_map, f)
bool_musk = (df['set'] == 'train').values
X_tr, X_ts = X.loc[bool_musk].drop('set', axis=1), X.loc[~bool_musk].drop('set', axis=1)
y_tr, y_ts = y[bool_musk], y[~bool_musk]

X_tr.to_csv('split-data/processed/X_tr.csv',index=False)
X_ts.to_csv('split-data/processed/X_ts.csv',index=False)
y_tr.to_csv('split-data/processed/y_tr.csv',index=False)
y_ts.to_csv('split-data/processed/y_ts.csv',index=False)

print('Succesfully Run')