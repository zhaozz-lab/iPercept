
import pandas as pd
import numpy as np
import os
import time

base_path = "../../outputs/"
path_out = os.path.join(base_path, "DenseNetFixedBagged")
path_out = os.path.join(path_out, "to_submit_to_kaggle_{}.csv".format(int(time.time())))

files = ['DenseNetFixed_RS1/to_submit_to_kaggle_1528533245.csv',
         'DenseNetFixed_RS2/to_submit_to_kaggle_1528533024.csv',
         'DenseNetFixed_RS3/to_submit_to_kaggle_1528533011.csv',
         'DenseNetFixed_RS4/to_submit_to_kaggle_1528537012.csv',
         'DenseNetFixed_RS5/to_submit_to_kaggle_1528536860.csv',
         'DenseNetFixed_RS6/to_submit_to_kaggle_1528537180.csv',
         'DenseNetFixed_RS7/to_submit_to_kaggle_1528541308.csv',
         'DenseNetFixed_RS8/to_submit_to_kaggle_1528540806.csv',
         'DenseNetFixed_RS9/to_submit_to_kaggle_1528540857.csv'
         ]

def get_dataframes(base_path, files):
    return [pd.read_csv(os.path.join(base_path, file)) for file in files]


def get_average(list_df, column):

    data = {i: list_df[i][column] for i in range(len(list_df))}
    df_all = pd.DataFrame(data)
    return np.mean(df_all, axis=1)


list_df = get_dataframes(base_path, files)

df_result = pd.DataFrame(
    {
        'pitch': get_average(list_df, 'pitch'),
        'yaw': get_average(list_df, 'yaw')
     }
)

df_result.index.name = 'Id'
df_result.to_csv(path_out)
print(df_result.describe())
