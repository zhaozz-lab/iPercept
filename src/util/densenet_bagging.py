
import pandas as pd
import numpy as np
import os
import time
import re

base_path = "../../outputs/DenseNetBagged/"
path_out = os.path.join(base_path, "to_submit_to_kaggle_{}.csv".format(int(time.time())))


def get_latest_submission(folder):
    all_files = os.listdir(folder)
    target_files = [f for f in all_files if f.startswith("to_submit_to_kaggle_")]

    def extract_timestamp(filename):
        result = re.sub('[^0-9]', '', filename)
        return int(result)
    target_files_sorted = sorted(target_files, key=lambda a: extract_timestamp(a))
    print(folder, target_files_sorted)
    return target_files_sorted[-1]


def get_files(base_path, prefix):
    all_folders = os.listdir(os.path.join(base_path))
    target_folders = [f for f in all_folders if f.startswith(prefix)]
    return [os.path.join(os.path.join(base_path, folder),
            get_latest_submission(os.path.join(base_path, folder))) for folder in target_folders]


def get_dataframes(files):
    return [pd.read_csv(file) for file in files]


def get_average(list_df, column):
    data = {i: list_df[i][column] for i in range(len(list_df))}
    df_all = pd.DataFrame(data)
    return np.mean(df_all, axis=1)


prefix = "DenseNetBagged_RS"
files = get_files(base_path, prefix)


list_df = get_dataframes(files)

df_result = pd.DataFrame(
    {
        'pitch': get_average(list_df, 'pitch'),
        'yaw': get_average(list_df, 'yaw')
     }
)

df_result.index.name = 'Id'
df_result.to_csv(path_out)
print(df_result.describe())
