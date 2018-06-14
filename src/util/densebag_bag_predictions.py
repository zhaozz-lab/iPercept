"""
Averages the outputs of all DenseBag models and creates a kaggle submission file in outputs/DenseBag/ folder.
"""

import os
import time

import pandas as pd

from densebag_utils import get_files, get_average, get_dataframes_from_csv_list

base_path = "../../outputs/DenseBag/"
path_out = os.path.join(base_path, "to_submit_to_kaggle_{}.csv".format(int(time.time())))
prefix = "DenseBag_RS"
files = get_files(base_path, prefix)

list_df = get_dataframes_from_csv_list(files)

df_result = pd.DataFrame(
    {
        'pitch': get_average(list_df, 'pitch'),
        'yaw': get_average(list_df, 'yaw')
    }
)

df_result.index.name = 'Id'
df_result.to_csv(path_out)
print("Evaluated for B={}".format(len(list_df)))
print(df_result.describe())
