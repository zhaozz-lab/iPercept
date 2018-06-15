import pandas as pd

from densebag_utils import load_validation_gaze, sample_random_submission_files, get_average_df_from_files, \
    get_angular_error
from gaze import angular_error
import numpy as np
import os

base_path = "../../outputs/DenseBagValidation/"
model_prefix = "DenseBag_Validation_RS"
file_prefix = "validation_predictions"
B = 21


df_true = load_validation_gaze()
submission_files = sample_random_submission_files(B, base_path, model_prefix, file_prefix )
df_pred = get_average_df_from_files(submission_files)

angular = angular_error(df_true.values, df_pred.values)
print(np.mean(angular))
print(np.std(angular))

df_result = pd.DataFrame({'pitch': df_true['pitch'], 'yaw': df_true['yaw'], 'angular': angular, 'pitch_pred': df_pred['pitch'], 'yaw_pred': df_pred['yaw'],})
print(df_result.head())
df_result.to_csv(os.path.join(base_path, 'soup_plate.csv'))
