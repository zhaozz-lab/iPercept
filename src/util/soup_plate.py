"""
This script calculates a dataframe containing
- the true pitch and yaw values (in radians) for the validation set.
- the predicted pitch and yaw values (in radians) for a given set of B (see B_list). The models are randomly sampled.
- the angular error

Output file:
outputs/DenseBagValidation/soup_plate.csv
"""


import pandas as pd

from densebag_utils import load_validation_gaze, sample_random_submission_files, get_average_df_from_files
from gaze import angular_error
import numpy as np
import os

base_path = "../../outputs/DenseBagValidation/"
model_prefix = "DenseBag_Validation_RS"
file_prefix = "validation_predictions"
B_list = [3, 7, 11, 15, 21]

for B in B_list:
    df_true = load_validation_gaze()

    submission_files = sample_random_submission_files(B, base_path, model_prefix, file_prefix )
    df_pred = get_average_df_from_files(submission_files)

    angular = angular_error(df_true.values, df_pred.values)
    print('B:', B)
    print(np.mean(angular))
    print(np.std(angular))

    df_result = pd.DataFrame({'pitch': df_true['pitch'], 'yaw': df_true['yaw'], 'angular': angular, 'pitch_pred': df_pred['pitch'], 'yaw_pred': df_pred['yaw'],})

    df_result.to_csv(os.path.join(base_path, 'soup_plate_{}.csv'.format(B)))
