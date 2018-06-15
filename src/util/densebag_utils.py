import logging
import os
import re

import h5py
import pandas as pd
import numpy as np
import tensorflow

from gaze import tensorflow_angular_error_from_pitchyaw

logger = logging.getLogger("DenseBagBagger")


def get_latest_submission(folder, submission_prefix):
    """
    Gets latest submission "to_submit_to_kaggle..." from given folder
    :param folder:
    :return: str filename of latest submission
    """
    all_files = os.listdir(folder)
    target_files = [f for f in all_files if f.startswith(submission_prefix)]

    def extract_timestamp(filename):
        result = re.sub('[^0-9]', '', filename)
        try:
            return int(result)
        except ValueError:
            # There is not number (i.e. timestamp) in filename.
            # In this case the order does not matter (there is only one file)
            return 0
    target_files_sorted = sorted(target_files, key=lambda a: extract_timestamp(a))
    if len(target_files_sorted) == 0:
        logger.warning("No submission found in folder {}".format(folder))
        return ""
    return target_files_sorted[-1]


def get_all_submission_files(base_path, model_prefix, file_prefix="to_submit_to_kaggle_"):
    """
    Returns submission files in basepath where model name starts with model_prefix and submission file starts with file_prefix.
    :param base_path: absolute or relative path
    :param model_prefix:
    :return: list of path for all files in folder that start with prefix
    """
    all_folders = os.listdir(os.path.join(base_path))
    target_folders = [f for f in all_folders if f.startswith(model_prefix)]
    return [os.path.join(os.path.join(base_path, folder),
        get_latest_submission(os.path.join(base_path, folder), file_prefix)) for folder in target_folders]


def get_dataframes_from_csv_list(files):
    """
    Opens all files in files and returns them as pd.DataFrame
    :param files:
    :return:
    """
    return [pd.read_csv(file) for file in files]


def get_average(list_df, column):
    """
    :param list_df:
    :param column:
    :return: the average of given column across all DataFrames in list
    """
    data = {i: list_df[i][column] for i in range(len(list_df))}
    df_all = pd.DataFrame(data)
    return np.mean(df_all, axis=1)


def sample_random_submission_files(n: int, base_path, model_prefix, file_prefix):
    all_submission_files = get_all_submission_files(base_path, model_prefix, file_prefix)
    return np.random.choice(all_submission_files, size=n, replace=False)


def get_average_df_from_files(files):
    list_df = get_dataframes_from_csv_list(files)
    return pd.DataFrame(
        {
            'pitch': get_average(list_df, 'pitch'),
            'yaw': get_average(list_df, 'yaw')
        }
    )


def load_validation_gaze():
    file = h5py.File("../../datasets/MPIIGaze_kaggle_students.h5")
    n = file['validation']['gaze'].shape[0]
    pitch = [file['validation']['gaze'][i][0] for i in range(n)]
    yaw = [file['validation']['gaze'][i][1] for i in range(n)]
    return pd.DataFrame({'pitch': pitch, 'yaw': yaw})


def calculate_mse(df_pred, df_true):
    assert len(df_pred.index) == len(df_true.index)

    mse_pitch = np.square(df_pred['pitch'] - df_true['pitch'])
    mse_yaw = np.square(df_pred['yaw'] - df_true['yaw'])
    return np.mean([mse_pitch, mse_yaw])


def get_angular_error(df_pred, df_true):
    a = df_true.values
    b = df_pred.values
    with tensorflow.Session() as sess:
        f = tensorflow_angular_error_from_pitchyaw(a, b)
        return sess.run(f)