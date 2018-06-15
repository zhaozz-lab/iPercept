"""
Averages the outputs of all DenseBag models and creates a kaggle submission file in outputs/DenseBag/ folder.
"""
import argparse
import os
import time

import pandas as pd

from densebag_utils import get_all_submission_files, get_average, get_dataframes_from_csv_list, get_average_df_from_files, sample_random_submission_files

base_path = "../../outputs/DenseBag/"
prefix = "DenseBag_RS"


def success_message(files, df_result, path_out):
    print("Evaluated for B={}".format(len(files)))
    print(df_result.describe())
    print("Written to file {}".format(path_out))


def bag_random_sample(sample_size, file_prefix):
    path_out = os.path.join(base_path, "random_sample_B_{}_{}.csv".format(sample_size, int(time.time())))
    files = sample_random_submission_files(sample_size, base_path, prefix, file_prefix)
    df_result = get_average_df_from_files(files)
    df_result.index.name = 'Id'
    df_result.to_csv(path_out)
    success_message(files, df_result, path_out)



def bag_all():
    files = get_all_submission_files(base_path, prefix)
    df_result = get_average_df_from_files(files)
    df_result.index.name = 'Id'

    path_out = os.path.join(base_path, "to_submit_to_kaggle_B_{}_{}.csv".format(len(files), int(time.time())))
    df_result.to_csv(path_out)
    success_message(files, df_result, path_out)


if __name__ == '__main__':
    # Set global log level
    parser = argparse.ArgumentParser(description='Average prediction using trained models (bagging).')
    parser.add_argument('--sample_size', type=int, help='Number of random samples')
    args = parser.parse_args()

    if args.sample_size:
        bag_random_sample(args.sample_size)
    else:
        bag_all()
