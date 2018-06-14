import pandas as pd
import numpy as np


path_pred = "../outputs/DenseBag_RS011_1528831752/validation_predictions.csv"
path_truth = "../outputs/DenseBag_RS011_1528831752/validation_truth.csv"

df = pd.read_csv(path_pred)
df.columns = ['Id', 'pitch_pred', 'yaw_pred']


df_truth = pd.read_csv(path_truth)
df_truth.columns = ['Id', 'pitch_true', 'yaw_true']

df = df.merge(df_truth)
print(df.head())

print(np.sqrt(np.sum(np.square(df['pitch_true'] - df['pitch_pred']))) / len(df.index))
print(np.sqrt(np.sum(np.square(df['yaw_true'] - df['yaw_pred']))) / len(df.index))

