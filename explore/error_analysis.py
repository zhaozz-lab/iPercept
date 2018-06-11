import h5py
import pandas as pd
import numpy as np

hdf_path = "../datasets/MPIIGaze_kaggle_students.h5"
hdf5 = h5py.File(hdf_path, 'r')
validation_true = hdf5["validation"]["gaze"]

validation_pred = pd.read_csv('../outputs/DenseNetReg/predictions_validationset.csv')

df = pd.DataFrame(validation_true.value, columns=['pitch_true', 'yaw_true'])
df['pitch_pred'] = validation_pred['pitch']
df['yaw_pred'] = validation_pred['yaw']

df['pitch_diff'] = df['pitch_pred'] - df['pitch_true']
df['yaw_diff'] = df['yaw_pred'] - df['yaw_true']


print(df.head())

print(np.sqrt(np.sum(np.square(df['pitch_diff']))))

