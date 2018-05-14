#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from datasources.hdf5 import HDF5SourceRaw
from models.inception import Inception
from util.train_model import get_model


DEBUG = False
if DEBUG:
    NUM_EPOCHS = 2
else:
    NUM_EPOCHS = 100

# Declare some parameters
batch_size = 128
learning_rate = 4e-4
hdf_path = '../datasets/MPIIGaze_kaggle_students_bottleneck.h5'



if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Inception2')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
        model = get_model(Inception, session, learning_rate, batch_size, hdf_path=hdf_path, Hdf5Source=HDF5SourceRaw)

        # Train this model for a set number of epochs
        model.train(
            num_epochs=NUM_EPOCHS,
        )

        # Evaluate for Kaggle submission
        model.evaluate_for_kaggle(
            HDF5SourceRaw(
                session,
                batch_size,
                hdf_path=hdf_path,
                keys_to_use=['test'],
                testing=True,
            )
        )
