#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from models.densenetReg import DenseNetRegV2

Model = DenseNetRegV2

DEBUG = False
if DEBUG:
    NUM_EPOCHS = 2
else:
    NUM_EPOCHS = 45

LEARNING_RATE = 0.01
BATCH_SIZE = 128

tf.set_random_seed(20180607)


if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
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

        # Declare some parameters
        batch_size = BATCH_SIZE

        # Define model
        from datasources import HDF5Source

        model = Model(
            # Tensorflow session
            # Note: The same session must be used for the model and the data sources.
            session,

            # The learning schedule describes in which order which part of the network should be
            # trained and with which learning rate.
            #
            # A standard network would have one entry (dict) in this argument where all model
            # parameters are optimized. To do this, you must specify which variables must be
            # optimized and this is done by specifying which prefixes to look for.
            # The prefixes are defined by using `tf.variable_scope`.
            #
            # The loss terms which can be specified depends on model specifications, specifically
            # the `loss_terms` output of `BaseModel::build_model`.
            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'gaze_mse': ['block_initial', 'block1', 'block2', 'block3', 'regression'],
                    },
                    'metrics': ['gaze_angular'],
                    'learning_rate': LEARNING_RATE,
                },
            ],

            test_losses_or_metrics=['gaze_mse', 'gaze_angular'],

            # Data sources for training and testing.
            train_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                    keys_to_use=['train'],
                    min_after_dequeue=100,
                ),
            },
            test_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                    keys_to_use=['validation'],
                    testing=True,
                ),
            },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=NUM_EPOCHS,
        )
        #
        # # Evaluate for Kaggle submission
        # model.evaluate_for_kaggle(
        #     HDF5Source(
        #         session,
        #         batch_size,
        #         hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
        #         keys_to_use=['test'],
        #         testing=True,
        #     )
        # )

        # Evaluate for Kaggle submission
        # model.evaluate_for_kaggle(
        #     HDF5Source(
        #         session,
        #         batch_size,
        #         hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
        #         keys_to_use=['test'],
        #         testing=True,
        #     )
        # )
        # model.evaluate(
        #     HDF5Source(
        #         session,
        #         batch_size,
        #         hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
        #         keys_to_use=['train'],
        #         testing=True,
        #     ),
        #     'explore/densenetreg_train_pred.csv'
        # )
        model.evaluate(
            HDF5Source(
                session,
                batch_size,
                hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                keys_to_use=['validation'],
                testing=True,
            ),
            '../explore/densenetreg_validation_pred.csv'
        )

