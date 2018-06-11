#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from models.densenetfixed import DenseNetFixed as Model
import time
from datasources import HDF5Source
import logging

logger = logging.getLogger(__name__)

def get_model(session, learning_rate, identifier):
    model = Model(
        session,
        learning_schedule=[
            {
                'loss_terms_to_optimize': {
                    'gaze_mse': ['block_initial', 'block1', 'block2', 'block3', 'regression'],
                },
                'metrics': ['gaze_angular'],
                'learning_rate': learning_rate,
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
    # random_seed is used as identifier
    model.model_identifier = identifier
    return model


if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('-B', type=int, help='number of models to train', default=3)
    parser.add_argument('-B_start', type=int, help='Start training from model with index B (e.g. use B=3 if you have already trained three models, i.e. models 0,1 and 2).', default=0)
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
        batch_size = 64
        B = args.B
        B_start = args.B_start
        logger.info("Training models {} to {}...".format(B_start, B))

        for b in range(B_start, B):
            tf.set_random_seed(b)
            model_identifier = "DenseNetBagged_RS{}_{}".format(b, int(time.time()))
            model = get_model(session, 0.01, model_identifier)
            # Train this model for a set number of epochs
            model.train(num_epochs=1)
            model = get_model(session, 0.001, model_identifier)
            model.train(num_epochs=1)
            model = get_model(session, 0.0001, model_identifier)
            model.train(num_epochs=1)
            # Evaluate for Kaggle submission
            model.evaluate_for_kaggle(
                HDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                    keys_to_use=['test'],
                    testing=True,
                )
            )
