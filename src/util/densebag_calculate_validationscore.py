#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import tensorflow as tf

from models.densenetfixed import DenseNetFixed as Model
import time
from datasources import ValidationSetHDF5Source, HDF5Source
import logging

logger = logging.getLogger(__name__)


def get_model(session, learning_rate, identifier, random_seed):
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
                # We use bootstrapping to sample from both training and validation set.
                keys_to_use=['train', 'validation'],
                min_after_dequeue=100,
            ),
        },
        test_data={}, # we don't use a validation set any more
    )
    model.model_identifier = identifier
    return model


def update_learning_rate(session, learning_rate_variable, new_learning_rate):
    assign_op = learning_rate_variable.assign(new_learning_rate)
    session.run(assign_op)


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

    # Declare some parameters
    batch_size = 64
    B = args.B
    B_start = args.B_start
    if B_start >= B:
        logger.warning("B_start must be smaller than B. B_start={} is not smaller than B={}!".format(B_start, B))
        exit()

    logger.info("Evaluating models {}, ... , {}".format(B_start, B-1))

    B_start = 0
    B = 1
    for b in range(B_start, B):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # This is the name of the folder where we store our results (weights and predictions)
            model_identifier = "DenseBag_RS000_1528804911"
            model = get_model(session, 0.01, random_seed=b, identifier=model_identifier)

            path_out = "../outputs/DenseBag/{}/pred_validation.csv"

            # Evaluate for Kaggle submission
            pred_validationsset = model.evaluate_validationset(
                ValidationSetHDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                    keys_to_use=['train', 'validation'],
                    testing=True,
                    model_identifier=model_identifier
                )
            )
            print(pred_validationsset.head())
        # We need to reset the default_graph such that we can train new models that use the same variable names.
        tf.reset_default_graph()




