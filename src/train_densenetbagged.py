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

    # Declare some parameters
    batch_size = 64
    B = args.B
    B_start = args.B_start
    logger.info("Training models {}, ... , {}".format(B_start, B-1))

    for b in range(B_start, B):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Use a tf.Variable so that we can adjust its value
            learning_rate = tf.Variable(initial_value=0.01, trainable=False, name="learning_rate")

            logger.info("Training model with index b={}".format(b))

            # For reproduceability we want to set a specific random seed
            tf.set_random_seed(b)

            # This is the name of the folder where we store our results (weights and predictions)
            model_identifier = "DenseNetBagged_RS{}_{}".format(str(b).zfill(3), int(time.time()))

            model = get_model(session, learning_rate, model_identifier)
            # Train the model for 10 epochs using the initial learning rate
            model.train(num_epochs=10)

            # Update learning rate and train some more
            assign_op = learning_rate.assign(0.001)
            session.run(assign_op)
            model.train(num_epochs=15)

            # Last update of learning rate
            assign_op = learning_rate.assign(0.0001)
            session.run(assign_op)
            model.train(num_epochs=20)

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
        tf.reset_default_graph()




