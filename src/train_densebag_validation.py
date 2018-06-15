#!/usr/bin/env python3
"""
Train the DenseBag eye gaze estimation model
--------------------------------------------


"""
import argparse

import coloredlogs
import tensorflow as tf

from models.densenetfixed import DenseNetFixed as Model
import time
from datasources import BootstrappedHDF5Source, HDF5Source
import logging

logger = logging.getLogger(__name__)


def get_model(session: tf.Session, learning_rate: tf.Variable, identifier: str, random_seed: int):
    """
    Loads the Model given specifications.
    :param session:
    :param learning_rate: tf.Variable holding the float learning rate
    :param identifier: e.g. DenseBag_RS000_1528804911
    :param random_seed: e.g. 12345
    :return: Model
    """
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
            'real': BootstrappedHDF5Source(
                session,
                batch_size,
                hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
                # We use bootstrapping to sample from both training and validation set.
                keys_to_use=['train'],
                min_after_dequeue=100,
                random_seed=random_seed,
                model_identifier=model_identifier
            ),
        },
        test_data={
        #     'real': HDF5Source(
        #         session,
        #         batch_size,
        #         hdf_path='../datasets/MPIIGaze_kaggle_students.h5',
        #         # We use bootstrapping to sample from both training and validation set.
        #         keys_to_use=['validation'],
        #         min_after_dequeue=100,
        #         testing=True
        #     ),
        }
    )
    # We want to use a custom identifier.
    model.model_identifier = identifier
    return model


def update_learning_rate(session: tf.Session, learning_rate_variable: tf.Variable, new_learning_rate: float):
    """
    Runs a tf.Session and updates the current learning rate stored in learning_rate_variable.
    :param session:
    :param learning_rate_variable:
    :param new_learning_rate: e.g. 0.001
    :return: None
    """
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

    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)

    # Declare some parameters
    batch_size = 64
    B = args.B
    B_start = args.B_start

    # Make sure the user did not accidently swap the parameters
    if B_start >= B:
        logger.warning("B_start must be smaller than B. B_start={} is not smaller than B={}!".format(B_start, B))
        exit()

    logger.info("Training models {}, ... , {}".format(B_start, B-1))

    for b in range(B_start, B):
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
            # Use a tf.Variable so that we can adjust its value
            learning_rate = tf.Variable(initial_value=0.01, trainable=False, name="learning_rate")

            logger.info("Training model with index b={}".format(b))

            # For reproduceability we want to set a specific random seed
            tf.set_random_seed(b)

            # This is the name of the folder where we store our results (weights and predictions)
            model_identifier = "DenseBag_Validation_RS{}_{}".format(str(b).zfill(3), int(time.time()))

            model = get_model(session, learning_rate, random_seed=b, identifier=model_identifier)
            # Train the model for 10 epochs using the initial learning rate
            model.train(num_epochs=10)

            # Update learning rate and train some more
            update_learning_rate(session, learning_rate, 0.001)
            model.train(num_epochs=15)

            # Last update of learning rate
            update_learning_rate(session, learning_rate, 0.0001)
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
        # We need to reset the default_graph such that we can train new models that use the same variable names.
        tf.reset_default_graph()




