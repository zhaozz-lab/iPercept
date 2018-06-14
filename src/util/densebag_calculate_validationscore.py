#!/usr/bin/env python3
"""Main script for training a model for gaze estimation."""
import argparse

import coloredlogs
import os

import re
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


def get_trained_model_identifiers(path, prefix="DenseBag_RS"):
    model_folders = [f for f in os.listdir(path) if f.startswith(prefix)]
    return model_folders


if __name__ == '__main__':
    # Set global log level
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('-i', type=int, help='until model index i', default=3)
    parser.add_argument('-i_start', type=int, help='Validate from model index (e.g. use i_start=5 if you want to start with model 005).', default=0)
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
    to_index = args.i
    from_index = args.i_start
    if from_index >= to_index:
        logger.warning("B_start must be smaller than B. B_start={} is not smaller than B={}!".format(from_index, to_index))
        exit()

    trained_model_identifiers = get_trained_model_identifiers("../outputs/")
    trained_model_identifiers = sorted(trained_model_identifiers, key=lambda identifier: int(re.sub("[^0-9]", "", identifier)[:3]))

    logger.info("Evaluating models {}".format(", ".join(trained_model_identifiers)))
    for i, model_identifier in enumerate(trained_model_identifiers):
        logger.info("Evaluating model {}/{} (identifier: {})".format(i, len(trained_model_identifiers), model_identifier))

        path_out = '../outputs/{}/validation_predictions.csv'.format(model_identifier)
        if not os.path.exists(path_out):

            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
                # This is the name of the folder where we store our results (weights and predictions)
                # model_identifier = "DenseBag_RS026_1528958733"

                model = get_model(session, 0.01, random_seed=1, identifier=model_identifier)

                # path_out = "../outputs/DenseBag/{}/pred_validation.csv"

                # this will load the saved weights
                model.train(num_epochs=0)

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
            # We need to reset the default_graph such that we can train new models that use the same variable names.
            tf.reset_default_graph()
            # break

        else:
            logger.info("Already done: "+model_identifier)




