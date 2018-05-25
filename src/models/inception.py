"""Example architecture."""
from typing import Dict

import tensorflow as tf

import util.gaze
from core import BaseDataSource
from models.custom import CustomModel
import logging


data_format = "channels_last"  # Change this to "channels_first" to run on GPU
logger = logging.getLogger(__name__)


class Inception(CustomModel):
    """An example neural network architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""
        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors

        logger.info("input_tensors.keys(): "+str(input_tensors.keys()))  # eye, gaze, head

        x = input_tensors['eye']
        y = input_tensors['gaze']

        with tf.variable_scope('fc'):
            self.summary.histogram('inception/bottleneckactivations', x)

            with tf.variable_scope('dropout0.2'):
                x = tf.layers.dropout(x, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

            # FC layer
            x = tf.layers.dense(x, units=2048, name='fc2048', activation=tf.nn.relu)
            self.summary.histogram('fc/activations2048', x)

            with tf.variable_scope('dropout2048'):
                x = tf.layers.dropout(x, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

            x = tf.layers.dense(x, units=1024, name='fc1024', activation=tf.nn.relu)
            self.summary.histogram('fc/activations1024', x)

            with tf.variable_scope('dropout1024'):
                x = tf.layers.dropout(x, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

            x = tf.layers.dense(x, units=512, name='fc512', activation=tf.nn.relu)
            self.summary.histogram('fc/activations512', x)

            # Directly regress two polar angles for gaze direction
            x = tf.layers.dense(x, units=2, name='fc2')
            self.summary.histogram('fc/activations2', x)

        # Define outputs
        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(x, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(x, y),
            }
        return {'gaze': x}, loss_terms, metrics
