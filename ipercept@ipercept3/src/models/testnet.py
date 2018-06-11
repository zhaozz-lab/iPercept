"""DenseNet architecture."""
from typing import Dict


import tensorflow as tf
#from tensorpack import *
#from tensorpack.tfutils.symbolic_functions import *
#from tensorpack.tfutils.summary import *

from core import BaseDataSource, BaseModel
import util.gaze

data_format = "channels_last"  # Change this to "channels_first" to run on GPU


class TestNet(BaseModel):
    """An implementation of the DenseNet architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y = input_tensors['gaze']

        with tf.variable_scope('testscope'):

            x = tf.layers.conv2d(x, filters=10, kernel_size=3, strides=1,
                             padding='same', data_format=data_format)

            x = tf.nn.relu(x)
            x = tf.layers.batch_normalization(x, name='bn1', training=tf.estimator.ModeKeys.TRAIN == mode)
            x = tf.layers.flatten(x)
            output = tf.layers.dense(x, units=2, name='fc4', activation=None)

        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(output, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(output, y),
            }
        return {'gaze': output}, loss_terms, metrics


