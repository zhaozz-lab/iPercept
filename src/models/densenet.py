"""DenseNet architecture."""
from typing import Dict


import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

from core import BaseDataSource, BaseModel
import util.gaze

data_format = "channels_last"  # Change this to "channels_first" to run on GPU

class DenseNet(BaseModel):
    """An implementation of the DenseNet architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""

        # Hardcoded for now
        #self.N = int((depth - 4) / 3)
        self.N = 40
        self.growthRate = 12

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y = input_tensors['gaze']

        def conv(name, l, channel, stride):
            # added data_format (from examplenet)
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)),
                          data_format=data_format)

        def add_layer(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                #c = BatchNorm('bn1', l)
                c = tf.layers.batch_normalization(l, name='bn1')
                c = tf.nn.relu(c)
                c = conv('conv1', c, self.growthRate, 1)
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                #l = BatchNorm('bn1', l)
                l = tf.layers.batch_normalization(l, name='bn1')
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu, data_format=data_format)
                l = AvgPooling('pool', l, 2)
            return l

        def dense_net(name):
            l = conv('conv0', x, 16, 1)

            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)

            with tf.variable_scope('regression'):
                #l = BatchNorm('bnlast', l)
                l = tf.layers.batch_normalization(l, name='bnlast')
                l = tf.nn.relu(l)
                l = GlobalAvgPooling('gap', l)
                logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
                self.summary.histogram('regression/logits', logits)

            return logits

        output = dense_net('dense_net')

        with tf.variable_scope('mse'):  # To optimize
            loss_terms = {
                'gaze_mse': tf.reduce_mean(tf.squared_difference(output, y)),
            }
        with tf.variable_scope('ang'):  # To evaluate in addition to loss terms
            metrics = {
                'gaze_angular': util.gaze.tensorflow_angular_error_from_pitchyaw(output, y),
            }
        return {'gaze': output}, loss_terms, metrics


