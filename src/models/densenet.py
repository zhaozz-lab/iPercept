"""DenseNet architecture."""
from typing import Dict


import tensorflow as tf
#from tensorpack import *
#from tensorpack.tfutils.symbolic_functions import *
#from tensorpack.tfutils.summary import *

from core import BaseDataSource, BaseModel
import util.gaze

data_format = "channels_last"  # Change this to "channels_first" to run on GPU

class DenseNet(BaseModel):
    """An implementation of the DenseNet architecture."""

    def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
        """Build model."""

        # Hardcoded for now
        depth = 40
        self.N = int((depth - 4) / 3)
        self.growthRate = 12

        data_source = next(iter(data_sources.values()))
        input_tensors = data_source.output_tensors
        x = input_tensors['eye']
        y = input_tensors['gaze']

        def conv(name, l, channel, stride):
            # added data_format (from examplenet)
            #output= Conv2D(name, l, channel, 3, stride=stride,
            #              nl=tf.identity, use_bias=False,
            #              W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)),
            #              data_format=data_format)

            output=tf.layers.conv2d(l, filters=channel, kernel_size=3, strides=stride,
                                     padding='same', name=name, data_format=data_format)

            return output

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
                #l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu, data_format=data_format)
                l = tf.layers.conv2d(l, filters=in_channel, strides=1, kernel_size=1, padding='same',
                                     data_format=data_format, name='conv1')
                l = tf.nn.relu(l)
                # changed from tensorpack
                layer = tf.layers.AveragePooling2D(name='pool', padding='same', strides=2,
                                                   pool_size=2, data_format=data_format)
                l = layer.apply(l, scope=tf.get_variable_scope())
            return l

        def global_average_pooling(x, data_format='channels_last', name=None):
            """
            Global average pooling as in the paper `Network In Network
            <http://arxiv.org/abs/1312.4400>`_.
            Args:
                x (tf.Tensor): a 4D tensor.
            Returns:
                tf.Tensor: a NC tensor named ``output``.
            """
            assert x.shape.ndims == 4
            axis = [1, 2] if data_format == 'channels_last' else [2, 3]
            return tf.reduce_mean(x, axis, name=name)

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
                l = global_average_pooling(name='gap', x=l, data_format=data_format)
                #logits = FullyConnected('linear', l, out_dim=2, nl=tf.identity)
                regressed_output = tf.layers.dense(l, units=2, name='fc4', activation=None)

            return regressed_output

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


