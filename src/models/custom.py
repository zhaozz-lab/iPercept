
from core import BaseModel
import tensorflow as tf


class CustomModel(BaseModel):
    """
    Custom extension to base model.
    We incorporate methods that help building larger networks faster.
    """
    #
    # def build_model(self, data_sources: Dict[str, BaseDataSource], mode: str):
    #     """Build model."""
    #     raise NotImplementedError('BaseModel::build_model is not yet implemented.')

    def get_conv2d_multi(self, base_name, n, x, filters, kernel_size=2, strides=1, padding='same',
                         data_format='channels_last', activation=tf.nn.relu):
        """
        Creates n convolutional layers with given specifications and appends them to x. Returns the last layer x.
        Layers are named in the following way:
        <BASE_NAME>.<INDEX>-<NUMBER_OF_FILTERS>
        e.g. "conv2.1-512 for the second (index 1) layer of the group called conv2. All layers in conv2 have 512 filters.
        :param base_name:
        :param n:
        :param x:
        :param filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :param data_format:
        :param activation:
        :return: x, the last added layer
        """
        # We add n layers
        for i in range(n):
            # <BASE_NAME>.<INDEX>-<NUMBER_OF_FILTERS>
            layer_name = "{}.{}-{}".format(base_name, i, filters)
            x = self.get_conv2d(name=layer_name, x=x, filters=filters, kernel_size=kernel_size, strides=strides,
                                padding=padding, data_format=data_format, activation=activation)
        return x

    def get_conv2d(self, name, x, filters, kernel_size=2, strides=1, padding='same', data_format='channels_last',
                   activation=tf.nn.relu):
        """
        Creates a single convolutional layer with the given name.
        :param name:
        :param x:
        :param filters:
        :param kernel_size:
        :param strides:
        :param padding:
        :param data_format:
        :param activation:
        :return: x, the new layer
        """
        with tf.variable_scope(name):
            x = tf.layers.conv2d(x, filters=filters, kernel_size=kernel_size, strides=strides,
                                 padding=padding, data_format=data_format, activation=activation)
            self.summary.filters('filters', x)
            self.summary.feature_maps('features', x, data_format=data_format)
        return x

    def get_max_pooling2d(self, base_name, x, pool_size, strides, padding, data_format):
        """
        Creates a single max pooling layer.
        Layer is named in the following way:
        <BASE_NAME>-<POOL_SIZE>-<STRIDES>
        :param base_name:
        :param x:
        :param pool_size:
        :param strides:
        :param padding:
        :param data_format:
        :return: x, the new layer
        """
        layer_name = "{}-{}-{}".format(base_name, pool_size, strides)
        with tf.variable_scope(layer_name):
            x = tf.layers.max_pooling2d(x, pool_size=pool_size, strides=strides, padding=padding,
                                        data_format=data_format)
            self.summary.filters('filters', x)
            self.summary.feature_maps('features', x, data_format=data_format)
        return x
