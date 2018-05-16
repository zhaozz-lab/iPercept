
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

    def get_max_pooling2d(self, base_name: str, x, pool_size: int, strides: int, padding, data_format):
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
            #self.summary.filters('filters', x)
            self.summary.feature_maps('features', x, data_format=data_format)
        return x

    def get_residual_block(self, x_input: tf.Tensor, x_output: tf.Tensor):
        """
        Resizes x_input to match the size of x_output. Returns sum.
        :param x_input: Input to block
        :param x_output: Output of inner block
        :return:
        """
        # Resize x_output
        residual = tf.image.resize_images(x_input[:,:,:,0:1], size=x_output.get_shape()[1:3])
        return x_output + residual

    def augment_x(self, x: tf.Tensor, y, add_noise=False):
        list_x = [x]
        list_y = [y]

        x_random_brightness = tf.map_fn(lambda img: tf.image.random_brightness(img, max_delta=.8), x)
        x_random_contrast = tf.map_fn(lambda img: tf.image.random_contrast(img, .2, 1.8), x)
        list_x.append(x_random_contrast)
        list_x.append(x_random_brightness)

        list_y.append(y)
        list_y.append(y)

        result_x = tf.concat(list_x, axis=0)
        result_y = tf.concat(list_y, axis=0)

        if add_noise:
            noise = tf.random_normal(shape=tf.shape(result_x), mean=0.0, stddev=0.01)
            result_x = result_x + noise
        return result_x, result_y
