import tensorflow as tf
import tensorflow.contrib.layers as layers
class Network:
    def __init__(self):
        pass

    def build_network(self):
        pass

    def _conv2d(self, inputs, kernel_size, stride, input_channel,
                output_channel, biased, name):
        """
        init weights(and bias maybe) for a convolutional layer, and do
        convolution using same padding for the input feature
        :param inputs: a tensor of [batch_size, width, height, channel]
        :param kernel_size:
        :param stride:
        :param input_channel: number of input channel
        :param output_channel: number of output channel
        :param biased: a boolean indicating whether add bias or not
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            w = self._init_weight_for_conv2d(kernel_size, input_channel,
                                             output_channel)
            s = [1, stride, stride, 1]
            output = tf.nn.conv2d(input=inputs, filter=w,
                                  strides=s, padding='SAME')
            if biased:
                b = self._init_for_bias(output_channel)
                output = tf.nn.bias_add(output, b)
            return output

    def _atrous_conv2d(self, inputs, kernel_size, dilation_rate,
                       input_channel, output_channel, biased, name):
        """
        init weights(and bias maybe) for a atrous convolutional layer,
        and do convolution using same padding for the input feature
        :param inputs:
        :param kernel_size:
        :param dilation_rate:
        :param input_channel:
        :param output_channel:
        :param biased:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            w = self._init_weight_for_conv2d(kernel_size, input_channel,
                                             output_channel)
            output = tf.nn.atrous_conv2d(value=inputs, filters=w, rate=dilation_rate,
                                         padding='SAME')
            if biased:
                b = self._init_for_bias(output_channel)
                output = tf.nn.bias_add(output, b)
            return output

    def _add_element_wise(self, name, *a):
        """
        add all parameters element wise, so all parameters must have same
        shape
        :param a: tensors to be added
        :return:
        """
        return tf.add_n(a, name)

    def _relu(self, name, inputs):
        return tf.nn.relu(inputs, name)

    def _batch_normal(self, name, inputs, active_func, is_training,
                      trainable):
        """
        doing batch normal with multiplied by `gamma`
        :param name:
        :param inputs:
        :param active_func: function(like relu) apply after normalization
        :param is_training: if is_training, it will use moving average of
                            all the batches have been trained, else using
                            the average of trained batches to predict.
                            set to False to frozen the statistics of the BN layers
                            and just use the pre-trained data when the batch size
                            is not big enough
        :param trainable: decide whether train gamma and bate can be trained
        :return:
        """
        with tf.variable_scope(name) as scope:
            output = layers.batch_norm(inputs=inputs,
                                       scale=True,
                                       activation_fn=active_func,
                                       is_training=is_training,
                                       trainable=trainable,
                                       scope=scope)
            return output

    def _init_weight_for_conv2d(self, kernel_size, input_channel, output_channel):
        w = tf.get_variable(name='weights',
                            shape=[kernel_size, kernel_size,
                                   input_channel, output_channel],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        return w

    def _init_for_bias(self, output_channel):
        b = tf.get_variable(name='biases',
                            shape=[output_channel],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
        return b

    def _max_pool2d(self, name, inputs, kernel_size, stride):
        k = [1, kernel_size, kernel_size, 1]
        s = [1, stride, stride, 1]
        return tf.nn.max_pool(inputs, k, s, padding='SAME', name=name)
