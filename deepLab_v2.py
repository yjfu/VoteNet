from network import Network
import tensorflow as tf
class DeepLabV2(Network):
    def __init__(self, inputs, classes_num):
        self.inputs = inputs
        self.classes_num = classes_num
        self._building_network()

    def _building_network(self):
        self.output = self._encoder()
        self.output = self._decoder(self.output, 2048)

    def _encoder(self):
        """
        building encoding part to process self.inputs, with 33 (atrous or normal)
        residual block and a simple conv layer in front
        :return: a tensor of 2048 channel, and size of w/8, h/8, where w and h
                are width and height of self.inputs, respectively
        """
        print "building encoder..."

        print "building first block..."
        output = self._first_block(self.inputs, input_channel=3)
        print "after first block:", output.shape
        print "==========================================================="
        print "building residual_blocks..."
        print "res1 with 256 channel"
        output = self._bottleneck_residual_block("2a", output,
                                                      input_channel=64,
                                                      output_channel=256,
                                                      half_size=False,
                                                      identity_connection=False)
        output = self._bottleneck_residual_block("2b", output,
                                                      256, 256, False, True)
        output = self._bottleneck_residual_block("2c", output,
                                                      256, 256, False, True)
        print "after res1 blocks: ", output.shape
        print "-----------------------------------------------------------"
        print "res2 with 512 channel"
        output = self._bottleneck_residual_block("3a", output,
                                                      256, 512, True, False)
        for i in range(1, 4):
            output = self._bottleneck_residual_block("3b%d" % i, output,
                                                          512, 512, False, True)
        print "after res2 blocks: ", output.shape
        print "-----------------------------------------------------------"
        print "res3 with 1024 channel, atrous conv"
        output = self._atrous_bottleneck_residual_block("4a", output,
                                                             512, 2, 1024, False)
        for i in range(1, 23):
            output = self._atrous_bottleneck_residual_block("4b%d" % i, output,
                                                                 1024, 2, 1024, True)
        print "after res3 block: ", output.shape
        print "-----------------------------------------------------------"
        print "res4 with 2048 channel, atrous conv"
        output = self._atrous_bottleneck_residual_block("5a", output,
                                                             1024, 4, 2048, False)
        output = self._atrous_bottleneck_residual_block("5b", output,
                                                             2048, 4, 2048, True)
        output = self._atrous_bottleneck_residual_block("5c", output,
                                                             2048, 4, 2048, True)
        print "after res4 block: ", output.shape
        print "==========================================================="
        return output

    def _decoder(self, inputs, input_channel):
        """
        building decoder, which is essentially an ASPP structure
        :param inputs: typically the output of encoder here
        :param input_channel: channel of input, in deeplabv2, this parameter is 2048
        :return: a feature map(or, prediction map) with channel of classes number,
                one channel for one class
        """
        print "building decoder..."
        output = self._ASPP(inputs, input_channel, self.classes_num, [6, 12, 18, 24])
        print "after decoder(ASPP): ", output.shape
        print "==========================================================="
        return output

    def _bottleneck_residual_block(self, name, inputs, input_channel, output_channel,
                                   half_size, identity_connection):
        """
        doing bottleneck residual block, which will use convolution layer of 1*1
        kernel to quarter and, recover output channel number to reduce computation.
        Note that all conv layers are with a batch normal operator later and without
        any biases.
        :param name:
        :param inputs: input tensor
        :param input_channel: input channel number
        :param output_channel: output channel number. Note that because of bottleneck,
                                the output channel number will be quartered by 1*1 conv
                                layer, so it must be a multiple of 4
        :param half_size: if True, using stride=2 at first layer so it will output a
                            feature map have size of h/2, w/2, where h and w are the
                            height and width of input feature map, respectively.
        :param identity_connection: if False, the residual block will not add the copy
                                    of input in the end but the feature map after
                                    processing of a conv layer
        :return:
        """
        assert output_channel % 4 == 0, \
            "ERROR: output channel must be a multiple of 4 using bottleneck residual"
        assert not half_size & identity_connection, \
            "ERROR: cannot keep input identical while making feature size half"
        first_stride = 2 if half_size else 1
        # input branch
        if not identity_connection:
            branch_1 = self._conv2d(inputs, kernel_size=1, stride=first_stride,
                                    input_channel=input_channel,
                                    output_channel=output_channel,
                                    biased=False, name='res%s_branch1' % name)
            branch_1 = self._batch_normal(name='bn%s_branch1' % name, inputs=branch_1,
                                          active_func=tf.nn.relu,
                                          is_training=False, trainable=False)
        else:
            branch_1 = inputs
        # residual stack branch
        # first conv+bn
        branch_2 = self._conv2d(inputs, 1, first_stride, input_channel,
                                output_channel/4, False, 'res%s_branch2a' % name)
        branch_2 = self._batch_normal('bn%s_branch2a' % name, branch_2, tf.nn.relu,
                                      False, False)
        # second conv+bn
        branch_2 = self._conv2d(branch_2, 3, 1, output_channel/4,
                                output_channel/4, False, 'res%s_branch2b' % name)
        branch_2 = self._batch_normal('bn%s_branch2b' % name, branch_2, tf.nn.relu,
                                      False, False)
        # third conv+bn
        branch_2 = self._conv2d(branch_2, 1, 1, output_channel/4,
                                output_channel, False, 'res%s_branch2c' % name)
        branch_2 = self._batch_normal('bn%s_branch2c' % name, branch_2, None,
                                      False, False)
        # merge branch
        output = self._add_element_wise('merge', branch_1, branch_2)
        output = self._relu('output', output)
        return output

    def _atrous_bottleneck_residual_block(self, name, inputs, input_channel, rate,
                                          output_channel, identity_connection, ):
        """
        Same as function _bottleneck_residual_block, but the middle conv layer use
        atrous convolution with dilate rate of rate
        :param name:
        :param inputs:
        :param input_channel:
        :param rate: dilate rate
        :param output_channel:
        :param identity_connection:
        :return:
        """
        assert output_channel % 4 == 0, \
            "ERROR: output channel must be a multiple of 4 using bottleneck residual"

        # input branch
        if not identity_connection:
            branch_1 = self._conv2d(inputs, kernel_size=1, stride=1,
                                    input_channel=input_channel,
                                    output_channel=output_channel,
                                    biased=False, name='res%s_branch1' % name)
            branch_1 = self._batch_normal(name='bn%s_branch1' % name, inputs=branch_1,
                                          active_func=tf.nn.relu,
                                          is_training=False, trainable=False)
        else:
            branch_1 = inputs
        # residual stack branch
        # first conv+bn
        branch_2 = self._conv2d(inputs, 1, 1, input_channel,
                                output_channel / 4, False, 'res%s_branch2a' % name)
        branch_2 = self._batch_normal('bn%s_branch2a' % name, branch_2, tf.nn.relu,
                                      False, False)
        # second atrous_conv+bn
        branch_2 = self._atrous_conv2d(branch_2, 3, rate, output_channel/4, output_channel/4,
                                       False, 'res%s_branch2b' % name)
        branch_2 = self._batch_normal('bn%s_branch2b' % name, branch_2, tf.nn.relu,
                                      False, False)
        # third conv+bn
        branch_2 = self._conv2d(branch_2, 1, 1, output_channel / 4,
                                output_channel, False, 'res%s_branch2c' % name)
        branch_2 = self._batch_normal('bn%s_branch2c' % name, branch_2, None,
                                      False, False)
        # merge branch
        output = self._add_element_wise('merge', branch_1, branch_2)
        output = self._relu('output', output)
        return output

    def _ASPP(self, inputs, input_channel, output_channel, rates):
        """
        doing atrous spatial pyramid pooling
        :param inputs:
        :param input_channel:
        :param output_channel:
        :param rates: a list of dilation rates
        :return:
        """
        outputs = []
        for index, dilation_rate in enumerate(rates):
            branch = self._atrous_conv2d(inputs, 3, dilation_rate, input_channel,
                                         output_channel, True,
                                         'fc1_voc12_c%d' % index)
            outputs.append(branch)
        return self._add_element_wise("ASPP/output", *outputs)

    def _first_block(self, inputs, input_channel):
        """
        first block of deepLab2, including a convolution layer, a batch normal,
        and a max pooling layer
        :param inputs:
        :param input_channel:
        :return:
        """
        output = self._conv2d(inputs, kernel_size=7, stride=2,
                              input_channel=input_channel, output_channel=64,
                              biased=False, name='conv1')
        output = self._batch_normal('bn_conv1', output, tf.nn.relu, False, False)
        output = self._max_pool2d('pool_conv1', output, kernel_size=3, stride=2)
        return output

