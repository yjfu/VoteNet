import tensorflow as tf
from  deepLab_v2 import DeepLabV2

import numpy as np

z = [[1,2, 0]]
z = tf.constant(z, tf.float32)

# DeepLabV2(input, 3)
# w = tf.global_variables()
output = tf.nn.softmax(z)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a = sess.run(output)
    print a
