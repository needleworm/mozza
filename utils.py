"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.01.
"""

import re
import tensorflow as tf


def get_conv_shape(name):
    spec = re.split(':|, |->', name)
    kernel_size = int(spec[5])
    stride = int(spec[7])
    input_fm = int(spec[9])
    output_fm = int(spec[10])
    conv_shape = [kernel_size, kernel_size, input_fm, output_fm]
    return conv_shape, stride


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d(x, W, bias, stride=1):
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def deconv(x, W, b, output_shape, stride=1):
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)

