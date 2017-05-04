"""
    Author : Byunghyun Ban
    needleworm@kaist.ac.kr
    latest modification :
        2017.05.04.
"""

import tensorflow as tf
import numpy as np
import os
import re
import utils



logs_dir = "logs"
results_dir = "results"
val_dir = "high resolution directory"


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("device", "/gpu:0", "device : /cpu:0, /gpu:0, /gpu:1. [Default : /gpu:0]")
tf.flags.DEFINE_bool("Train", "True", "mode : train, test. [Default : train]")
tf.flags.DEFINE_bool("reset", "True", "mode : True or False. [Default : train]")
tf.flags.DEFINE_integer("batch_size", "5", "batch size. [Default : 5]")
tf.flags.DEFINE_integer("num_keys", "88", "number of keys. [Default : 88]")
tf.flags.DEFINE_string("evaluator", "precision",
                       "binary classification methods : precision, recall, specificity, accuracy, miu, fiu.")


if FLAGS.reset:
    os.popen("rm -rf " + logs_dir + " " + results_dir)
    os.popen("mkdir " + logs_dir + " " + results_dir)

learning_rate = 0.0001
MAX_ITERATION = 10000
stddev = 0.02



class Mozza_fnn:
    def __init__(self):
        self.FNN1_shape = [FLAGS.batch_size * FLAGS.num_keys, 2048]
        self.kernel1 = tf.get_variable("index_1_W", initializer=tf.truncated_normal(self.FNN1_shape, stddev=stddev))
        self.bias1 = tf.get_variable("index_1_B", initializer=tf.constant(0.1, shape=[self.FNN1_shape[-1]]))

        self.FNN2_shape = [2048, 2048]
        self.kernel2 = tf.get_variable("index_2_W", initializer=tf.truncated_normal(self.FNN2_shape, stddev=stddev))
        self.bias2 = tf.get_variable("index_2_B", initializer=tf.constant(0.1, shape=[self.FNN2_shape[-1]]))

        self.FNN3_shape = [2048, 2048]
        self.kernel3 = tf.get_variable("index_3_W", initializer=tf.truncated_normal(self.FNN3_shape, stddev=stddev))
        self.bias3 = tf.get_variable("index_3_B", initializer=tf.constant(0.1, shape=[self.FNN3_shape[-1]]))

        self.FNN4_shape = [2048, FLAGS.batch_size * FLAGS.num_keys]
        self.kernel4 = tf.get_variable("index_4_W", initializer=tf.truncated_normal(self.FNN4_shape, stddev=stddev))
        self.bias4 = tf.get_variable("index_3_B", initializer=tf.constant(0.1, shape=[self.FNN4_shape[-1]]))

    def graph(self):
        seed = tf.placeholder(tf.float32, [None, FLAGS.num_keys])

        W1 = tf.matmul(seed, self.kernel1)
        W1 = tf.nn.bias_add(W1, self.bias1)
        R1 = tf.nn.relu(W1, name="index_1_R")

        W2 = tf.matmul(R1, self.kernel2)
        W2 = tf.nn.bias_add(W2, self.bias2)
        R2 = tf.nn.relu(W2, name="index_1_R")

        W3 = tf.matmul(R2, self.kernel3)
        W3 = tf.nn.bias_add(W3, self.bias3)
        R3 = tf.nn.relu(W3, name="index_1_R")

        W4 = tf.matmul(R3, self.kernel4)
        W4 = tf.nn.bias_add(W4, self.bias4)

        predict = tf.nn.sigmoid(W4)

        return seed, predict


def train(is_training=True):
    # Define placeholders to catch inputs and add options
    with tf.device(FLAGS.device):
        ###############################  GRAPH PART  ###############################
        print("Graph Initialization...")
        mozza = Mozza_fnn()
        seed, predict = mozza.graph()
        print("Done")

        ############################  Placeholder Part  ############################
        print("Setting up Placeholders...")
        predicted_keys = utils.prediction_2_keys(predict)
        ground_truth = tf.placeholder(tf.float32, [None, FLAGS.num_keys])

        true_positive = tf.reduce_sum(tf.multiply(predicted_keys, ground_truth))
        true_negative = tf.reduce_sum(tf.multiply(1 - predicted_keys, 1 - ground_truth))
        false_positive = tf.reduce_sum(tf.multiply(predicted_keys, 1 - ground_truth))
        false_negative = tf.reduce_sum(tf.multiply(1 - predicted_keys, ground_truth))

        loss = tf.square(utils.calculate_loss(true_positive, true_negative,
                                              false_positive, false_negative, FLAGS.evaluator))
        model = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        print("Done")

        ##############################  Summary Part  ##############################
        print("Setting up summary op...")
        loss_placeholder = tf.placeholder(dtype=tf.float32)
        loss_summary_op = tf.summary.scalar("loss", loss_placeholder)
        loss_summary_writer = tf.summary.FileWriter(logs_dir + "/loss1/")
        score_placeholder = tf.placeholder(dtype=tf.float32)
        score_summary = tf.summary.scalar("Score", score_placeholder)
        score_summary_writer = tf.summary.FileWriter(logs_dir + "/score/")
        print("Done")

        ############################  Model Save Part  #############################
        print("Setting up Saver...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("model restored...")
    else:
        sess.run(tf.global_variables_initializer())

def main():
    pass
