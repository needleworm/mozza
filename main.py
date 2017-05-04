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


if FLAGS.reset:
    os.popen("rm -rf " + logs_dir + " " + results_dir)
    os.popen("mkdir " + logs_dir + " " + results_dir)

learning_rate = 0.0001
MAX_ITERATION = 10000


class Mozza:
    def __init__(self):
        self.conv1_shape = [3, 3, 1, 32]
        self.kernel1 = tf.get_variable("index_1_W", initializer=tf.truncated_normal(self.conv1_shape, stddev=stddev))
        self.bias1 = tf.get_variable("index_1_B", initializer=tf.constant(0.0, shape=[self.conv1_shape[-1]]))

        self.conv2_shape = [3, 3, 32, 64]
        self.kernel2 = tf.get_variable("index_2_W", initializer=tf.truncated_normal(self.conv2_shape, stddev=stddev))
        self.bias2 = tf.get_variable("index_2_B", initializer=tf.constant(0.0, shape=[self.conv2_shape[-1]]))

        self.FNN1_shape = [int(FLAGS.YDIM / 9) * int(FLAGS.XDIM / 9) * 64, 1024]
        self.kernel3 = tf.get_variable("index_10_W", initializer=tf.truncated_normal(self.FNN1_shape, stddev=stddev))
        self.bias3 = tf.get_variable("index_10_B", initializer=tf.constant(0.1, shape=[self.FNN1_shape[-1]]))

        self.FNN2_shape = [1024, 2]
        self.kernel4 = tf.get_variable("index_11_W", initializer=tf.truncated_normal(self.FNN2_shape, stddev=stddev))
        self.bias4 = tf.get_variable("index_11_B", initializer=tf.constant(0.1, shape=[self.FNN2_shape[-1]]))

    def graph(self):
        observation = tf.placeholder(tf.float32, [None, FLAGS.YDIM, FLAGS.XDIM, 1])  # 81 x 72 x frame_stack

        # Conv-Relu-MaxPool 1
        C1 = tf.nn.conv2d(observation, self.kernel1, strides=[1, stride, stride, 1], padding="SAME")
        C1 = tf.nn.bias_add(C1, self.bias1)
        C1 = tf.contrib.layers.batch_norm(C1, decay=0.9, is_training=FLAGS.Train, updates_collections=None)
        R1 = tf.nn.relu(C1, name="index_1_RL")
        P1 = tf.nn.max_pool(R1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")  # [?, 27, 24, 32]
        # Conv-Relu-MaxPool 2
        C2 = tf.nn.conv2d(P1, self.kernel2, strides=[1, stride, stride, 1], padding="SAME")
        C2 = tf.nn.bias_add(C2, self.bias2)
        C2 = tf.contrib.layers.batch_norm(C2, decay=0.9, is_training=FLAGS.Train, updates_collections=None)
        R2 = tf.nn.relu(C2, name="index_2_RL")
        P2 = tf.nn.max_pool(R2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding="SAME")  # [?, 9, 8, 64]

        # Fully Connected
        W1 = tf.matmul(tf.reshape(P2, [-1, self.FNN1_shape[0]]), self.kernel3)

        # Dropout with Sigmoid
        D1 = tf.nn.sigmoid(tf.matmul(W1, self.kernel4))

        return observation, D1

    # Sample action with random rate eps
    def dropout_2_keys(self, P, feed, eps):
        act_values = P.eval(feed_dict=feed)
        if random.random() <= eps:
            action_index = random.randrange(FLAGS.num_actions)
        else:
            action_index = np.argmax(act_values[-1])
        action = np.zeros(FLAGS.num_actions)
        action[action_index] = 1
        return action


def train(is_training=True):
    # Define placeholders to catch inputs and add options
    with tf.device(FLAGS.device):
        ###############################  GRAPH PART  ###############################
        print("Graph Initialization...")
        agent = QAgent()
        observation1, Q1 = agent.graph()
        observation2, Q2 = agent.graph()
        print("Done")

        ##############################  Summary Part  ##############################
        print("Setting up summary op...")
        loss_placeholder = tf.placeholder(dtype=tf.float32)
        loss_summary_op = tf.summary.scalar("entropy", loss_placeholder)
        loss1_summary_writer = tf.summary.FileWriter(logs_dir + "/loss1/")
        loss2_summary_writer = tf.summary.FileWriter(logs_dir + "/loss2/")
        score_placeholder = tf.placeholder(dtype=tf.float32)
        score_summary = tf.summary.scalar("Score", score_placeholder)
        score_summary_writer = tf.summary.FileWriter(logs_dir + "/score/")
        print("Done")

        ############################  Model Save Part  #############################
        print("Setting up Saver...")
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        print("Done")

        ############################  Placeholder Part  ############################
        print("Setting up Placeholders...")
        actions = tf.placeholder(tf.float32, [None, FLAGS.num_actions])
        R = tf.placeholder(tf.float32, [None, ])
        values1 = Q1 - actions
        # values2 = R + tf.reduce_mean(FLAGS.gamma * Q2 - Q1)
        values3 = tf.reduce_sum(tf.multiply(Q1, actions), axis=1)
        values2 = R + FLAGS.gamma * tf.reduce_max(Q2, axis=1)
        loss2 = tf.reduce_mean(tf.square(values3 - values2))
        loss1 = tf.reduce_sum(tf.square(values1))

        model1 = tf.train.AdamOptimizer(learning_rate).minimize(loss1)
        model2 = tf.train.AdamOptimizer(learning_rate).minimize(loss2)
        print("Done")

    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)

    ################################  Session Part  ################################
    print("Session Initialization...")
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("model restored...")
    else:
        sess.run(tf.global_variables_initializer())



def main():
    pass
