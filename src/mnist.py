"""Builds the MNIST network based on the template provided at
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/mnist/mnist.py

The network is divided into four parts
1. Inference: Build the network from image input to the part where we get a inference output or a forward pass output
2. Loss: Builds over Inference by adding ops to calculate the loss function
3. Training: Builds over Loss by adding ops to train the network using back-propagation
4. Evaluation: Evaluates the performance of the network in calculating the ability of the network to predict the labels
"""
import tensorflow as tf
import math


# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def weight_variable(shape):
    """
    # Variable Initializing Functions
    # function for initializing with better values for W
    :param shape:
    :return: tf.Variable
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


def bias_variable(shape):
    """
    # Variable Initializing Functions
    # function for initializing with better values for b
    :param shape:
    :return: tf.Variable
    """
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial_value=initial)


# Convolutions and Pooling
def conv2d(x,W):
    return tf.nn.conv2d(x,W,[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def inference(images, keep_prob):
    # reshape input image
    x_image = tf.reshape(images, shape=[-1,28,28,1])

    # layer 1
    # variable def
    with tf.name_scope('hidden1'):
        w_conv1 = weight_variable([5,5,1,32])
        b_conv1 = bias_variable([32])

        # operation def
        o_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
        o_pool1 = max_pool_2x2(o_conv1)

    # layer 2
    # variable def
    with tf.name_scope('hidden2'):
        w_conv2 = weight_variable([5,5,32,64])
        b_conv2 = bias_variable([64])

        # operation def
        o_conv2 = tf.nn.relu(conv2d(o_pool1, w_conv2) + b_conv2)
        o_pool2 = max_pool_2x2(o_conv2)

    # densely connected layer
    # variable def
    with tf.name_scope('dc1'):
        w_conv3 = weight_variable([7*7*64, 1024])
        b_conv3 = bias_variable([1024])

        # reshape the output of layer 2
        o_inp3 = tf.reshape(o_pool2, shape=[-1,7*7*64])

        # operation def
        o_conv3 = tf.nn.relu(tf.matmul(o_inp3, w_conv3) + b_conv3)

    # dropout layer
    with tf.name_scope('dropout'):
        o_dc1_drop = tf.nn.dropout(o_conv3, keep_prob=keep_prob)

    # readout layer
    with tf.name_scope('readout'):
        w_readout = weight_variable([1024,10])
        b_readout = bias_variable([10])
        logits = tf.matmul(o_dc1_drop, w_readout) + b_readout
    return logits


def loss(logits, labels):
    # Loss function
    # define the loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return tf.reduce_mean(cross_entropy)


def training(loss):
    # Model Training
    # define train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step


def evaluation(logits, labels):
    # evaluation
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.int32))
    return accuracy



