"""Using the methods in mnist.py building training and evaluation procedure"""
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from . import mnist

# External flags
FLAGS = None

# function to set the placeholders
def get_placeholders(batch_size):
    """
    Generate placeholder variables which are used to give input to the graph
    :param
        batch_size: the batch size
    :return:
        images_placeholder: placeholder variable containing images data
        labels_placeholder: placeholder variable containing the ground truth value
    """
    images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, mnist.IMAGE_PIXELS])
    labels_placeholder = tf.placeholder(tf.int32, shape=[batch_size, 10])
    keep_prob = tf.placeholder(tf.float32)
    return images_placeholder, labels_placeholder, keep_prob


# function for feed_dict
def data_feed(data_set, batch_size, images_placeholder, labels_placeholder, keep_prob, keep_prob_value):
    """
    Generate a feed_dict object which can be used to get batch wise data during training
    :param data_set: data_set is a generator object which will give us dat based on the batch size
    :param images_placeholder: images placeholder which will be initialed to the data taken from data_set
    :param labels_placeholder: labels placeholder which will be initialized with the ground truth from data_set
    :param keep_prob: placeholder variable for probability for dropout layer
    :param keep_prob_value: value for the keep_prob placeholder
    :return: feed_dict: dict object mapping placeholders to example values
    """
    images_feed, labels_feed = data_set.next_batch(batch_size)
    feed_dict = {
        images_placeholder: images_feed,
        labels_placeholder: labels_feed,
        keep_prob: keep_prob_value
    }
    return feed_dict


# function for evaluation
def do_eval(sess, data_set, images_placeholder, labels_placeholder, eval_correct, batch_size, keep_prob):
    """
    Run one evalutation on one full epoch of data
    :param data_set:
    :param images_placeholder:
    :param labels_placeholder:
    :param eval_correct:
    :param batch_size:
    :return:
    """
    true_count = 0  # count of correct predictions
    steps_per_epoch = data_set.num_examples / batch_size
    no_of_examples = steps_per_epoch * batch_size
    for i in range(steps_per_epoch):
        feed_dict = data_feed(data_set,batch_size, images_placeholder, labels_placeholder, keep_prob, 1)
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / no_of_examples
    print("Num examples: %d, True count: %d, precision: %f"%(no_of_examples, true_count, precision))


# function to run training
def do_training():
    """
    Train MNIST for max_steps number of iterations
    :param max_steps: number of training iteration
    :param batch_size: The batch size
    :return:
    """
    max_steps = FLAGS[0].max_steps
    batch_size = FLAGS[0].batch_size
    data_set = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # define the placeholder variables
    images_placeholder, labels_placeholder, keep_prob = get_placeholders(batch_size)

    # define the graph
    logits = mnist.inference(images_placeholder, keep_prob)
    loss = mnist.loss(logits=logits, labels=labels_placeholder)
    train_step = mnist.training(loss)
    evaluation = mnist.evaluation(logits=logits, labels=labels_placeholder)

    # Variable initialization
    # Before we can use the variables in the graph, we need to initialize them with the data provided in the variable
    # definition command
    init = tf.global_variables_initializer()

    # start a session
    sess = tf.Session()

    # run init
    sess.run(init)

    # feed_dict = data_feed(data_set.train, batch_size, images_placeholder, labels_placeholder, keep_prob, 0.5)
    # output = sess.run([logits], feed_dict=feed_dict)

    # training operation for max_steps iteration
    for i in range(max_steps):
        feed_dict = data_feed(data_set.train, batch_size, images_placeholder, labels_placeholder, keep_prob, 0.5)
        _, loss_value= sess.run([train_step,loss], feed_dict=feed_dict)
        if i % 100 == 0:
            print("step : %d, loss : %g" % (i, loss_value))

    # validation evaluation
    print("Validation Eval: ")
    do_eval(sess=sess, data_set=data_set.validation, images_placeholder=images_placeholder,
            labels_placeholder=labels_placeholder, eval_correct=evaluation, batch_size=batch_size)

    # test evaluation
    print("Test Eval: ")
    do_eval(sess=sess, data_set=data_set.test, images_placeholder=images_placeholder,
            labels_placeholder=labels_placeholder, eval_correct=evaluation, batch_size=batch_size)


# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100, help="Batch Size")
    parser.add_argument('-s', '--max_steps', type=int, default=10000, help='Max number of iterations')

    FLAGS = parser.parse_known_args()
    do_training()
