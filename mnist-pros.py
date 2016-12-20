# Load data from examples
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print ( 'hola')
# Build softmax regression layer
# Start single hidden layer
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# define the variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



# calculate the output
y = tf.matmul(x, W) + b

print('2')
# define the loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print('3')
sess = tf.Session()

# initialize the variables
sess.run(tf.global_variables_initializer())

# training time
for i in range(1000):
    print(i)
    batch = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})

# how much accuracy did we hit
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

sess.close()