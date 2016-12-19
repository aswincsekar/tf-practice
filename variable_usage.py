# Learn about variables in TF
import tensorflow as tf

# define a variable called counter
state = tf.Variable(0,name='counter')

# add one to counter and some extra
one = tf.constant(1)
extra = tf.placeholder(tf.int32)
two = tf.add(one, extra)
added_value = tf.add(state, two)
update = tf.assign(state, added_value)

# Variables are initialized only when you run a 'init' Op
# define the 'init' Op
init_op = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
    # run the init Op
    sess.run(init_op)
    # print the current value of the state
    print(sess.run(state))
    # run the update op multiple times
    for i in range(3):
        sess.run(update, feed_dict={extra: 7})
        print(sess.run(state))

