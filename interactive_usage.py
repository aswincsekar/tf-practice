# Interactive sessions are only good for ipython etc.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initializing 'x' using the run() method of its initializer
x.initializer.run()

# Add an op to subtract 'a' from 'x'. Run it and print
sub = tf.sub(x, a)
print(sub.eval())

# Close the session
sess.close()
