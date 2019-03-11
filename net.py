import tensorflow as tf
import numpy as np

# fake data
x_data = np.random.rand(20000).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tensorflow structure
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

# build model
y_predicted = x_data * weights + biases
# loss
loss = tf.reduce_mean(tf.square(y_data - y_predicted))
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        sess.run(optimizer)

        if step % 20 == 0:
            print(step, sess.run(weights), sess.run(biases))
