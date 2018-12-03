# coding: utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('data/mnist_data/', one_hot=True)

session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_act = tf.placeholder(tf.float32, [None, 10])

y_pre = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_act * tf.log(y_pre)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

session.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    session.run(train_step, feed_dict={x: batch_xs, y_act: batch_ys})

correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_act, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(session.run(accuracy, feed_dict={x: mnist.test.images, y_act: mnist.test.labels}))

