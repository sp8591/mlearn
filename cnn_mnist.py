import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
data_path = '../data/mnist'
mnist = input_data.read_data_sets(data_path, one_hot=True)
tr_x, tr_y, te_x, te_y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
tr_x = tr_x.reshape(-1, 28, 28, 1)
te_x = te_x.reshape(-1, 28, 28, 1)
x = tf.placeholder('float', [None, 28, 28, 1])
y = tf.placeholder('float', [None, 10])
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128 * 4 * 4, 625])
w_o = init_weights([625, 10])
def model(x, w, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden):
    lla = tf.nn.relu(tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME'))
    ll = tf.nn.max_pool(lla, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ll = tf.nn.dropout(ll, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.conv2d(ll, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(tf.reshape(l3, [-1, 2048]), w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_out)
    return pyx
p_keep_conv = tf.placeholder('float')
p_keep_hidden = tf.placeholder('float')
py_x = model(x, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=y))
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, axis=1)

batch_size = 128
test_size = 256

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100):
        train_batch = zip(range(0, len(tr_x), batch_size), range(batch_size, len(tr_x) + 1, batch_size))
        for start, end in train_batch:
            sess.run(train_op, feed_dict={x:tr_x[start: end], y: tr_y[start: end], p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_indices = np.arange(len(te_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0: test_size]
        print (i , np.mean(np.argmax(te_y[test_indices], axis=1) ==
                           sess.run(predict_op, feed_dict={x: te_x[test_indices], p_keep_conv: 1.0,
                                                           p_keep_hidden: 1.0})))
