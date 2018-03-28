from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
data_path = '../data/mnist'
mnist = input_data.read_data_sets(data_path, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, w) + b
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.image('input_image', tf.reshape(x, [-1, 28, 28, 1]), 10)
tf.summary.scalar('w', tf.reduce_mean(w))
tf.summary.scalar('b', tf.reduce_mean(b))
tf.summary.histogram('y', y)
tf.summary.scalar('cross_entropy' , cross_entropy)
tf.summary.histogram('cross_entropyh', cross_entropy)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()


sess = tf.Session()
train_writer = tf.summary.FileWriter('./log/train', sess.graph)
sess.run(tf.global_variables_initializer())
a = 'con'
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    summ, _ = sess.run([merged, train_step], feed_dict= {x: batch_xs, y_:batch_ys})
    train_writer.add_summary(summ, i)
    if i % 100 == 0:
        print 'cross_entropy: %s' % sess.run(cross_entropy, feed_dict= {x: batch_xs, y_:batch_ys})
        print 'accuracy: %s' % sess.run(accuracy, feed_dict= {x: batch_xs, y_:batch_ys})
        if a == 'con':
            a = raw_input('input: ')
        else:
            pass
train_writer.close()
sess.close()