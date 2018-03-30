
# coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
print 'ss'


i = 1;

li = [[1, 2], [1, 2], [1, 2]]
x= li[0] [:]
print x
import tensorflow as tf
t = tf.add(8, 9)

a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
sess = tf.Session()
print sess.run(c)

with tf.variable_scope('foo') as scope_foo:
    with tf.variable_scope('foo1'):
        x = tf.get_variable('var', initializer=[11])
        x2 = tf.Variable(name='var2', initial_value=[1, 2])
x = tf.Variable(name='var3', initial_value=[1, 2])

init = tf.global_variables_initializer()
sess.run(init)
print sess.run(x)

a = tf.constant([[1.0, 2.0], [1.0, 2.0]])
print sess.run(tf.nn.sigmoid(a))

a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])

b = tf.nn.dropout(a, 0.5, noise_shape=[1, 4])
print sess.run(b)
q = tf.FIFOQueue(3, 'float')
init = q.enqueue_many(([0.1, 0.2, 0.3], ))
x = q.dequeue()
y = x + 1
q_inc = q.enqueue([y])
sess.run(init)
quelen = sess.run(q.size())
for i in range(2):
    sess.run(q_inc)
quelen = sess.run(q.size())
for i in range(quelen):
    print sess.run(q.dequeue())

filename = './out.records'
print 'writing %s' % filename
writer = tf.python_io.TFRecordWriter(filename)
str = 'aaabbbccc结'
example = tf.train.Example(features=tf.train.Features(feature={
    'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[700, 200])),
    'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
    'str': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str]))
}))
writer.write(example.SerializeToString())
writer.close()

reader = tf.TFRecordReader()
file_name_queue = tf.train.string_input_producer([filename])
_, serialized_example = reader.read(file_name_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'height': tf.FixedLenFeature([2], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'str': tf.FixedLenFeature([], tf.string)
    }
)
# height = tf.decode_raw(features['height'], tf.int64)
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess=sess)

print sess.run(features['height'])
print sess.run(features['str'])
coord.request_stop()
coord.join(threads)





sess.close()


