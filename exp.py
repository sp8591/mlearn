import numpy as np
import tensorflow as tf
a = np.array([[2, 3], [4, 5]])
b = np.array([2, 3])
print np.matmul(b, a)

cluster = tf.train.ClusterSpec({'local': ['localhost:2222', 'localhost:2223']})
a = tf.constant([1.0, 2.0, 3.0], shape=[3], name = 'a')
b = tf.constant([1.0, 2.0, 3.0], shape=[3], name = 'b')
c = a + b
server = tf.train.Server(cluster, job_name='local', task_index=1)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
print sess.run(c)
sess.close()
