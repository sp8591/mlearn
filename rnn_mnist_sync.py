import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
data_path = '../data/mnist'
lr = 0.001
training_iters = 100000
batch_size = 128

n_input = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', 'worker', '"ps" or "worker"')

tf.app.flags.DEFINE_string(
    'ps_hosts', 'controller:2222,controller:2223', 'hostname:port,...'
)
tf.app.flags.DEFINE_string(
    'worker_hosts', 'controller:2224,controller:2225', 'hostname:port,...'
)
tf.app.flags.DEFINE_integer(
    'task_id', 0, 'task Id'
)



def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x_in = tf.matmul(x, weights['in']) + biases['in']
    x_in = tf.reshape(x_in, [-1, n_steps, n_hidden_units])
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias = 1.0
                                             , state_is_tuple=True)
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32 )
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, x_in, initial_state=init_state, time_major=False)
    results = tf.matmul(final_state[1], weights['out']) + biases['out']
    return results



def main(argv=None):
    ps_hosts = flags.ps_hosts.split(',')
    worker_hosts = flags.worker_hosts.split(',')
    cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
    server = tf.train.Server(
        cluster, job_name=flags.job_name, task_index=flags.task_id
    )
    if flags.job_name == 'ps':
        server.join()
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    is_chief = (flags.task_id == 0)
    with tf.device(tf.train.replica_device_setter(
        worker_device='/job:worker/task:%d' % flags.task_id,
        cluster=cluster
    )):
        x = tf.placeholder(tf.float32, [None, n_steps, n_input])
        y = tf.placeholder(tf.float32, [None, n_classes])
        global_step = tf.Variable(0, trainable=False)
        weights = {
            'in': tf.Variable(tf.random_normal([n_input, n_hidden_units])),
            'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
        }
        pred = RNN(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        opt = tf.train.SyncReplicasOptimizer(
            tf.train.AdamOptimizer(lr),
            replicas_to_aggregate=len(worker_hosts),
            total_num_replicas=len(worker_hosts),
        )
        train_op = opt.minimize(cost, global_step=global_step)
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()
        init_op = tf.initialize_all_variables()
        summary_op = tf.summary.merge_all()

        if is_chief:
            chief_queue_runner = opt.get_chief_queue_runner()
            init_tokens_op = opt.get_init_tokens_op(0)


        sv = tf.train.Supervisor(
            is_chief=is_chief,
            init_op=init_op,
            logdir='./log',
            saver=saver,
            global_step=global_step,
            save_model_secs=60,
            summary_op=summary_op,
            save_summaries_secs=60,
        )
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
        )
        sess = sv.prepare_or_wait_for_session(
            server.target, config=sess_config
        )
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_tokens_op)

        step = 0
        start_time = time.time()
        while not sv.should_stop():
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape([batch_size, n_steps, n_input])
            _, global_step_value = sess.run(
                [train_op, global_step],
                feed_dict={
                x: batch_xs,
                y: batch_ys
            })
            if global_step_value >= 1000:
                break
            if step > 0 and step % 20 == 0:
                acc =  (sess.run(accuracy, feed_dict={
                    x: batch_xs,
                    y: batch_ys
                }))
                print 'global_step %d, from job %s, task_id %s, accurcy: %.3f' % (global_step_value, flags.task_id, flags.job_name, acc)
            step = step + 1


if __name__ == '__main__':
    tf.app.run()


