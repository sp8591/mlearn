#!/bin/sh
python rnn_mnist_async.py --job_name='ps' \
--task_id=0 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'

python rnn_mnist_async.py --job_name='worker' \
--task_id=0 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'

python rnn_mnist_async.py --job_name='worker' \
--task_id=1 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'

------------------------------------------------

python rnn_mnist_sync.py --job_name='ps' \
--task_id=0 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'

python rnn_mnist_sync.py --job_name='worker' \
--task_id=0 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'

python rnn_mnist_sync.py --job_name='worker' \
--task_id=1 \
--ps_hosts='controller:2222' \
--worker_hosts='controller:2223,controller:2224'