# -*- coding:utf-8 -*- 
from __future__ import print_function
__Developer__ = 'Ssong'
__Version__ = 'For testing'

import tensorflow as tf 
import sys, time
import numpy as np

parameter_servers = ['#.#.#.#:2222'] # Please input your ip address for a parameter server. 
workers = ['#.#.#.#:2223', 
	'#.#.#.#:2224', 
	'#.#.#.#:2225', 
	'#.#.#.#:2226'] # Please input your ip address for worker nodes. 

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers}) # The test that experiment between two parameter and multiple client environment will be progressed.

# Setting the input flags 
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS 

# Start a server for a specific task 
server = tf.train.Server(cluster, 
			job_name = FLAGS.job_name, 
			task_index = FLAGS.task_index)

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)


batch_size = 200 
iter_num = int(mnist.train.num_examples / batch_size) # 275 
learning_rate = 1e-4
epochs = 10 
logs_path = "/tmp/mnist/CNN" 


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

training_count_per_epochs = 100 
logs_path = "/tmp/mnist/CNN" 

from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def get_session(sess):
	session = sess
	while type(session).__name__ != 'Session':
		session = session._sess
	return session


if FLAGS.job_name == "ps":
	server.join()
elif FLAGS.job_name == "worker":
	with tf.device(tf.train.replica_device_setter(
		worker_device = "/job:worker/task:%d" % FLAGS.task_index,
		cluster=cluster)):
		global_step = tf.contrib.framework.get_or_create_global_step()

		with tf.name_scope('input'):
			x = tf.placeholder('float', shape=[None, 784], name='x-input') # It will be activated by execution of session object
			y_ = tf.placeholder('float', shape=[None, 10], name='y-input')
			x_image = tf.reshape(x, [-1, 28, 28, 1])

		tf.set_random_seed(1)
		with tf.name_scope('weights'):
			W1_conv = weight_variable([5,5,1,32]) # 5x5x1(channel) filter / output 32 
			W2_conv = weight_variable([5,5,32,64]) # 5x5x1(channel) filter / output 64 
			W3_conv = weight_variable([7 * 7 * 64, 1024]) # output 1024 
			W4_lastlayer = weight_variable([1024,10])

		with tf.name_scope('biases'):
			B1_conv = bias_variable([32]) # Bias 
			B2_conv = bias_variable([64]) # Bias 
			B3_conv = bias_variable([1024]) # Bias 
			B4_lastlayer = bias_variable([10])

		with tf.name_scope('softmax'):
			activation1_conv = tf.nn.relu(conv2d(x_image, W1_conv) + B1_conv)
			pool1_conv = max_pool_2x2(activation1_conv)
			activation2_conv = tf.nn.relu(conv2d(pool1_conv, W2_conv) + B2_conv)
			pool2_conv = max_pool_2x2(activation2_conv)
			p2conv_reshape = tf.reshape(pool2_conv, [-1,7*7*64])
			fully_step = tf.nn.relu(tf.matmul(p2conv_reshape, W3_conv) + B3_conv)
			keep_prob = tf.placeholder('float')
			fully_drop = tf.nn.dropout(fully_step, keep_prob)
			y_conv = tf.nn.softmax(tf.matmul(fully_drop, W4_lastlayer) + B4_lastlayer)
		

		with tf.name_scope('cross_entropy'):
			cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

		with tf.name_scope('train'):
			grad_op = tf.train.AdamOptimizer(learning_rate)
			train_op = grad_op.minimize(cross_entropy, global_step=global_step)
		

		with tf.name_scope('Accuracy'):
			correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

		tf.summary.scalar("cost", cross_entropy)
		tf.summary.scalar("accuracy", accuracy)
		summary_op = tf.summary.merge_all()

		#init_op = tf.initialize_all_variables()
		init_op = tf.global_variables_initializer()
		print ('Variables initialized...')
	
	
	# This codes follow between-graph replication and asynchronous traning method # 
	iteration_count = 100
	epochs_count = 0 
	hooks = [tf.train.StopAtStepHook(last_step=iter_num * epochs)]

	# Temp option parameter
	begin_time = time.time()

	# The saver object creating 
	saver = tf.train.Saver()
	
	with tf.train.MonitoredTrainingSession(master=server.target, 
						is_chief=(FLAGS.task_index == 0),
						hooks=hooks) as mon_sess:
		start_time = time.time()
		count = 0 
		while not mon_sess.should_stop():
			elapsed_time = time.time() - start_time 
			count += 1 
			#mon_sess.run(tf.global_variables_initializer())
			batch = mnist.train.next_batch(batch_size)
			_, cost, summary, step = mon_sess.run([train_op, cross_entropy, summary_op, global_step], 
						feed_dict = { x: batch[0], y_: batch[1], keep_prob: 0.5})

			print ('Step : %d,' % (step+1),
				'Epoch : %2d,' % (epochs_count),
				'Cost : %.8f,' % cost,
				'AvgTime : %3.2fms' % float(elapsed_time*1000/training_count_per_epochs))
			if count % iter_num  == 0:
				epochs_count += 1 
				
				print ('Test-Accuracy : %2.9f' % mon_sess.run(accuracy, feed_dict= { x : mnist.test.images, y_ : mnist.test.labels, keep_prob:1.0}))

			
		if FLAGS.job_name == "worker" and FLAGS.task_index == 0 :
			print ("A worker of zero in index is saving the trained datas")
			
	#sv.stop()
	processing_time = time.time() - begin_time
	print ("Total Time : %3.4fs" % float(processing_time))
	print ("Done")
		
				
