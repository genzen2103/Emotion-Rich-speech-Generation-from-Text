from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 10

def BiRNN(x, weights, biases):

	# Prepare data shape to match `bidirectional_rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, n_steps, 1)

	# Define lstm cells with tensorflow
	# Forward direction cell
	lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
	# Backward direction cell
	lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	# Get lstm cell output
	try:
		outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
											  dtype=tf.float32)
	except Exception: # Old TensorFlow version only returns outputs not states
		outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
										dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

# Network Parameters
n_input = 2000 
n_steps = 12
n_hidden = 1000

#load train data
with open('datafile_8_6_17', 'rb') as f:
	train_x,train_y=pickle.load(f)
	X_train, X_test, y_train, y_test = train_test_split(train_x,train_y, test_size=0.35, random_state=21)

print(train_x.shape,train_y.shape)

max_items=train_x.shape[0]
n_output = max_items

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])

# Define weights
weights = {
	# Hidden layer weights => 2*n_hidden because of forward + backward cells
	'out': tf.Variable(tf.random_normal([2*n_hidden, n_output]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_output]))
}

pred = BiRNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
def next_batch(step1,batch_size):
	return X_train[(step1-1)*batch_size:step1*batch_size,:,:],y_train[(step1-1)*batch_size:step1*batch_size,:]

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step,step1 = 1,1

	while step * batch_size < training_iters:
		if step1*batch_size>max_items:
			step1=1
		batch_x,batch_y=next_batch(step1,batch_size)
		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
		loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
		print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))
		step += 1
		step1+=1
	print("Optimization Finished!")

	print("Testing Accuracy:",sess.run(accuracy, feed_dict={x: X_test, y: y_test}))
