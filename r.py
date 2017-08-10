from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split

# Parameters
learning_rate = 0.001
training_iters = 5000
batch_size = 100
display_step = 200

def RNN(x, weights, biases):

	# Prepare data shape to match `rnn` function requirements
	# Current data input shape: (batch_size, n_steps, n_input)
	# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

	# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
	x = tf.unstack(x, n_steps, 1)

	# Define a lstm cell with tensorflow
	lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

	# Get lstm cell output
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	# Linear activation, using rnn inner loop last output
	return tf.matmul(outputs[-1], weights['out']) + biases['out']

#load train data
fct=5
train_x,train_y=[],[]
for i in range(fct):
	with open('datafile_'+str(i), 'rb') as f:
		tx,ty=pickle.load(f)
		#print(type(tx),type(ty))
		train_x.append(tx)
		train_y.append(ty)
train_x,train_y=np.asarray(train_x),np.asarray(train_y)
train_x=train_x.reshape(train_x.shape[0]*train_x.shape[1],train_x.shape[2],train_x.shape[3])
train_y=train_y.reshape( train_y.shape[0]*train_y.shape[1],train_y.shape[2])
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.25, random_state=29)

print(X_train.shape,y_train.shape)
# Network Parameters
n_input = X_train.shape[2]
n_steps = X_train.shape[1]
n_hidden = 300
max_items= X_train.shape[0]
n_output = y_train.shape[1]

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_output])

# Define weights
weights = {
	'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
}
biases = {
	'out': tf.Variable(tf.random_normal([n_output]))
}

pred = RNN(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

def next_batch(step1,batch_size,vflag=0):
	if vflag:
		rx,ry=X_val[(step1-1)*batch_size:step1*batch_size,:,:],y_val[(step1-1)*batch_size:step1*batch_size,:]
	else:
		rx,ry=X_train[(step1-1)*batch_size:step1*batch_size,:,:],y_train[(step1-1)*batch_size:step1*batch_size,:]
	perm=np.random.permutation(len(rx))
	rx=rx[perm]
	ry=ry[perm]
	return rx,ry

#saver = tf.train.Saver()
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	step,step1 = 1,1
	while step * batch_size < training_iters:
		if step1*batch_size>max_items:
			step1=1
		batch_x,batch_y=next_batch(step1,batch_size)

		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		#saver.save(sess, 'model2_rnn',global_step=3000)
		acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
		loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                if (step*batch_size)%display_step ==0:
					print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))
		step  += 1
		step1 += 1
	print("Optimization Finished!")
	
	print ('validation:')
	step,step1 = 1,1
	while step * batch_size < training_iters/2:
		if step1*batch_size>max_items:
			step1=1
		batch_x,batch_y=next_batch(step1,batch_size,1)

		sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
		#saver.save(sess, 'model2_rnn',global_step=3000)
		acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
		loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                if (step*batch_size)%display_step ==0:
					print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))
		step  += 1
		step1 += 1
	
	print("Optimization Finished!")

	print( "Testing:")
	tot,pas=0,0
	bx,by=next_batch(step1,10)
	for i in range(11):
		if i==10:
			tx,ty=bx,by
		else:
			with open('stest'+str(i), 'rb') as f:
				tx,ty=pickle.load(f)
		result=sess.run(pred,feed_dict={x: tx, y: ty})
		acc = sess.run(accuracy, feed_dict={x: tx, y: ty})
		loss = sess.run(cost, feed_dict={x: tx, y: ty})
		print(" Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= "+"{:.5f}".format(acc))
		for j in range(len(result)):
			tot+=1
			pred_val=softmax(result[j])
			if np.argmax(pred_val)==np.argmax(ty[j]):
				pas+=1
			print (np.argmax(pred_val),np.argmax(ty[j]))
	print ("Accuracy:",pas/float(tot))
