#export PYTHONPATH=$PYTHONPATH:~/gentle/
import numpy as np 
import gentle
import wave
import string
import pickle
import struct 
import os
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
import subprocess
from scipy.io import wavfile
from tensorflow.contrib import rnn

learning_rate = 0.001
training_iters = 100000
batch_size = 100
display_step = 100

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
	return tf.nn.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])

class Preprocess_data:

	def __init__(self):
		self.vocabulary={}
		self.emb_length=0

	def add_to_vocab(self,word_list):

		for word in word_list:
			if self.vocabulary.has_key(word[0]):
				self.vocabulary[word[0]].append(word[1])
			else:
				self.vocabulary.update({word[0]:[word[1]]})


	def save_vocabulary(self,filename):
		with open(filename, 'wb') as f:
			pickle.dump([self.vocabulary,self.emb_length], f, pickle.HIGHEST_PROTOCOL)

	def load_vocabulary(self,filename):
		with open(filename, 'rb') as f:
			self.vocabulary,self.emb_length=pickle.load(f)

	def get_word_list(self,sentence,audio_file,load_old=0,load_from=''):

		word_list=[]

		resources = gentle.Resources()

		sentence=sentence.lower()
		sentence=sentence.translate(None, string.punctuation)
		sentence=sentence.decode('unicode_escape').encode('ascii','ignore')
		with gentle.resampled(audio_file) as wfile:
			if load_old and len(load_from)>0:
				print 'Loaded old result'
				with open(load_from, 'rb') as f:
					result=pickle.load(f)
			else:
				aligner = gentle.ForcedAligner(resources, sentence, nthreads=4)
				print 'Alignment in Progress'
				result = aligner.transcribe(wfile)
				print 'Alignment Complete'

			
			frameRate,wave_data = wavfile.read(wfile)
			for w in result.words:
				if w.start and w.end and w.end>w.start and w.alignedWord==w.word and w.word.isalpha():
					start = int(w.start*frameRate)
					end = int(w.end*frameRate)
					chunkData = wave_data[ start:end ]
					word_list.append([w.word,chunkData])	
		return word_list

	def cosine_dist(self,X,Y):
		if len(X)<len(Y):
			X=np.pad(X,(0,len(Y)-len(X) ),'constant')
		if len(Y)<len(X):
			Y=np.pad(Y,(0,len(X)-len(Y) ),'constant')
		val=cosine_similarity(X.reshape(1,-1),Y.reshape(1,-1))
		return (1.0-val[0][0])/2.0
	
	def generate_embeddings(self,word,window_size,disp):
		embeddings,emb,index=[],[],0
		emb=np.zeros(window_size-disp)
		
		while index<self.emb_length:
			if index+window_size < self.emb_length:
				emb=self.vocabulary[word][0][index:index+window_size]
			else:
				emb=self.vocabulary[word][0][index:self.emb_length]
				emb=np.pad(emb,(0,window_size-len(emb)),'constant')
			embeddings.append(emb)
			index+=disp
			emb=[]
		return embeddings

	def replaceBy_cluster_mean(self,word):
		if self.vocabulary.has_key(word)==0:
			print "ERROR: Word not found in vocabulary"
			return
		n = len(self.vocabulary[word])
		
		if n==1:	
			return
		elif n==2:
			p=np.random.rand()
			if p>0.5:
				del self.vocabulary[word][-1]
			else:
				del self.vocabulary[word][0]
		else:
			print "N=",n
			center=-1
			min_dist=99999999.9999
			for i in xrange(n-1):
				sum_dist=0.0
				for j in xrange(i+1,n):
					sum_dist+=self.cosine_dist( self.vocabulary[word][i],self.vocabulary[word][j] )
				if sum_dist<min_dist:
					min_dist=sum_dist
					center=i
			self.vocabulary[word][0] = self.vocabulary[word][center]
			del self.vocabulary[word][1:n]

	def align_vocab(self,mx=0):
		max_len=0
		if mx==0:
			max_word=0
			for k in self.vocabulary.keys():
				if len( self.vocabulary[k][0] )>max_len:
					max_len=len( self.vocabulary[k][0] )
				max_word=k

			print "Max word:",max_word," Length:",max_len
		else:
			max_len=mx

		for k in self.vocabulary.keys():
			l=len ( self.vocabulary[k][0] )
			if l<max_len:
				self.vocabulary[k][0]=np.pad(self.vocabulary[k][0],(0,max_len-l),'constant')
		self.emb_length=max_len

	def generate_vocab(self,words,mx=0):
		print "Generating tts"
		word_list=[]
		for w in words:
			print w
			audio_file='./vocab/'+w+'.wav'
			subprocess.call(["espeak", "-w"+audio_file, w])
			with gentle.resampled(audio_file) as wfile:
				frameRate,wave_data = wavfile.read(wfile)
				word_list.append([w,wave_data])
				#os.remove(wfile)
		self.add_to_vocab(word_list)
		self.align_vocab(mx)
		#self.save_vocabulary('tts_vocab')

def generate_test_data(filename,max_emb_len,window_size,disp):
	with gentle.resampled(filename) as wfile:
		frameRate,wave_data = wavfile.read(wfile)
	if len(wave_data)<max_emb_len:
		wave_data=np.pad(wave_data,(0,max_emb_len-len(wave_data)),'constant')
	else:
		print 'embedding too long'

	embeddings,emb,index=[],[],0
	emb=np.zeros(window_size-disp)
		
	while index<max_emb_len:
		if index+window_size < max_emb_len:
			emb=wave_data[index:index+window_size]
		else:
			emb=wave_data[index:max_emb_len]
			emb=np.pad(emb,(0,window_size-len(emb)),'constant')
		embeddings.append(emb)
		index+=disp
		emb=[]
	
	return embeddings

if __name__=="__main__":
	narated_obj = Preprocess_data()
	if os.path.exists('narated_vocab'):
		narated_obj.load_vocabulary('narated_vocab')
	else:
		fh  = open('full_text.txt')
		text = fh.readline()
		fh.close()
		narated_obj.add_to_vocab( narated_obj.get_word_list(text,'full_media.mp3',1,'narated_align_result') )
		for w in narated_obj.vocabulary.keys():
			narated_obj.replaceBy_cluster_mean(w)
			print w," done"
		narated_obj.align_vocab()
		narated_obj.save_vocabulary('narated_vocab')

	tts_obj = Preprocess_data()
	if os.path.exists('tts_vocab'):
		tts_obj.load_vocabulary('tts_vocab')
	else:
		tts_obj.generate_vocab(narated_obj.vocabulary.keys())


	itr=0
	lista=[]
	for sentence in open('sentences.txt'):
		random_words=[]
		sentence=sentence.lower()
		sentence=sentence.translate(None, string.punctuation)
		sentence=sentence.decode('unicode_escape').encode('ascii','ignore')
		for w in sentence.split(' '):
			if w in narated_obj.vocabulary.keys():
				random_words.append(w)
		if len(random_words)>0:
			print random_words
			test_obj=Preprocess_data()
			test_obj.generate_vocab(random_words,tts_obj.emb_length)
			data_inputs=[]
			data_outs=[]
			for w in test_obj.vocabulary.keys()  :
				embs = test_obj.generate_embeddings(w,2000,1000)
				data_inputs.append(embs)
				data_outs.append( [ 1 if w==nw else 0 for nw in sorted(narated_obj.vocabulary.keys()) ] )


			inputs=np.random.rand(len(random_words),len(data_inputs[0]),2000)
			outputs = np.random.rand( len(random_words), len(data_outs[0] ) )
			
			print inputs.shape,outputs.shape
			
			for i in xrange(len(random_words)):
				for j in xrange(len(data_inputs[0])):
					for k in xrange(2000):
						inputs[i][j][k]=data_inputs[i][j][k]
				for j in xrange( len(data_outs[0] ) ):
					outputs[i][j]=data_outs[i][j]

			lista.append([inputs,outputs])

	import tensorflow as tf
	n_input = 2000
	n_steps = len(data_inputs[0])
	n_hidden = 1000
	n_output = len(sorted(narated_obj.vocabulary.keys()))

	# tf Graph input
	x = tf.placeholder("float", [None, n_steps, n_input])
	y = tf.placeholder("float", [None, n_output])

	weights = {
		'out': tf.Variable(tf.random_normal([n_hidden, n_output]))
	}
	biases = {
		'out': tf.Variable(tf.random_normal([n_output]))
	}
	pred = RNN(x, weights, biases)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	init = tf.global_variables_initializer()
	with tf.Session() as sess: 
		sess.run(init)
		saver = tf.train.import_meta_graph('model2_rnn-3000.meta',clear_devices=True)
		saver.restore(sess,tf.train.latest_checkpoint('./'))
		for i in range(len(lista)):
			result=sess.run(pred,feed_dict={x: lista[i][0], y: lista[i][1]})
			print result.shape
			wave_d=[]
			for j in range(len(result)):
				print np.argmax(result[j]),np.argmax(lista[i][1][j])
				wave_d.append(narated_obj.vocabulary[sorted(narated_obj.vocabulary.keys())[ np.argmax(result[j]) ]  ][0])

			wave_d=np.asarray(wave_d)
			wave_d=wave_d.reshape(wave_d.shape[0]*wave_d.shape[1])
			wavfile.write(str(i)+'test.wav',8000,wave_d)


