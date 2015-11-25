import numpy as np
import theano
import theano.tensor as T
import math
import time

## define MLP Layer
# input -> hidden layer -> output layer, sigmoid as an hidden activation function
class MLP(object) :
	def __init__(self, input, input_shape, hidden_num, output_num, p) :
		# input : theano symbolic variable of input, 2D tensor 
		# input_shape : shape of input / (mini-batch size, vector length)
		# hidden_num : number of hidden layer nodes
		# output_num : number of output layer nodes, in CIFAR-10 case : 10
		# p : dropconnect rate

		input_num = input_shape[1]

		# initialize W1, W2 (input->hidden, hidden->output) randomly
		rng = np.random.RandomState(int(time.time()))
		w1_bound = math.sqrt(input_num)
		w2_bound = math.sqrt(hidden_num)

		self.W11 = theano.shared(np.asarray(rng.uniform(low=-1.0/w1_bound, high=1.0/w1_bound, size=(input_num, hidden_num)), dtype=theano.config.floatX), name='W11', borrow=True)
		self.W12 = theano.shared(np.asarray(rng.uniform(low=-1.0/w1_bound, high=1.0/w1_bound, size=(input_num, hidden_num)), dtype=theano.config.floatX), name='W12', borrow=True)
		self.W13 = theano.shared(np.asarray(rng.uniform(low=-1.0/w1_bound, high=1.0/w1_bound, size=(input_num, hidden_num)), dtype=theano.config.floatX), name='W13', borrow=True)

		self.W2 = theano.shared(np.asarray(rng.uniform(low=-1.0/w2_bound, high=1.0/w2_bound, size=(hidden_num, output_num)), dtype=theano.config.floatX), name='W2', borrow=True)

		# initialize b1, b2 (input->hidden, hidden->output) randomly
		self.b11 = theano.shared(np.asarray(np.zeros(hidden_num,), dtype=theano.config.floatX), name='b11', borrow=True)
		self.b12 = theano.shared(np.asarray(np.zeros(hidden_num,), dtype=theano.config.floatX), name='b12', borrow=True)
		self.b13 = theano.shared(np.asarray(np.zeros(hidden_num,), dtype=theano.config.floatX), name='b13', borrow=True)

		self.b2 = theano.shared(np.asarray(np.zeros(output_num,), dtype=theano.config.floatX), name='b2', borrow=True)

		# DropConnect
		srng = T.shared_randomstreams.RandomStreams(int(time.time()))
		select_array1 = T.cast(srng.binomial(n=1, p=1-p, size=(input_num, hidden_num)), theano.config.floatX)
		select_array2 = T.cast(srng.binomial(n=1, p=1-p, size=(hidden_num, output_num)), theano.config.floatX)
		select_vec1 = T.cast(srng.binomial(n=1, p=1-p, size=(hidden_num,)), theano.config.floatX)
		select_vec2 = T.cast(srng.binomial(n=1, p=1-p, size=(output_num,)), theano.config.floatX)

		# possible hidden calculation
		hid1 = T.dot(input, self.W11*select_array1)+self.b11*select_vec1
		hid2 = T.dot(input, self.W12*select_array1)+self.b12*select_vec1
		hid3 = T.dot(input, self.W13*select_array1)+self.b13*select_vec1

		# maxout, softmax
		self.hidden = T.maximum(hid1, T.maximum(hid2, hid3))
		#self.output = T.nnet.softmax(T.dot(self.hidden, self.W2*select_array2)+self.b2*select_vec2)
		x = T.dot(self.hidden, self.W2*select_array2)+self.b2*select_vec2
		x_prime = x - x.max(axis=1, keepdims=True)
		x_prime2 = x_prime - T.log(T.sum(T.exp(x_prime),axis=1,keepdims=True))
		self.output = T.exp(x_prime2)

		# save parameter of this layer for back-prop convinience
		self.params = [self.W2, self.b2, self.W11, self.W12, self.W13, self.b11, self.b12, self.b13]
		self.paramins = [hidden_num, hidden_num, input_num, input_num, input_num, input_num, input_num, input_num]
		self.input = input