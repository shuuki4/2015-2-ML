import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import time
import math

# define convolution layer for CNN (only convolution)

class ConvLayer(object) :
	def __init__(self, input, input_shape, filter_shape, border_mode="valid") :
		# input : theano symbolic variable of input, 4D tensor
		# input_shape : shape of input / (minibatch size, input channel num, image height, image width)
		# filter_shape : shape of filter / (# of new channels to make, input channel num, filter height, filter width)

		# initialize W (weight) randomly
		rng = np.random.RandomState(int(time.time()))
		w_bound = math.sqrt(filter_shape[1] * filter_shape[2] * filter_shape[3])
		self.W1 = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=filter_shape), dtype=theano.config.floatX), name='W', borrow=True)
		self.W2 = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=filter_shape), dtype=theano.config.floatX), name='W', borrow=True)
		self.W3 = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=filter_shape), dtype=theano.config.floatX), name='W', borrow=True)
		
		# initialize b (bias) with zeros
		self.b1 = theano.shared(np.asarray(np.zeros(filter_shape[0],), dtype=theano.config.floatX), name='b', borrow=True)
		self.b2 = theano.shared(np.asarray(np.zeros(filter_shape[0],), dtype=theano.config.floatX), name='b', borrow=True)
		self.b3 = theano.shared(np.asarray(np.zeros(filter_shape[0],), dtype=theano.config.floatX), name='b', borrow=True)

		# convolution & sigmoid calculation
		#self.conv_out = conv.conv2d(input, self.W, image_shape=input_shape, filter_shape=filter_shape)
		#self.output = 1.7159*T.tanh((self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))*(2.0/3.0))

		# maxout : 3
		out1 = conv.conv2d(input, self.W1, image_shape=input_shape, filter_shape=filter_shape, border_mode=border_mode) + self.b1.dimshuffle('x', 0, 'x', 'x')
		out2 = conv.conv2d(input, self.W2, image_shape=input_shape, filter_shape=filter_shape, border_mode=border_mode) + self.b2.dimshuffle('x', 0, 'x', 'x')
		out3 = conv.conv2d(input, self.W3, image_shape=input_shape, filter_shape=filter_shape, border_mode=border_mode) + self.b3.dimshuffle('x', 0, 'x', 'x')

		self.output = T.maximum(out1, T.maximum(out2, out3))

		# save parameter of this layer for back-prop convinience
		self.params = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
		insize = input_shape[1] * input_shape[2] * input_shape[3]
		self.paramins = [insize, insize, insize, insize, insize, insize]