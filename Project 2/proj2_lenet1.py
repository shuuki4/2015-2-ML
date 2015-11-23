import numpy as np
import theano
import theano.tensor as T
import tarfile
import cPickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import math

import ConvLayer
import PoolLayer
import MLP
import Dropout

# function to load specific data from cifar-10 file
def load(fo, filenameidx) :
	filenamestr = 'cifar-10-batches-py/'
	if(filenameidx==1) : 
		filenamestr += 'data_batch_1'
	elif(filenameidx==2) : 
		filenamestr += 'data_batch_2'
	elif(filenameidx==3) :
		filenamestr += 'data_batch_3'
	elif(filenameidx==4) :
		filenamestr += 'data_batch_4'
	elif(filenameidx==5) :
		filenamestr += 'data_batch_5'
	elif(filenameidx==6) :
		filenamestr += 'test_batch'

	return cPickle.load(fo.extractfile(filenamestr))


######################################################################################
##########							REAL RUN AREA					##################
######################################################################################

## basic variables. 
# _data = 4D numpy array with (image number, channel number, image height, image width)
# _label = python list with image labels 0~9
train_data = np.empty(shape=(0,0))
train_label = np.empty(shape=(0,))
val_data = np.empty(shape=(0,0))
val_label = np.empty(shape=(0,))
test_data = np.empty(shape=(0,0))
train_label = np.empty(shape=(0,))

## basic numbers.
# number of each sets' data
train_num = 40000
val_num = 10000
test_num = 10000

# other variables
mini_batch_size = 50
learning_rate = 0.03 # TODO : adaptive 
# input_shape determined
input_shape = (mini_batch_size, 3, 32, 32)

# start time record
start_time = time.time()

# log
print "Starting to fetch data... %f" % (time.time()-start_time)

# data fetch from tar.gz file
fo = tarfile.open("cifar-10-python.tar.gz", 'r:gz')
for i in range(1, 7) :
	dict = load(fo, i)
	if i<=4 : # put into train data
		if train_data.size==0 : 
			train_data = dict['data'].reshape(10000, 3, 32, 32)
			train_label = dict['labels']
		else :
			train_data = np.vstack([train_data, dict['data'].reshape(10000, 3, 32, 32)])
			train_label += dict['labels']
	elif i==5 : # put into validation data
		val_data = dict['data'].reshape(10000, 3, 32, 32)
		val_label = dict['labels']
	else : # put into test data
		test_data = dict['data'].reshape(10000, 3, 32, 32)
		test_label = dict['labels']
fo.close();

# log
print "Start Preprocessing..."

train_data = np.asarray(train_data, dtype=theano.config.floatX)
val_data = np.asarray(val_data, dtype=theano.config.floatX)
test_data = np.asarray(test_data, dtype=theano.config.floatX)

corr_array = np.zeros((3, 1024, 1024))

# GCN, ZCA
for i in range(train_num) :
	if i%1000 == 0 : print "Transforming %d" % (i+1)
	for j in range(3) :
		original_array = train_data[i,j,:,:]
		# GCN
		original_array = np.add(original_array, -np.mean(original_array))
		gcn_array = (original_array / np.std(original_array))
		train_data[i, j, :, :] = gcn_array #np.dot(ZCA_array, gcn_array)

		# prepare for ZCA
		flat = gcn_array.reshape(1024, 1)
		corr_array[j,:,:] += np.dot(flat, flat.T)

for i in range(val_num) :
	if i%1000 == 0 : print "Transforming %d" % (i+1)
	for j in range(3) :
		original_array = val_data[i,j,:,:]
		# GCN
		original_array = np.add(original_array, -np.mean(original_array))
		gcn_array = (original_array / np.std(original_array))
		val_data[i, j, :, :] = gcn_array #np.dot(ZCA_array, gcn_array)

for i in range(test_num) :
	if i%1000 == 0 : print "Transforming %d" % (i+1)
	for j in range(3) :
		original_array = test_data[i,j,:,:]
		# GCN
		original_array = np.add(original_array, -np.mean(original_array))
		gcn_array = (original_array / np.std(original_array))
		test_data[i, j, :, :] = gcn_array #np.dot(ZCA_array, gcn_array)

# svd & zca
epsilon = 0.001
corr_array /= train_num
for j in range(3) :
	print "SVD of filter %d.." % (j+1)
	U, s, V = np.linalg.svd(corr_array[j,:,:])
	ZCA_array = np.dot(np.dot(U, np.diagflat(np.diag(1.0/np.sqrt(np.diag(s)+epsilon)))), U.T)
	for i in range(train_num) :
		flat = train_data[i,j,:,:].reshape(1024, 1)
		train_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)
	for i in range(val_num) :
		flat = val_data[i,j,:,:].reshape(1024, 1)
		val_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)
	for i in range(test_num) :
		flat = test_data[i,j,:,:].reshape(1024, 1)
		test_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)


# log
print "Building Layer Structure... %f" % (time.time()-start_time)

## build layer structure : LeNet model
# Input (3 channel) -> Conv (4 channel) -> Max-Pool -> Conv (6 channel) -> Max-Pool -> MLP

# input symbol variable
input = T.tensor4(name='input')
test_input = T.tensor4(name='test_input')

###layer construction : conv->pool->conv->pool->mlp
"""
# Conv, Pool layers
convlayer1 = ConvLayer.ConvLayer(input, input_shape, filter_shape=(4, 3, 3, 3)) # use 3*3 for convolution filter
poollayer1 = PoolLayer.PoolLayer(convlayer1.output, input_shape=(mini_batch_size, 4, 30, 30), pool_shape=(2, 2)) # use 2*2 for pool filter
convlayer2 = ConvLayer.ConvLayer(poollayer1.output, input_shape=(mini_batch_size, 4, 15, 15), filter_shape=(6, 4, 4, 4))
poollayer2 = PoolLayer.PoolLayer(convlayer2.output, input_shape=(mini_batch_size, 6, 12, 12), pool_shape=(2, 2))
convlayer3 = ConvLayer.ConvLayer(poollayer2.output, input_shape=(mini_batch_size, 6, 6, 6), filter_shape=(8, 6, 3, 3))

# After these layers : 8 channels * (4*4) values per image
# use these variables to construct MLP
debug_img = theano.shared(np.asarray((mini_batch_size, 8, 4, 4)))
mlp_input = T.reshape(convlayer3.output, (mini_batch_size, 8*4*4), ndim=2)
MLPlayer = MLP.MLP(mlp_input, input_shape=(mini_batch_size, 8*4*4), hidden_num=100, output_num=10)
"""
# deep, deep layer with dropout
convlayer1 = ConvLayer.ConvLayer(input, input_shape, filter_shape=(64, 3, 3, 3))
convlayer2 = ConvLayer.ConvLayer(convlayer1.output, input_shape=(mini_batch_size, 64, 30, 30), filter_shape=(64, 64, 3, 3))
poollayer1 = PoolLayer.PoolLayer(convlayer2.output, input_shape=(mini_batch_size, 64, 28, 28), pool_shape=(2,2))
dropout1 = Dropout.Dropout(poollayer1.output, input_shape=(mini_batch_size, 64, 14, 14), p=0.25)
convlayer3 = ConvLayer.ConvLayer(dropout1.output, input_shape=(mini_batch_size, 64, 14, 14), filter_shape=(128, 64, 3, 3))
convlayer4 = ConvLayer.ConvLayer(convlayer3.output, input_shape=(mini_batch_size, 128, 12, 12), filter_shape=(128, 128, 3, 3))
poollayer2 = PoolLayer.PoolLayer(convlayer4.output, input_shape=(mini_batch_size, 128, 10, 10), pool_shape=(2,2))
dropout2 = Dropout.Dropout(poollayer2.output, input_shape=(mini_batch_size, 128, 5, 5), p=0.25)
convlayer5 = ConvLayer.ConvLayer(dropout2.output, input_shape=(mini_batch_size, 128, 5, 5), filter_shape=(256, 128, 3, 3))
convlayer6 = ConvLayer.ConvLayer(convlayer5.output, input_shape=(mini_batch_size, 256, 3, 3), filter_shape=(256, 256, 3, 3))

mlp_input = T.reshape(convlayer6.output, (mini_batch_size, 256*1*1), ndim=2)
MLPlayer = MLP.MLP(mlp_input, input_shape=(mini_batch_size, 256*1*1), hidden_num=200, output_num=10, p=0.3)

# After these layers : mini_batch_size * 10 tensor generated
# use 'cross-entropy' as a cost function
y = T.matrix(name='y') # real one-hot result
cost = T.nnet.categorical_crossentropy(MLPlayer.output, y).sum()

# gradient calculation
params = MLPlayer.params + convlayer6.params + convlayer5.params + convlayer4.params + convlayer3.params + convlayer2.params + convlayer1.params
paramins = MLPlayer.paramins + convlayer6.paramins + convlayer5.paramins + convlayer4.paramins + convlayer3.paramins + convlayer2.paramins + convlayer1.paramins
grad = T.grad(cost, params)
#updates= [(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params, grad)]

# momentum learning
momentum = 0.5
updates = []
for param_i, grad_i, in_i in zip(params, grad, paramins) :
	prev_grad_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
	nowgrad = momentum * prev_grad_i - learning_rate * grad_i / math.sqrt(in_i)
	updates.append((prev_grad_i, nowgrad))
	updates.append((param_i, param_i + nowgrad))

# functions
f = theano.function([input, y], cost, updates=updates) # for train
test_f = theano.function([input], MLPlayer.output) # for validation, test
#debug_f = theano.function([input], [MLPlayer.hidden, MLPlayer.output]) # for debug

#### Training Region ####

# parameters
max_iter = 32000 # one cycle = 800 times

# log
# print "Start Training... %f" % (time.time()-start_time)

for loop in range(max_iter) :
	"""
	# learning rate decay
	if loop%10==0 : learning_rate *= 0.9

	# log
	print "Loop #%d... " % (loop+1)
	"""
	# fetch appropriate 'mini_batch_size' training datas
	startidx = (loop*mini_batch_size)%train_num
	nowinput = train_data[startidx:startidx+mini_batch_size, :, :, :]
	# make y data for this loop
	nowy = np.zeros((mini_batch_size, 10), dtype=theano.config.floatX)
	for idx in range(mini_batch_size) :
		nowy[idx, train_label[startidx+idx]]=1.0
	# proceed!
	#Ww, Bb = convlayer1.W.get_value(), convlayer1.b.get_value()
	nowcost = f(nowinput, nowy)
	#print Ww[:, :, :]
	#print Bb[:]
	#print "Now cost : %f" % (nowcost)

	# check validation data error rate
	if loop%200==0 and loop>0:
		print "Validation Data Check for Loop %d !" % loop
		check_valnum = 5000
		errorcnt = 0.0
		for i in range(check_valnum/mini_batch_size) :
			check_startidx = (i*mini_batch_size)%check_valnum
			check_nowinput = val_data[check_startidx:check_startidx+mini_batch_size, :, :, :]
			result = test_f(check_nowinput).argmax(axis=1)
			#print test_f(check_nowinput)
			for j in range(mini_batch_size) :
				if result[j] != val_label[check_startidx+j] :
					errorcnt += 1.0
		print "Error Rate : %f" % (errorcnt/check_valnum)

	# learning_rate drop
	if loop%4000==0 and loop>0 :
		learning_rate *= 0.7

# final check : test data
print "Test Data!!!"
errorcnt = 0.0
for i in range(test_num/mini_batch_size) :
	startidx = (i*mini_batch_size)%test_num
	nowinput = test_data[startidx:startidx+mini_batch_size, :, :, :]
	result = test_f(nowinput).argmax(axis=1)
	for j in range(mini_batch_size) :
		if result[j] != test_label[startidx+j] :
			errorcnt += 1.0
print "Test Data Error Rate : %f" % (errorcnt/test_num)
