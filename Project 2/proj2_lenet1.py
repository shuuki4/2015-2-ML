import numpy as np
import theano
import theano.tensor as T
import tarfile
import cPickle
import pickle
import gzip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import math
import random
from skimage import transform, exposure

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

# function that generates random augmentation data
def image_generate(img, ZCA_array) :
	timg = np.copy(img)
	timg = timg.transpose(1, 2, 0) # transpose to use skimage
	augnum = random.randrange(0, 5)
	
	if augnum==0 or augnum==1 : 
		# change nothing
		pass
	elif augnum==2 :
		# horizontal flip 
		for j in range(3) :
			timg[:,:,j] = np.fliplr(timg[:,:,j])
	elif augnum==3 :
		# random rotation of -15~15 degrees
		angle = random.random()*30-15
		timg = transform.rotate(timg/256.0, angle)
	elif augnum==4 :
		# gamma correction (luminance adjust) - random gamma 0.7~1.3
		gamma = random.random()*0.6+0.7
		timg = exposure.adjust_gamma(timg/256.0, gamma)

	timg = timg.transpose(2, 0, 1)
	# GCN, ZCA
	for i in range(3) :
		timg[i,:,:] -= np.mean(timg[i,:,:])
		timg[i,:,:] /= np.std(timg[i,:,:])
		timg[i,:,:] = np.dot(ZCA_array, timg[i,:,:].reshape(1024, 1)).reshape(32, 32)

	return timg

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
learning_rate = 0.05 # TODO : adaptive 
weight_decay = 0.0001
# input_shape determined
input_shape = (mini_batch_size, 3, 24, 24)

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

f = open('ZCAarray.txt', 'rb')
ZCA_array = pickle.load(f)
f.close()

# GCN and ZCA for val, test data
for i in range(val_num) :
	for j in range(3) :
		original_array = val_data[i,j,:,:]
		# GCN
		original_array = np.add(original_array, -np.mean(original_array))
		gcn_array = (original_array / np.std(original_array))
		val_data[i, j, :, :] = gcn_array
		flat = val_data[i,j,:,:].reshape(1024, 1)
		val_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)

for i in range(test_num) :
	for j in range(3) :
		original_array = test_data[i,j,:,:]
		# GCN
		original_array = np.add(original_array, -np.mean(original_array))
		gcn_array = (original_array / np.std(original_array))
		test_data[i, j, :, :] = gcn_array #np.dot(ZCA_array, gcn_array)
		flat = test_data[i,j,:,:].reshape(1024, 1)
		test_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)

# log
print "Building Layer Structure... %f" % (time.time()-start_time)

## build layer structure 

# input symbol variable
input = T.tensor4(name='input')
test_input = T.tensor4(name='test_input')

###layer construction : conv->pool->conv->pool->mlp
"""
# Conv, Pool layers
convlayer1 = ConvLayer.ConvLayer(input, input_shape, filter_shape=(16, 3, 5, 5)) # use 3*3 for convolution filter
poollayer1 = PoolLayer.PoolLayer(convlayer1.output, input_shape=(mini_batch_size, 16, 20, 20), pool_shape=(2, 2)) # use 2*2 for pool filter
convlayer2 = ConvLayer.ConvLayer(poollayer1.output, input_shape=(mini_batch_size, 16, 10, 10), filter_shape=(32, 16, 3, 3))
poollayer2 = PoolLayer.PoolLayer(convlayer2.output, input_shape=(mini_batch_size, 32, 8, 8), pool_shape=(2, 2))

# After these layers : 8 channels * (4*4) values per image
# use these variables to construct MLP
# debug_img = theano.shared(np.asarray((mini_batch_size, 8, 4, 4)))
mlp_input = T.reshape(poollayer2.output, (mini_batch_size, 32*4*4), ndim=2)
MLPlayer = MLP.MLP(mlp_input, input_shape=(mini_batch_size, 32*4*4), hidden_num=1400, output_num=10, p=0.5)
"""

# deep, deep layer with dropout
convlayer1 = ConvLayer.ConvLayer(input, input_shape, filter_shape=(64, 3, 3, 3), border_mode="full")
convlayer2 = ConvLayer.ConvLayer(convlayer1.output, input_shape=(mini_batch_size, 64, 26, 26), filter_shape=(64, 64, 3, 3))
poollayer1 = PoolLayer.PoolLayer(convlayer2.output, input_shape=(mini_batch_size, 64, 24, 24), pool_shape=(2,2))
dropout1 = Dropout.Dropout(poollayer1.output, input_shape=(mini_batch_size, 64, 12, 12), p=0.25)
convlayer3 = ConvLayer.ConvLayer(dropout1.output, input_shape=(mini_batch_size, 64, 12, 12), filter_shape=(128, 64, 3, 3), border_mode="full")
convlayer4 = ConvLayer.ConvLayer(convlayer3.output, input_shape=(mini_batch_size, 128, 14, 14), filter_shape=(128, 128, 3, 3))
poollayer2 = PoolLayer.PoolLayer(convlayer4.output, input_shape=(mini_batch_size, 128, 12, 12), pool_shape=(2,2))
dropout2 = Dropout.Dropout(poollayer2.output, input_shape=(mini_batch_size, 128, 6, 6), p=0.25)
convlayer5 = ConvLayer.ConvLayer(dropout2.output, input_shape=(mini_batch_size, 128, 6, 6), filter_shape=(256, 128, 3, 3))
convlayer6 = ConvLayer.ConvLayer(convlayer5.output, input_shape=(mini_batch_size, 256, 4, 4), filter_shape=(256, 256, 3, 3))

mlp_input = T.reshape(convlayer6.output, (mini_batch_size, 256*2*2), ndim=2)
MLPlayer = MLP.MLP(mlp_input, input_shape=(mini_batch_size, 256*2*2), hidden_num=700, output_num=10, p=0.5)

# After these layers : mini_batch_size * 10 tensor generated
# use 'cross-entropy' as a cost function
y = T.matrix('y') # real one-hot indexes
cost = T.nnet.categorical_crossentropy(MLPlayer.output, y).sum()

# gradient calculation
params = MLPlayer.params + convlayer6.params + convlayer5.params + convlayer4.params + convlayer3.params + convlayer2.params + convlayer1.params
paramins = MLPlayer.paramins + convlayer6.paramins + convlayer5.paramins + convlayer4.paramins + convlayer3.paramins + convlayer2.paramins + convlayer1.paramins
#params = MLPlayer.params + convlayer2.params + convlayer1.params
#paramins = MLPlayer.paramins + convlayer2.paramins + convlayer1.paramins
grad = T.grad(cost, params)
#updates= [(param_i, param_i-learning_rate*grad_i) for param_i, grad_i in zip(params, grad)]

# momentum learning
momentum = 0.2
updates = []

for param_i, grad_i, in_i in zip(params, grad, paramins) :
	prev_grad_i = theano.shared(np.zeros(param_i.get_value().shape, dtype=theano.config.floatX))
	nowgrad = momentum * prev_grad_i - learning_rate * grad_i / math.sqrt(in_i)
	updates.append((prev_grad_i, nowgrad))
	updates.append((param_i, param_i + nowgrad))

# functions
f = theano.function([input, y], cost, updates=updates) # for train
test_f = theano.function([input], MLPlayer.output) # for validation, test
debug_f = theano.function([input], [MLPlayer.input, MLPlayer.hidden, MLPlayer.output]) # for debug

#### Training Region ####

# parameters
max_iter = 120000 # one cycle = 800 times

# log
# print "Start Training... %f" % (time.time()-start_time)

for loop in range(max_iter) :
	"""
	# learning rate decay
	if loop%10==0 : learning_rate *= 0.9

	# log
	print "Loop #%d... " % (loop+1)
	"""
	if (loop*mini_batch_size)%train_num==0 :
		random_idx = np.random.permutation(train_num)

	# fetch appropriate 'mini_batch_size' training datas : randomly
	startidx = (loop*mini_batch_size)%train_num
	#nowinput = train_data[random_idx[startidx:startidx+mini_batch_size], :, :, :]
	nowinput = np.zeros((mini_batch_size, 3, 24, 24), dtype=theano.config.floatX)
	for i in range(mini_batch_size) :
		# take random 24x24 crop of image
		x_st = random.randrange(0, 9)
		y_st = random.randrange(0, 9)
		# random augmentation
		#nowinput[i,:,:,:] = train_data[random_idx[startidx+i],:,x_st:x_st+24,y_st:y_st+24]
		nowinput[i,:,:,:] = image_generate(train_data[random_idx[startidx+i],:,:,:], ZCA_array)[:,x_st:x_st+24,y_st:y_st+24]

	# make y data for this loop
	nowy = np.zeros((mini_batch_size, 10), dtype=theano.config.floatX)
	for idx in range(mini_batch_size) :
		nowy[idx, train_label[random_idx[startidx+idx]]]=1.0
	# proceed!
	#Ww, Bb = convlayer1.W.get_value(), convlayer1.b.get_value()
	nowcost = f(nowinput, nowy)
	#print Ww[:, :, :]
	#print Bb[:]
	#print "Now cost : %f" % (nowcost)
	
	# check validation data error rate
	if loop%400==0 and loop>0:
		print "Validation Data Check for Loop %d !" % loop
		check_valnum = 5000
		errorcnt = 0.0
		for i in range(check_valnum/mini_batch_size) :
			check_startidx = (i*mini_batch_size)%check_valnum
			check_nowinput = val_data[check_startidx:check_startidx+mini_batch_size, :, 4:28, 4:28]
			result = test_f(check_nowinput).argmax(axis=1)
			#print test_f(check_nowinput)
			for j in range(mini_batch_size) :
				if result[j] != val_label[check_startidx+j] :
					errorcnt += 1.0
			"""
			if i==0 :
				in_, hid_, out_ = debug_f(check_nowinput)
				print in_[0,:]
				print hid_[0,:]
				print out_[0:5,:]
				print val_label[check_startidx:check_startidx+5]
			"""
		print "Error Rate : %f" % (errorcnt/check_valnum)
	
	# learning_rate drop
	if loop%4000==0 and loop>=4000 and loop<=36000 :
		learning_rate *= 0.6
	
	if loop%8000==0 and loop>0 :
		errorcnt = 0.0
		conf_matrix = np.zeros((10, 10), dtype=int)
		print "Test Data Check for Loop %d !" % loop
		for i in range(test_num/mini_batch_size) :
			startidx = (i*mini_batch_size)%test_num
			result = np.zeros((mini_batch_size, 5), dtype=int)

			nowinput = test_data[startidx:startidx+mini_batch_size, :, 4:28, 4:28]
			result[:,0] = test_f(nowinput).argmax(axis=1)
			nowinput = test_data[startidx:startidx+mini_batch_size, :, 0:24, 0:24]
			result[:,1] = test_f(nowinput).argmax(axis=1)
			nowinput = test_data[startidx:startidx+mini_batch_size, :, 0:24, 8:32]
			result[:,2] = test_f(nowinput).argmax(axis=1)
			nowinput = test_data[startidx:startidx+mini_batch_size, :, 8:32, 0:24]
			result[:,3] = test_f(nowinput).argmax(axis=1)
			nowinput = test_data[startidx:startidx+mini_batch_size, :, 8:32, 8:32]
			result[:,4] = test_f(nowinput).argmax(axis=1)

			for j in range(mini_batch_size) :
				vote = np.bincount(result[j,:]).argmax()
				if vote != test_label[startidx+j] :
					errorcnt += 1.0
				conf_matrix[test_label[startidx+j], vote]+=1

		print "Test Data Error Rate : %f" % (errorcnt/test_num)
		print "Confusion Matrix : "
		print conf_matrix		

# final check : test data
print "Test Data!!!"
errorcnt = 0.0
conf_matrix = np.zeros((10,10), dtype=int)
for i in range(test_num/mini_batch_size) :
	startidx = (i*mini_batch_size)%test_num
	result = np.zeros((mini_batch_size, 5), dtype=int)

	nowinput = test_data[startidx:startidx+mini_batch_size, :, 4:28, 4:28]
	result[:,0] = test_f(nowinput).argmax(axis=1)
	nowinput = test_data[startidx:startidx+mini_batch_size, :, 0:24, 0:24]
	result[:,1] = test_f(nowinput).argmax(axis=1)
	nowinput = test_data[startidx:startidx+mini_batch_size, :, 0:24, 8:32]
	result[:,2] = test_f(nowinput).argmax(axis=1)
	nowinput = test_data[startidx:startidx+mini_batch_size, :, 8:32, 0:24]
	result[:,3] = test_f(nowinput).argmax(axis=1)
	nowinput = test_data[startidx:startidx+mini_batch_size, :, 8:32, 8:32]
	result[:,4] = test_f(nowinput).argmax(axis=1)

	for j in range(mini_batch_size) :
		vote = np.bincount(result[j,:]).argmax()
		if vote != test_label[startidx+j] :
			errorcnt += 1.0
		conf_matrix[test_label[startidx+j], vote]+=1

print "Test Data Error Rate : %f" % (errorcnt/test_num)
print "Confusion Matrix : "
print conf_matrix