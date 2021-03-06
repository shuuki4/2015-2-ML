import cPickle
import pickle
import tarfile
import gzip
import numpy as np
import theano
from skimage import data

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

		# prepare for ZCA
		flat = gcn_array.reshape(1024, 1)
		corr_array[j,:,:] += np.dot(flat, flat.T)

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
corr_array /= (train_num+val_num)
for j in range(3) :
	print "SVD of filter %d.." % (j+1)
	U, s, V = np.linalg.svd(corr_array[j,:,:])
	ZCA_array = np.dot(np.dot(U, np.diagflat(np.diag(1.0/np.sqrt(np.diag(s)+epsilon)))), U.T)
	"""
	for i in range(train_num) :
		flat = train_data[i,j,:,:].reshape(1024, 1)
		train_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)
	for i in range(val_num) :
		flat = val_data[i,j,:,:].reshape(1024, 1)
		val_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)
	for i in range(test_num) :
		flat = test_data[i,j,:,:].reshape(1024, 1)
		test_data[i,j,:,:] = np.dot(ZCA_array, flat).reshape(32, 32)
	"""

"""
f = open('preprocessed.txt', 'wb')
pickle.dump([train_data, train_label, val_data, val_label, test_data, test_label], f)
f.close()
"""



f = open('ZCAarray.txt', 'wb')
pickle.dump(ZCA_array, f)
f.close()