import gzip
import cPickle
import os.path
import math
import random
from numpy import *
import numpy as np
from time import time

# change data into numpy array vector. vector is (nx1) form
def data2vector(data) :
	return array(data).T

def result2vector(dataY) : 
	returnVec = zeros(10)
	returnVec[dataY] = 1
	return returnVec

def RBF(x, mean, covpinv) :
	covpinv = mat(covpinv)
	x = mat(x).T
	mean = mat(mean).T
	val = (x-mean).T*covpinv*(x-mean)
	val/=(-2)
	return math.exp(val)

def RBF2(x, mean, var) :
	x = mat(x).T
	mean = mat(mean).T
	val = (x-mean).T*(x-mean)
	val/=(var*(-2))
	return math.exp(val)

def RBF3(xArray, meanArray, covVec) :
	distArray = xArray - meanArray
	valVec = np.power(np.linalg.norm(distArray, axis=1), 2)
	valVec = np.exp(divide(valVec, covVec*(-2)))
	return append(valVec, 1.0)

def logGaussian(x, mean, cov, dim) :
	# when cov is single-valued covariance matrix (var cov = double)
	x = mat(x).T
	mean = mat(mean).T
	val = (x-mean).T*(x-mean)
	val /= (cov*(-2))
	return val-dim/2*(math.log(2*math.pi)+math.log(cov))

def sig(x) :
	return 1.0/(1.0+math.exp(-x))

def dsig(x) : 
	val = sig(x)
	return val*(1-val)

def deslant(image) :
	imgArray = np.reshape(image, (28, 28))
	xavg = 0.0
	yavg = 0.0
	sumpixel = 0.0
	m = 0.0
	msum = 0.0
	for i in range(28) :
		for j in range(28) :
			xavg += imgArray[i, j] * i
			yavg += imgArray[i, j] * j
			sumpixel += imgArray[i,j]
	xavg /= sumpixel
	yavg /= sumpixel
	for i in range(28) :
		for j in range(28) :
			m += imgArray[i, j] * i * j
			msum += imgArray[i, j] * i * i
	m -= xavg*yavg*sumpixel
	msum -= xavg*xavg*sumpixel
	m /= msum

	returnImg = zeros((28, 28))
	for i in range(28) :
		for j in range(28) :
			xprime = j + m * (i-xavg)
			val1=0.0; val2=0.0; x1=math.floor(xprime); x2=math.ceil(xprime);
			if x1<0 : x1=30
			if x2<0 : x2=30
			try : val1 = imgArray[i, x1]
			except : val1 = 0.0
			try : val2 = imgArray[i, x2]
			except : val2 = 0.0
			
			returnImg[i,j] = (x2-xprime)*val1 + (xprime-x1)*val2

	return np.reshape(returnImg, 784)

# for time limit
start_time = time()

# get data from mnist.pkl.gz
codePath = os.path.dirname(__file__)
dataPath = os.path.abspath(os.path.join(codePath, os.pardir))
f = gzip.open(os.path.join(dataPath, 'data\mnist.pkl.gz'), 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_x, train_y = train_set
test_x, test_y = test_set


# Set train / test n umber 
trainNumber = 10000
testNumber = 2000
dataSize = len(train_x[0])

# deslanting images
for i in range(trainNumber) :
	train_x[i] = deslant(train_x[i])
for i in range(testNumber) :
	test_x[i] = deslant(test_x[i])

### PCA Session : Dimension Reduction! ###

newDimension = 40

meanVector = zeros(dataSize)
for i in range(trainNumber) : 
	train_x[i] = data2vector(train_x[i])
	meanVector += data2vector(train_x[i])
meanVector /= trainNumber
for i in range(trainNumber) : train_x[i] -= meanVector
variance = 0.0
for i in range(trainNumber) : variance += np.linalg.norm(train_x[i])
variance /= trainNumber
for i in range(trainNumber) : train_x[i] /= math.sqrt(variance)

covMatrix = zeros((dataSize, dataSize))
for i in range (trainNumber) : covMatrix += mat(train_x[i]).T * mat(train_x[i])
covMatrix /= trainNumber

eigenVal, eigenVec = np.linalg.eig(covMatrix)
idx = argsort(eigenVal)
newtrain = []
newtest = []

for i in range(trainNumber) :
	newList = []
	for j in range(newDimension) :
		newList.append(float((mat(eigenVec[:,j])*mat(train_x[i]).T)[0,0]))
	newtrain.append(newList)

for i in range(testNumber) : 
	test_x[i] = (data2vector(test_x[i])-meanVector)/math.sqrt(variance)
	newList = []
	for j in range(newDimension) : 
		newList.append(float((mat(eigenVec[:,j])*mat(test_x[i]).T)[0,0]))
	newtest.append(newList)

dataSize = newDimension
train_x = newtrain
test_x = newtest

### preprocessing session : make K gaussian kernels ###
"""
# preprocessing 1 - intuition : make 10 kernels by seperating by digits (0~9)

kernelNum = 10
trainNum = zeros((kernelNum, 1))
meanVecArray = zeros((kernelNum, dataSize))
covArray = zeros((kernelNum, dataSize, dataSize))
covVec = zeros(kernelNum)

for i in range(trainNumber) :
	trainNum[train_y[i], 0]+=1
	trainVec = data2vector(train_x[i])
	meanVecArray[train_y[i],:] += trainVec
for i in range(kernelNum) :
	meanVecArray[i,:] /= trainNum[i, 0]
for i in range(trainNumber) :
	trainVec = data2vector(train_x[i])
	# for nonsingular covariance matrix, constraint covariance as a single-valued diagonal matrix
	for j in range(dataSize) :
		covVec[train_y[i]] += (trainVec[j]-meanVecArray[train_y[i], j])*(trainVec[j]-meanVecArray[train_y[i], j])
for i in range(kernelNum) :
	covVec[i] /= (trainNum[i,0])
	for j in range(dataSize) :
		covArray[i, j, j] = covVec[i]

"""

# preprocessing 2 : generate K kernels by k-means

kernelNum = 4000
trainNum = zeros((kernelNum, 1))
meanVecArray = zeros((kernelNum, dataSize))
covVec = zeros(kernelNum)

# k-means session

# initialize - Forgy selection
centerList = []
indexList = []
for i in range(trainNumber) : indexList.append(i)
isSelected = zeros(trainNumber)

chosen = 0
while chosen < kernelNum : 
	index = int(math.floor(random.random() * len(indexList)))
	chosen+=1
	centerList.append(data2vector(train_x[indexList[index]]))
	del indexList[index]

iterationMax = 20
kernelAssign = zeros(trainNumber)
kernelCount = zeros(kernelNum)

for lev in range(iterationMax) :

	# cluster assign
	for i in range(trainNumber) : 
		distarray = tile(data2vector(train_x[i]), (kernelNum, 1))-array(centerList)
		kernelAssign[i] = np.argmin(np.linalg.norm(distarray, axis=1))

	# move cluster center
	for i in range(kernelNum) : 
		centerList[i] = zeros(dataSize)
		kernelCount[i] = 0
	for i in range(trainNumber) :
		kernelCount[int(kernelAssign[i])]+=1
		centerList[int(kernelAssign[i])]+=data2vector(train_x[i])
	for i in range(kernelNum) : centerList[i] /= kernelCount[i]

	if time()-start_time > 420.0 : break

for i in range(kernelNum) :
	trainNum[i, 0] = kernelCount[i]
	meanVecArray[i,:] = centerList[i]
for i in range(trainNumber) :
	trainVec = data2vector(train_x[i])
	# for nonsingular covariance matrix, constraint covariance as a single-valued diagonal matrix
	for j in range(dataSize) :
		covVec[kernelAssign[i]] += (trainVec[j]-meanVecArray[kernelAssign[i], j])*(trainVec[j]-meanVecArray[kernelAssign[i], j])
for i in range(kernelNum) :
	covVec[i] /= (trainNum[i,0])

# remove 'unselected' clusters 
deleteList = []
for i in range(kernelNum) :
	if (kernelCount[i]==0 or covVec[i]==0) :
		deleteList.append(i)
			
kernelNum -= len(deleteList)	
covVec = np.delete(covVec, deleteList)
meanVecArray = np.delete(meanVecArray, deleteList, 0)

"""
# Preprocessing 3 : Randomly Generating K kernels 

kernelNum = 1000
meanVecArray = zeros((kernelNum, dataSize))
covArray = zeros((kernelNum, dataSize, dataSize))
covVec = zeros(kernelNum)

selectedKernelList = []
isSelected = zeros(trainNumber)

chosen = 0
while chosen < kernelNum : 
	index = int(math.floor(random.random() * trainNumber))
	if isSelected[index]<0.5 :
		chosen+=1
		isSelected[index] = 1.0
		selectedKernelList.append(data2vector(train_x[index]))

for i in range(kernelNum) :
	meanVecArray[i, :] = selectedKernelList[i]
	covVec[i] = 99999999
	for j in range(kernelNum) :
		if i==j : continue
		dist = np.linalg.norm(selectedKernelList[i]-selectedKernelList[j])
		if dist < covVec[i] :
			covVec[i] = dist

for i in range(kernelNum) :
	for j in range(dataSize) :
		covArray[i, j, j] = covVec[i]

print "Preprocessing Done."
"""


### training session : change W by linear regression
# use Normal Equation

# global scaling factor : make covArray bigger & covPinvArray smaller 
covVec *= 5

H = mat(zeros((trainNumber, kernelNum+1)))
H2 = mat(zeros((trainNumber, kernelNum+1)))
Y = zeros((trainNumber, 10))
for i in range(trainNumber) :
	H[i, :] = RBF3(tile(data2vector(train_x[i]), (kernelNum, 1)), meanVecArray, covVec)

for i in range(trainNumber) :
	Y[i, :] = result2vector(train_y[i])
Y = mat(Y)

# W = (kernelNum+1 * 10), W[i, j] = weight from kernel i to result node j
W = (H.T*H).I*H.T*Y

"""
### training session 2 : used 3-layer NN (kernel -> hidden -> result)
# error : MSE, node : sigmoid unit

# global scaling factor : make covArray bigger & covPinvArray smaller 
# covVec *= 30

H = mat(zeros((trainNumber, kernelNum+1)))
Y = zeros((trainNumber, 10))
for i in range(trainNumber) :
	for j in range(kernelNum+1) :
		if(j<kernelNum) : H[i, j] = RBF2(data2vector(train_x[i]), meanVecArray[j,:], covVec[j])
		else : H[i, j] = 1 # for bias term
	print i

# RBF normalization : 0 to 1
maxArr = amax(H, axis=0)
minArr = amin(H, axis=0)
diffArr = maxArr-minArr
for i in range(trainNumber) :
	H[i,:] = (H[i,:].getA1()-minArr)/diffArr
	H[i,kernelNum] = 1

print H

hLayerNum = 40
W1 = mat(zeros((kernelNum+1, hLayerNum)))
W2 = mat(zeros((hLayerNum+1, 10)))
stepSize = 1

# weight initialize : range of (-1/sqrt(d), 1/sqrt(d))
for i in range(kernelNum+1) :
	for j in range(hLayerNum) : 
		W1[i, j] = (random.random()*2-1) / math.sqrt(kernelNum+1) # W1[i, j] = input kernel i to hidden kernel j
for i in range(hLayerNum+1) :
	for j in range(10) :
		W2[i, j] = (random.random()*2-1) / math.sqrt(hLayerNum+1)

# weight training : online learning
maxEpoch = 100
for lev in range(maxEpoch) :

	for i in range(trainNumber) : 
		f_h = zeros(hLayerNum+1)
		f_h[hLayerNum] = 1
		f_k = zeros(10)

		# feed-fwd
		for j in range(hLayerNum) :
			f_h[j] = sig((mat(W1[:, j]).T*mat(H[i, :]).T)[0, 0])
		for j in range(10) :
			f_k[j] = sig((mat(W2[:, j]).T*mat(f_h).T)[0,0])

		print f_h
		print f_k

		delta_k = zeros(10)

		# back-prop
		errsum = 0.0

		for j in range(10) :
			if j==train_y[i] :
				delta_k[j] = f_k[j]*(1-f_k[j])*(f_k[j]-1.0)
			else :
				delta_k[j] = f_k[j]*(1-f_k[j])*f_k[j]

		for j in range(hLayerNum+1) :
			delta_h = 0.0
			for k in range(10) :
				delta_h += delta_k[k] * W2[j, k] * f_h[j] * (1-f_h[j])
			if j == hLayerNum : break
			for k in range(kernelNum+1) :
				W1[k, j] -= stepSize * H[i, k] * delta_h

		for j in range(hLayerNum+1) :
			for k in range(10) :
				W2[j,k] -= stepSize * delta_k[k] * f_h[j]

		print delta_k


		print lev * trainNumber + i + 1
		if time()-start_time > 540 : # 9min.
			break

	print W2
	if time()-start_time > 540 :
		break
"""
### Test Session ###

f = open("result.txt", 'w')
errorCount = 0.0
for i in range(testNumber) :
	testKernel = zeros(kernelNum+1)
	bestValue = -99999999.0;
	RBFResult = -1;
	testKernel = RBF3(tile(data2vector(test_x[i]), (kernelNum, 1)), meanVecArray, covVec)
	
	# Using Linear Classifier
	for j in range(10) :
		nowValue = dot(testKernel, W[:,j].getA1())
		if (nowValue > bestValue) :
			bestValue = nowValue
			RBFResult = j
	
	"""
	# Using 3-Layer NN
	f_h = zeros(hLayerNum+1)
	f_h[hLayerNum] = 1
	f_k = zeros(10)

	for j in range(hLayerNum) :
		f_h[j] = sig((mat(W1[:, j]).T*mat(testKernel).T)[0, 0])
	print f_h
	for j in range(10) :
		f_k[j] = sig((mat(W2[:, j]).T*mat(f_h).T)[0,0])
		if (bestValue < f_k[j]) :
			bestValue = f_k[j]
			RBFResult = j
	print f_k
	"""

	toWrite = "%d " % RBFResult
	f.write(toWrite)
	"""
	print "Test # %d !! RBFResult : %d, Real Answer : %d" % (i+1, RBFResult, test_y[i])
	if RBFResult!=test_y[i] : errorCount += 1.0
	"""	

"""
print "Total error count : %d" % (int(errorCount))
print "Total error rate : %f" % (errorCount / float(testNumber))
"""
f.close()


