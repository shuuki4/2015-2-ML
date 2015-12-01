import theano
import theano.tensor as T
import theano.tensor.nnet.conv as conv
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm

# generate random seed
rng = np.random.RandomState(624072)

# input tensor shape
input = T.tensor4(name='input') # (mini_batch size, channel num, image height, image width)

# initialize weight
w_shape = (2,3,9,9) # (next layer channel num, this layer channel num, image height, image width)
w_bound = np.sqrt(3*9*9) # for good initialize
W = theano.shared(np.asarray(rng.uniform(low=-1.0/w_bound, high=1.0/w_bound, size=w_shape), dtype=theano.config.floatX), name='W')

# initialize bias
b_shp = (2,)
b = theano.shared(np.asarray(rng.uniform(low=-0.5, high=0.5, size=b_shp), dtype=theano.config.floatX), name='b')

# symbolic expression for output
conv_out = conv.conv2d(input, W)

# output symbol
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))

# function
f = theano.function([input], output)

# img read
img = mpimg.imread('apple_raw.png')
img = np.asarray(img, dtype=theano.config.floatX)[:,:,0:3]
img = img.transpose(2,0,1).reshape(1,3,314,305)
filtered_img = f(img)


##show filtered img

plt.imshow(filtered_img[0,0,:,:], cmap=cm.Greys_r)
plt.show()
plt.imshow(filtered_img[0,1,:,:], cmap=cm.Greys_r)
plt.show()