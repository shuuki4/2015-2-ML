# 2015-2-ML
Projects @ 2015-2 Machine Learning Course

## Project 1 : RBFN based MNIST Classifier
- 3-layer RBFN
- Preprocess (PCA, deslant) -> K-means -> RBFN
- ~ 97% Accuracy Rate

## Project 2 : CIFAR-10 Classifier
- Deep Conv Layer
- Theano based
- Preprocess : GCN+ZCA,    Data Augmentation : Rotate, Horizontal Flip, Gamma correction
- 6 Convolution Layer with maxout activation, 2 Pool Layer, 2 Dropout Layer, Fully Connected MLP with dropconnect
- ~ 88% Accuracy Rate after 24h (GTX 840m, 100 epochs)
