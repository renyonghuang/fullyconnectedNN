# fullyconnectedNN
Fully Connected Neural Network using mainly numpy arrays, without keras/tf 

## Introduction
Fully Connected Neural Network implemented using numpy arrays. Principles of vectorization and array broadcasting were applied. 
This python script was written based on instructions on a school assignment, which gave the data and the networks to try out. 

networkneurallayers.py
==============
networkneurallayers contain the layers in a neural network, with each layer implemented as a class. The FullyConnectedLayer is the base for all layers, and which all types of layers are derived from. The activation functions are applied as a layer on top the FullyConnectedLayer.
Each layer has a forward propagation mode and backward propagation mode. It stores its activations so as to be able to perform its backpropagation calculations accurately without having to recalculate anything from the forward propagation. 

neuralnetworks.py
==============
neuralnetworks.py contains the functions to populate, reset, and do forward and backward propagation for a given network.

trainneuralnetworks
=================
trainneuralnetworks.py implements 3 different neural networks by defining the layers from networkneurallayers.py, and performs forward and backward propagation using functions from neuralnetworks.py.
With all the files and data, run trainneuralnetworks.py to get the output 
The script first does a sanity check with the dw provided by the asssignment, followed by training and testing of the 3 networks. Mini-batch size and epoches can be specified by partition_train and epoch functions respectively. 
Overall, the graph for accuracy and cross-entropy loss indicates that network 3 has problems increasing its accuracy beyond a certain point. 
