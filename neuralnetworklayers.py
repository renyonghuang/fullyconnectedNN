#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 21:54:19 2018

@author: renyong
"""
import numpy as np
import math 

def one_hot_encode(y, num_class):
    m = y.shape[0]
    onehot = np.zeros((m, num_class), dtype="int32")
    for i in range(m):
        idx = y[i]
        onehot[i][idx] = 1
    return onehot

class FullyConnectedLayer:
    def __init__(self, num_input, num_output, lr=1e-3, scale=2):
        # layer parameters
        # Xavier/He initialization 
        self.W = np.random.randn(num_output, num_input) * np.sqrt(scale/(num_input+num_output))
        self.b = np.random.randn(num_output, 1) * np.sqrt(scale/(num_input+num_output))
        self.scale = scale
        
        # gradients and momentum 
        self.gW = np.zeros(self.W.shape).astype(np.float128)
        self.gb = np.zeros(self.b.shape).astype(np.float128)
        
        # W.T*delta
        self.gI = np.array([]).astype(np.float128)

        # momentum
        self.vW = np.zeros(self.W.shape).astype(np.float128)
        self.vb = np.zeros(self.b.shape).astype(np.float128)
        
        # layer input and output
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        
        # learning rate 
        self.lr = lr 
        
    def reset_W_b(self): 
        num_output, num_input = self.W.shape
        self.W = np.random.randn(num_output, num_input) * np.sqrt(self.scale/(num_input+num_output))
        self.b = np.random.randn(num_output, 1) * np.sqrt(self.scale/(num_input+num_output))
        
    def sanity_check(self, W_in, b_in): 
        self.W = W_in
        self.b = b_in
    
    def forward(self, input_data): 
        self.input_data = input_data
        #print('w shape', self.W.shape)
        #print('input data shape', input_data.shape)
        self.output_data = np.dot(self.W, input_data) + self.b
        return self.output_data
    
    def backward(self, gradient_data): 
        # gradient_data = delta_l
        self.gW = np.dot(gradient_data, np.transpose(self.input_data))
        self.gb = np.sum(gradient_data, axis=1, keepdims=True)
        self.gI = np.dot(np.transpose(self.W), gradient_data)
        return self.gI

def sigmoid(x): 
    return 1/(1+np.exp(-x))

class SigmoidLayer:
    def __init__(self): 
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        
    def forward(self, x): 
        self.input_data = x
        self.output_data = sigmoid(x)
        return self.output_data
    
    def backward(self, gradient): 
        self.gI = (gradient * 
                   sigmoid(self.input_data) *
                   (1-sigmoid(self.input_data))).astype(np.float128)
        return self.gI
    
class ReLuLayer: 
    def __init__(self): 
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
    def forward(self, x): 
        self.input_data = x
        self.output_data  = x.clip(0)
        return self.output_data
    def backward(self, gradient): 
        self.gI = gradient * (self.input_data > 0).astype(np.float128)
        return self.gI

class SoftmaxOutput_CrossEntropyLossLayer: 
    def __init__(self):
        self.gI = np.array([]).astype(np.float128)
        self.input_data = np.array([]).astype(np.float128)
        self.output_data = np.array([]).astype(np.float128)
        self.pred = np.array([]).astype(np.float128)
        self.loss = math.inf
        self.y = np.array([]).astype(np.float128)

    def cross_entropy_loss(self, p,y):
        """
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector. 
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = len(y)
        # We use multidimensional array indexing to extract 
        # softmax probability of the correct label for each sample.
        # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
        log_likelihood = -np.log(p[[int(elem) for elem in y], range(m)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def delta_cross_entropy(self, p,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        	Note that y is not one-hot encoded vector. 
        	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        m = len(y)
        grad = p.copy()
        grad[[int(elem) for elem in y], range(m)] -= 1
        grad = grad/m
        return grad
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def evaluate(self, X, label_data, phase):
        """
        X is the output from fully connected layer (num_classes x num_examples)
        """
        #print(X)
        self.input_data = X
        self.output_data = self.softmax(X) # output data: p
        self.y = label_data
        self.pred = self.output_data.argmax(axis = 0)
        self.loss = self.cross_entropy_loss(self.output_data, label_data)
        self.accuracy = sum(self.pred == self.y) *1.0/len(self.y)
        return self.loss, self.accuracy
    
    def backward(self): 
        self.gI = self.delta_cross_entropy(self.output_data, self.y)
        return self.gI

