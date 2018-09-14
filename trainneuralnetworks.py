#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 21:10:43 2018

@author: renyong
"""
import pandas as pd
import numpy as np
from neuralnetworklayers import *
from neuralnetworks import *
from numpy import genfromtxt
from sklearn.utils import shuffle
import math
import logging
import random
import plotgraphs

def load_data(x_train_file, x_test_file, y_train_file, y_test_file):
    '''
    Data in this format: 
    x_train = [[row1], [row2], [row3], [row4], [row5]]
    '''
    x_train = genfromtxt(x_train_file, delimiter=',')
    x_test = genfromtxt(x_test_file, delimiter=',')
    y_train = genfromtxt(y_train_file, delimiter=',')
    y_test = genfromtxt(y_test_file, delimiter=',') 
    return (x_train, y_train, x_test, y_test)

def upload_network(network, w_file, b_file): 
    '''upload with given weights and bias'''
    w_df = pd.read_csv(w_file, header=None)
    b_df = pd.read_csv(b_file, header=None)
    populate_network(network, w_df, b_df)

def compare_dw(network, w_file, b_file):
    w_df = pd.read_csv(w_file, header=None)
    b_df = pd.read_csv(b_file, header=None)
    return diff_dw(network, w_df, b_df)
    
def partition_train(network, x_train, y_train, x_test, y_test, logger, batch_size=1): 
    print('Start training: ')
    n_batch = math.ceil(x_train.shape[0]*1.0 / batch_size)
    training_costs = []
    test_costs = []
    training_accuracies = []
    test_accuracies = []
    for i in range(n_batch): 
        if i != (n_batch - 1): 
            x_batch_train = x_train[i*batch_size:(i+1)*batch_size, :].T
            y_batch_train = y_train[i*batch_size:(i+1)*batch_size].T
        else:
            x_batch_train = x_train[i*batch_size:, :].T
            y_batch_train = y_train[i*batch_size:].T
            
        idxs_sampled = random.sample(range(len(x_test)), batch_size)
        x_batch_test = np.array([x_test[i] for i in idxs_sampled]).T
        y_batch_test = np.array([y_test[i] for i in idxs_sampled]).T

        (test_loss, test_accuracy) = network_forward(network, x_batch_test, label_data=y_batch_test)       
        (train_loss, train_accuracy) = network_forward(network, x_batch_train, label_data=y_batch_train)
        training_costs.append(train_loss)
        test_costs.append(test_loss)
        training_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
        network_backward(network) 
        # gradient_descent(network, decay = 0.9999)
        network_momentum_SGD(network, decay=0.9999)
    
    plotgraphs.plot_graphs(training_costs, test_costs, training_accuracies, test_accuracies)


if __name__ == '__main__':
    logger = logging.getLogger()
    
    # preprocessing
    x_train, y_train, x_test, y_test = load_data('Assignment_1/Question_2_1/x_train.csv', 
                                                 'Assignment_1/Question_2_1/x_test.csv',
                                                 'Assignment_1/Question_2_1/y_train.csv',
                                                 'Assignment_1/Question_2_1/y_test.csv') 
    # shuffle rows of data in consistent way
    x_train, y_train = shuffle(x_train, y_train, random_state=0)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)
    
    # prepare network
    input_num = 14
    output_num = 4
    lr = 0.4
    
    network_1 = [FullyConnectedLayer(input_num, 100, lr=lr), ReLuLayer()]
    network_1.append(FullyConnectedLayer(100, 40, lr=lr))
    network_1.append(ReLuLayer())
    network_1.append(FullyConnectedLayer(40, output_num, lr=lr))

    network_1.append(SoftmaxOutput_CrossEntropyLossLayer())
    
    network_2 = [FullyConnectedLayer(input_num, 28, lr=lr), ReLuLayer()]
    for p in range(5): 
        network_2.append(FullyConnectedLayer(28, 28, lr=lr))
        network_2.append(ReLuLayer())
    network_2.append(FullyConnectedLayer(28, output_num, lr=lr))
    network_2.append(SoftmaxOutput_CrossEntropyLossLayer())
    
    network_3 = [FullyConnectedLayer(input_num, 14, lr=lr, scale=4), ReLuLayer()]
    for p in range(27): 
        network_3.append(FullyConnectedLayer(14, 14, lr=lr, scale=4))
        network_3.append(ReLuLayer())
    network_3.append(FullyConnectedLayer(14, output_num, lr=lr, scale=4))
    network_3.append(SoftmaxOutput_CrossEntropyLossLayer())
    
    # sanity check
    # input 
    X = np.array([[-1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 1, 1]]).T
    label = [3]
    
    upload_network(network_1, 'Assignment_1/Question_2_2/b/w-100-40-4.csv', 
                   'Assignment_1/Question_2_2/b/b-100-40-4.csv')
    upload_network(network_2, 'Assignment_1/Question_2_2/b/w-28_6-4.csv', 
                   'Assignment_1/Question_2_2/b/b-28_6-4.csv')
    upload_network(network_3, 'Assignment_1/Question_2_2/b/w-14_28-4.csv', 
                   'Assignment_1/Question_2_2/b/b-14_28-4.csv')    
    
    (loss, accuracy) = network_forward(network_1, X, label_data=label)
    network_backward(network_1) # backpropagation
    compare_dw(network_1, 'Assignment_1/Question_2_2/b/true-dw-100-40-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-100-40-4.csv')
    
    
    (loss, accuracy) = network_forward(network_2, X, label_data=label)
    network_backward(network_2) # backpropagation
    compare_dw(network_2, 'Assignment_1/Question_2_2/b/true-dw-28-6-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-28-6-4.csv')
    
    (loss, accuracy) = network_forward(network_3, X, label_data=label)
    network_backward(network_3) # backpropagation
    cp_success = compare_dw(network_3, 'Assignment_1/Question_2_2/b/true-dw-14-28-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-14-28-4.csv')    
    if cp_success: 
        logger.error('Given answers are not the same as gradients calculated by network')
    
    # reset everything 
    reset_layers(network_1)
    reset_layers(network_2)
    reset_layers(network_3)
    
    # mini-batch 
    partition_train(network_1, x_train, y_train, x_test, y_test, logger, batch_size=100)
    
    # TODO: 
    # network momentum SGD 
    
    
    
    
# =============================================================================
#     networks = {'network_1':network_1, 'network_2': network_2}
#     
#     for n_name in network_names: 
#         for b_size in [1, 30]: 
#             network = copy.deepcopy(networks[n_name])
# =============================================================================

