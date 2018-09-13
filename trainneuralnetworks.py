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

def load_data(x_train_file, x_test_file, y_train_file, y_test_file):
    x_train = genfromtxt(x_train_file, delimiter=',').T
    x_test = genfromtxt(x_test_file, delimiter=',').T
    y_train = genfromtxt(y_train_file, delimiter=',').T
    y_test = genfromtxt(y_test_file, delimiter=',').T    
    return (x_train, y_train, x_test, y_test)

def upload_network(network, w_file, b_file): 
    w_df = pd.read_csv(w_file, header=None)
    b_df = pd.read_csv(b_file, header=None)
    populate_network(network, w_df, b_df)

def compare_dw(network, w_file, b_file):
    w_df = pd.read_csv(w_file, header=None)
    b_df = pd.read_csv(b_file, header=None)
    diff_dw(network, w_df, b_df)
    


if __name__ == '__main__':
    
    # preprocessing, assume data is shuffled
    x_train, y_train, x_test, y_test = load_data('Assignment_1/Question_2_1/x_train.csv', 
                                                 'Assignment_1/Question_2_1/x_test.csv',
                                                 'Assignment_1/Question_2_1/y_train.csv',
                                                 'Assignment_1/Question_2_1/y_test.csv') 
    
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
    
    (network_1, loss) = network_forward(network_1, X, label_data=label)
    network_backward(network_1) # backpropagation
    compare_dw(network_1, 'Assignment_1/Question_2_2/b/true-dw-100-40-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-100-40-4.csv')
    
    
    (network_2, loss) = network_forward(network_2, X, label_data=label)
    network_backward(network_2) # backpropagation
    compare_dw(network_2, 'Assignment_1/Question_2_2/b/true-dw-28-6-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-28-6-4.csv')
    
    (network_3, loss) = network_forward(network_3, X, label_data=label)
    network_backward(network_3) # backpropagation
    compare_dw(network_3, 'Assignment_1/Question_2_2/b/true-dw-14-28-4.csv', 
               'Assignment_1/Question_2_2/b/true-db-14-28-4.csv')    
    
    
    # reset everything 
    reset_layers(network_1)
    reset_layers(network_2)
    reset_layers(network_3)

        
# =============================================================================
#     networks = {'network_1':network_1, 'network_2': network_2}
#     
#     for n_name in network_names: 
#         for b_size in [1, 30]: 
#             network = copy.deepcopy(networks[n_name])
# =============================================================================

