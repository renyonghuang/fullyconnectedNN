#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 22:43:24 2018

@author: renyong
"""
from neuralnetworklayers import *
import numpy as np

def populate_network(network, w_df, b_df): 
    """
    w_df, b_df: (pandas dataframe)
    Cannot initialize by populating. Must set random weights/bias first 
    before populating
    """
    W_idx = 0
    i = 0 
    for layer in network: 
        if type(layer) is FullyConnectedLayer: 
            # w_df
            #print("populate_network ", layer.W.shape)
            (W_output_dim, W_input_dim) = layer.W.shape
            layer.W = w_df.iloc[W_idx:(W_idx+W_input_dim), 1:(W_output_dim+1)].as_matrix().T # exclude text
            #print('new_layer_w', layer.W)
            W_idx += W_input_dim
            
            # b_df
            b_dims = layer.b.shape[0]
            layer.b = np.array([b_df.iloc[i, 1:(b_dims+1)].values.tolist()]).T # TODO: rid of nan
            i += 1
    
def diff_dw(network, gW_df, gb_df):
    W_idx = 0
    i = 0 
    
    gW_same = True
    gb_same = True
    for layer in network: 
        if type(layer) is FullyConnectedLayer: 
            # gW_df
            #print("populate_network ", layer.W.shape)
            (W_output_dim, W_input_dim) = layer.W.shape
            gW_ans = gW_df.iloc[W_idx:(W_idx+W_input_dim), :W_output_dim].as_matrix().T # exclude text
            gW_compare = np.allclose(layer.gW, gW_ans)
            print('gW equal for layer', i, ': ', gW_compare)
            gW_same &= gW_compare
            W_idx += W_input_dim
            
            # gb_df
            b_dims = layer.b.shape[0]
            gb_ans = np.array([gb_df.iloc[i, :b_dims].values.tolist()]).T 
            gb_compare = np.allclose(layer.gb, gb_ans)
            print('gb equal for layer', i, ': ', gb_compare)
            gb_same &= gb_compare
            i += 1
    print('gWs and gbs in network are compared to correct answer.')
    if gW_same is True: 
        print('gWs in network correspond to gWs in correct answer')
    else: 
        print('gWs in network do not correspond to gWs in correct answer')
        return False
    
    if gb_same is True: 
        print('gbs in network correspond to gbs in correct answer')
    else: 
        print('gbs in network do not correspond to gbs in correct answer')
        return False
        
    return True
    

def reset_layers(network): 
    for layer in network: 
        if type(layer) is FullyConnectedLayer: 
            layer.reset_W_b()
    
    
def network_forward(network, input_data, label_data=None, phase='train'): 
    for layer in network: 
        if type(layer) is not SoftmaxOutput_CrossEntropyLossLayer:
            input_data = layer.forward(input_data)
        else:
            loss, accuracy = layer.evaluate(input_data, label_data, phase)
    return (loss, accuracy)

def network_backward(network): 
    for layer in reversed(network): 
        if type(layer) is SoftmaxOutput_CrossEntropyLossLayer:
            gradient = layer.backward()
        else:
            gradient = layer.backward(gradient)
    return network 
    
def gradient_descent(network, decay=1.0, n = 1): 
    '''Stochastic or mini-batch Gradient Descent'''
    for layer in reversed(network):
        if type(layer) is FullyConnectedLayer: 
            layer.lr *= decay
            layer.W -= layer.lr * layer.gW 
            layer.b -= layer.lr * layer.gb 
        else:
            continue
    return network

def network_momentum_SGD(network, decay=1.0, rho=0.9): 
    '''rho: momentum, decay: eta'''
    for layer in reversed(network): 
        if type(layer) is FullyConnectedLayer: 
            layer.lr *= decay
            layer.vW = layer.vW * rho + layer.lr * layer.gW 
            layer.W -= layer.vW
            layer.vb = layer.vb * rho + layer.lr * layer.gb 
            layer.b -= layer.vb
        else:
            continue
    return network
