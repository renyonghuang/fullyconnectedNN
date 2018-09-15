#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 00:28:05 2018

@author: renyong

Matplotlib plot graphs 
"""

import matplotlib.pyplot as plt

def plot_graphs(tr_c_n1, tst_c_n1, tr_a_n1, tst_a_n1, tr_c_n2, tst_c_n2, tr_a_n2, tst_a_n2,
                tr_c_n3, tst_c_n3, tr_a_n3, tst_a_n3): 
    '''
    training_costs: list
    '''
    # cost graph
    pl_1 = plt.plot(range(len(tr_c_n1)), tr_c_n1, label='network_1_100_40_4')
    pl_2 = plt.plot(range(len(tr_c_n2)), tr_c_n2, label='network_2_28-6_4') 
    pl_3 = plt.plot(range(len(tr_c_n3)), tr_c_n3, label='network_3_14-28_4')
    plt.ylim((0.0, 3.0))
    plt.legend(loc='upper right')

    plt.xlabel('number of iterations')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Cross Entropy Loss for 10 epochs of minibatch size 10')
    plt.grid(True)
    plt.show()
    
    # accuracy graph
    pl_4 = plt.plot(range(len(tr_a_n1)), tr_a_n1, label='network_1_100_40_4')
    pl_5 = plt.plot(range(len(tr_a_n2)), tr_a_n2, label='network_2_28-6_4')
    pl_6 = plt.plot(range(len(tr_a_n3)), tr_a_n3, label='network_3_14-28_4')
    plt.legend(loc='upper right')
    plt.xlabel('number of iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy for 10 epochs of minibatch size 10')
    plt.grid(True)
    plt.show()
    
    