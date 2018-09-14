#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 00:28:05 2018

@author: renyong

Matplotlib plot graphs 
"""

import matplotlib.pyplot as plt

def plot_graphs(training_costs, test_costs, training_accuracies, test_accuracies): 
    '''
    training_costs: list
    '''
    plt.plot(range(len(training_costs)), training_costs, range(len(test_costs)), test_costs)
    plt.show()
    plt.plot(range(len(training_accuracies)), training_accuracies, 
             range(len(test_accuracies)), test_accuracies)
    plt.show()
    