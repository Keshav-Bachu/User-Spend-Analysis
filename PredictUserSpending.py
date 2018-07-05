# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:48:40 2018

@author: Keshav Bachu
"""
#Using the tensorflow archetecture for development
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import numpy as np

#Used to create the placeholders of float 32 with a given shape
def createPlaceholders(networkShape):
    placeholders = {}
    
    Xshape = networkShape[0]
    placeholders['X'] = tf.placeholder(tf.float32, shape = (Xshape, None))
    placeholders['Y'] = tf.placeholder(tf.float32, shape = (Xshape, None))
    
    for i in range(1, len(networkShape)):
        placeholders['W' + str(i)] = tf.get_variable(name = 'W' + str(i), shape=[networkShape[i], networkShape[i - 1]], initializer=tf.contrib.layers.xavier_initializer())
        placeholders['b' + str(i)] = tf.get_variable(name = 'b' + str(i), shape=[networkShape[i], 1], initializer=tf.zeros_initializer())
    return placeholders


def forwardProp(placeholders):
    #total number of parameters in the network, divided by 2 for the number of layers within it with X being 0
    totalLength = len(placeholders)/2
    
    pass_Z, pass_A = None
    #need to calculate the W and A porions of the code now
    for i in range (0, totalLength - 1):
        
        #First element needs the X version hence it has a seperate condition
        if(i == 0):
            val1 = placeholders['X']
            val2W = placeholders['W' + str(1)]
            val2b = placeholders['b' + str(1)]
            
            pass_Z = tf.matmul(val1, val2W) + b
            pass_A = tf.nn.relu(pass_Z)
        else:
            val_W = placeholders['W' + str(i + 1)]
            val_b = placeholders['b' + str(i + 1)]
            
            pass_Z = tf.matmul(pass_A, val_W) + val_b
            pass_A = tf.nn.relu(pass_Z)
            
    return pass_Z

def computeCost():
    return None

def trainModel(xTest, yTest, xDev, yDev, learning_rate = 0.0001, num_epochs = 1500, minibatch = None, print_Cost = True):
    
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = xTest.shape
    n_y = yTest.shape[0]
    
    costs = []                      #used to graph the costs at the end for a visual overview/analysis
    
    #Need to first create the tensorflow placeholders
    
    return None