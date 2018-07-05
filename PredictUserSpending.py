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
def createPlaceholders(X, Y):
    Xshape = X.shape[0]
    Xplace = tf.placeholder(tf.float32, shape = (Xshape, None))
    Yplace = tf.placeholder(tf.float32, shape = (1, None))  
    
    return Xplace, Yplace
#Used to create the placeholders of float 32 with a given shape
def createVariables(networkShape):
    placeholders = {}
    
    for i in range(1, len(networkShape)):
        placeholders['W' + str(i)] = tf.get_variable(name = 'W' + str(i), shape=[networkShape[i], networkShape[i - 1]], initializer=tf.contrib.layers.xavier_initializer())
        placeholders['b' + str(i)] = tf.get_variable(name = 'b' + str(i), shape=[networkShape[i], 1], initializer=tf.zeros_initializer())
    return placeholders


def forwardProp(X, placeholders):
    #total number of parameters in the network, divided by 2 for the number of layers within it with X being 0
    totalLength = len(placeholders)/2
    totalLength = int(totalLength)
    
    val1 = X
    val2W = placeholders['W' + str(1)]
    val2b = placeholders['b' + str(1)]
    
    pass_Z = tf.matmul(val2W, val1) + val2b
    pass_A = tf.nn.relu(pass_Z)
    
    #need to calculate the W and A porions of the code now
    for i in range (1, totalLength):
        #First element needs the X version hence it has a seperate condition
        val_W = placeholders['W' + str(i + 1)]
        val_b = placeholders['b' + str(i + 1)]
        
        pass_Z = tf.matmul(val_W, pass_A) + val_b
        pass_A = tf.nn.relu(pass_Z)
            
    return pass_Z

def computeCost(finalZ, Y):
    logits = tf.transpose(finalZ)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost

def trainModel(xTest, yTest, networkShape,  learning_rate = 0.0001, itterations = 1500, print_Cost = True):
 
    ops.reset_default_graph()
    costs = []                      #used to graph the costs at the end for a visual overview/analysis
 
    #Need to first create the tensorflow placeholders
    Xlen = xTest.shape[0]   #get the number of rows, aka features for the input data
    networkShape.insert(0, Xlen)
    
    X, Y = createPlaceholders(xTest, yTest)
    placeholders = createVariables(networkShape)
    
    Zfinal = forwardProp(X, placeholders)
    cost = computeCost(Zfinal, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        temp_cost = 0   
        for itter in range(itterations):
            _,temp_cost = sess.run([optimizer, cost], feed_dict={X:xTest, Y: yTest})
            
            if(itter % 100 == 0):
                print("Current cost of the function after itteraton " + str(itter) + " is: \t" + str(temp_cost))
                
            costs.append(temp_cost)
            
            
        parameters = sess.run(placeholders)
        prediction = tf.equal(tf.argmax(Zfinal), tf.argmax(yTest))
        accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    
        print ("Train Accuracy:", accuracy.eval({X: xTest, Y: yTest}))
    return parameters