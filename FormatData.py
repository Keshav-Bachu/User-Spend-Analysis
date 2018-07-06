# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:56:44 2018

@author: Keshav Bachu
"""
#This project's goal is to predict if a user will spend on a certian object by observing parameters of the user
#within the app
#The project will be using tensorflow and numpy to construct a sigmoid based neural-network but using structured
#data that was collected previously
#This marks version 1 of this project!

import numpy as np
import PredictUserSpending as PSpending
#import tensorflow as tf

#STEP 1: Data formatting
#The data is stored as a CSV format and specific columns must be extracted for use
yData = np.loadtxt(fname = 'UserData_Formatted.csv', delimiter = ',', usecols = [1], skiprows = 1)
yData = yData.reshape(yData.shape[0], 1)    #reshape just in case to avoid rank (1) matrix
yData = yData.T
xData = np.loadtxt(fname = 'UserData_Formatted.csv', delimiter = ',', usecols = [2,3,4,5,6,7,8,9,10,11], skiprows = 1)
xData = xData.T

#Split into X_train and X_test (No X_dev due to lack of data sufficient :( )
#Using a 90 - 10 split
boundary = int(xData.shape[1] * 0.9)
xTrain = xData[:, 0:boundary]
yTrain = yData[:, 0:boundary]

xDev = xData[:, boundary:]
yDev = yData[:, boundary:]

#Tensorflow function part
#PredictUserSpending will have the full function that takes the X and Y array input numpy arrays
#output params has the trained weights that will be saved, params can be used later to predict different sets
params = PSpending.trainModel(xTrain, yTrain, xDev, yDev, [16,8,4,4,4,1], itterations=2100)


"""
Reference to storing the numpy weights
In [819]: N
Out[819]: 
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])

In [820]: data={'N':N}

In [821]: np.save('temp.npy',data)

In [822]: data2=np.load('temp.npy')
          then use np.ndarray.tolist to get back dictionary of weights to input in again!
          useful for retraining weights, for a controlled model

In [823]: data2
Out[823]: 
array({'N': array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])}, dtype=object)
    
In [826]: data2[()]['N']
Out[826]: 
array([[  0.,   1.,   2.,   3.],
       [  4.,   5.,   6.,   7.],
       [  8.,   9.,  10.,  11.]])
"""