import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from Layer import *
from Network import *
from BatchNormMode import BatchNormMode
from LROptimizerType import LROptimizerType
from GradDescType import GradDescType
from sklearn.utils import shuffle
from ActivationType import ActivationType


train = np.empty((1000,28,28),dtype='float64')
trainY = np.zeros((1000,10))
test = np.empty((10000,28,28),dtype='float64')
testY = np.zeros((10000,10)) # Load in the images
i = 0
for filename in os.listdir('C:/Users/Windows User/Desktop/UB Spring 2019/Deep Learning/Assignment 3/Data/Training1000/'):
        y = int(filename[0])
        trainY[i,y] = 1.0
        train[i] = cv2.imread('C:/Users/Windows User/Desktop/UB Spring 2019/Deep Learning/Assignment 3/Data/Training1000/{0}'.format(filename),0)/255.0 # for color, use 1
        i = i + 1
i = 0 # read test data

for filename in os.listdir('C:/Users/Windows User/Desktop/UB Spring 2019/Deep Learning/Assignment 3/Data/Test10000'):
        y = int(filename[0])
        testY[i,y] = 1.0
        test[i] = cv2.imread('C:/Users/Windows User/Desktop/UB Spring 2019/Deep Learning/Assignment 3/Data/Test10000/{0}'.format(filename),0)/255.0
        i = i + 1
trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2])
testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2])

numLayers = [10,50]
doBatchNorm=True

NN = Network(trainX,trainY,numLayers,10,1.0,ActivationType.RELU, ActivationType.SOFTMAX) # try SOFTMAX
#NN.Train(30,0.1,0.1, GradDescType.STOCHASTIC,1)

#NN.Train(50,0.1,0.0,50,LROptimizerType.NONE, doBatchNorm) # with BatchNorm = True, try with and without ADAM
    
#(self, epochs, learningRate, lambda1, batchsize=1, LROptimization=LROptimizerType.NONE, doBatchNorm=False):

NN.Train(30,0.1,0,5,LROptimizerType.ADAM, True) # with doBatchNorm = False,90%

print("done training , starting testing..") 
accuracyCount = 0
for i in range(testY.shape[0]):
        # do forward pass
        #a2 = NN.Computation(testX[i])
        a2 = NN.Computation(testX[i], doBatchNorm, BatchNormMode.TEST)   # determine index of maximum output value
        maxindex = a2.argmax(axis = 0)
        if (testY[i,maxindex] == 1):
            accuracyCount = accuracyCount + 1
print("Accuracy count = " + str(accuracyCount/10000.0))