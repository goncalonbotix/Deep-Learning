import math
import numpy as np
from Layer import *
from GradDescType import GradDescType
from BatchNormMode import BatchNormMode
from LROptimizerType import LROptimizerType
from sklearn.utils import shuffle

class Network(object):
    
    def __init__(self,TrainX,TrainY, numLayers,batchsize,dropOut = 1.0, activationFunction=ActivationType.SIGMOID, lastLayerActivationFunction = ActivationType.SIGMOID):
        
        self.TrainX = TrainX
        self.TrainY = TrainY
        self.numLayers = numLayers
        self.batchsize = batchsize
        self.Layers = [] ## Python List with all Layers.
        self.lastLayerActivationFunction = lastLayerActivationFunction
        self.gradDescType = 1
        self.LROptimization = LROptimizerType.ADAM
        self.BatchNormMode = BatchNormMode
        


        for i in range(len( self.numLayers )):

            if (i == 0): # first layer
 
                layer = Layer(numLayers[i],TrainX.shape[1],batchsize,False,dropOut, activationFunction)

            elif (i == len(numLayers)-1): # last layer
 
                layer = Layer(TrainY.shape[1],numLayers[i-1], batchsize,True, dropOut, lastLayerActivationFunction)

            else: # intermediate layers
 
                layer = Layer(numLayers[i],numLayers[i-1],batchsize,False,dropOut, activationFunction)

            self.Layers.append(layer);

    def Computation(self,indata, doBatchNorm=False, batchMode= BatchNormMode.TEST): # Goes through all layers and executes method inside layer called Computation

        self.Layers[0].Computation(indata, doBatchNorm, batchMode)
 
        for i in range(1,len(self.numLayers)):
            
            self.Layers[i].Computation(self.Layers[i-1].a, doBatchNorm, batchMode)
 
        return self.Layers[len(self.numLayers)-1].a


    def Train(self, epochs, learningRate, lambda1, batchsize=1, LROptimization=LROptimizerType.NONE, doBatchNorm=False):

        iter=0
        
        for i in range (epochs) :
            loss = 0
            self.TrainX, self.TrainY = shuffle(self.TrainX, self.TrainY, random_state=0)


            for j in range(0,self.TrainX.shape[0],batchsize):

                # get (X, y) for current minibatch/chunk
                X_train_mini = self.TrainX[j:j + batchsize]
                Y_train_mini = self.TrainY[j:j + batchsize]

                self.Computation(X_train_mini, doBatchNorm, batchMode=BatchNormMode.TRAIN)

                if( self.lastLayerActivationFunction == ActivationType.SOFTMAX):

                    loss += -(Y_train_mini * np.log(self.Layers[len(self.numLayers)-1].a+0.001)).sum() ## LOSS FUNCTION
                else: 
                    loss += ((self.Layers[len(self.numLayers)-1].a - Y_train_mini) * (self.Layers[len(self.numLayers)-1].a - Y_train_mini)).sum()

                k = len(self.numLayers) -1 

                ## Compute Deltas on all layers and all Wgrads and Bgrads for all layers

                while(k >= 0):
                    
                    if (k == len(self.numLayers)-1): # Last Layer
                        if (self.lastLayerActivationFunction == ActivationType.SOFTMAX):

                            self.Layers[k].delta = -Y_train_mini+ self.Layers[k].a

                        else:

                            self.Layers[k].delta = -(Y_train_mini-self.Layers[k].a) * self.Layers[k].df

                    else: # intermediate layer

                            self.Layers[k].delta = np.dot(self.Layers[k+1].delta,self.Layers[k+1].w) * self.Layers[k].df

                    if (doBatchNorm == True):

                        self.Layers[k].dbeta = np.sum(self.Layers[k].delta,axis=0)
                        self.Layers[k].dgamma = np.sum(self.Layers[k].delta * self.Layers[k].Shat,axis=0)
                        self.Layers[k].deltabn = (self.Layers[k].delta * self.Layers[k].gamma)/(batchsize*np.sqrt(self.Layers[k].sigma2 + self.Layers[k].epsilon )) * (batchsize -1 - (self.Layers[k].Shat * self.Layers[k].Shat))

                    if(k > 0):
                            prevOut = self.Layers[k-1].a
                    else:
                            prevOut = X_train_mini

                    if( doBatchNorm == True):

                        self.Layers[k].wgrad = np.dot(self.Layers[k].deltabn.T,prevOut)
                        self.Layers[k].bgrad = self.Layers[k].deltabn.sum(axis=0)

                    else:
                        self.Layers[k].wgrad = np.dot(self.Layers[k].delta.T,prevOut)
                        self.Layers[k].bgrad = self.Layers[k].delta.sum(axis=0)
                    k = k - 1

                iter = iter+1
                self.UpdateGradsBiases(learningRate, lambda1, batchsize, LROptimization, iter, doBatchNorm)

            print("Iter = " + str(i) + " Error = "+ str(loss))

               

    def UpdateGradsBiases(self, learningRate, lambda1, batchSize, LROptimization, iter, doBatchNorm):  # update weights and biases for all layers
 
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8
            for ln in range(len(self.numLayers)):
                if (LROptimization == LROptimizerType.NONE):
                    self.Layers[ln].w = self.Layers[ln].w - learningRate * (1/batchSize) * self.Layers[ln].wgrad - learningRate * lambda1 * self.Layers[ln].w.sum()
                    self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * self.Layers[ln].bgrad

                elif (LROptimization == LROptimizerType.ADAM):
                
                    gtw = self.Layers[ln].wgrad
                    gtb = self.Layers[ln].bgrad
                    self.Layers[ln].mtw = beta1 * self.Layers[ln].mtw + (1 - beta1) * gtw
                    self.Layers[ln].mtb = beta1 * self.Layers[ln].mtb + (1 - beta1) * gtb
                    self.Layers[ln].vtw = beta2 * self.Layers[ln].vtw + (1 - beta2) * gtw*gtw
                    self.Layers[ln].vtb = beta2 * self.Layers[ln].vtb + (1 - beta2) * gtb*gtb
                    mtwhat = self.Layers[ln].mtw/(1 - beta1**iter)
                    mtbhat = self.Layers[ln].mtb/(1 - beta1**iter)
                    vtwhat = self.Layers[ln].vtw/(1 - beta2**iter)
                    vtbhat = self.Layers[ln].vtb/(1 - beta2**iter)
                    self.Layers[ln].w = self.Layers[ln].w - learningRate * (1/batchSize) * mtwhat /((vtwhat**0.5) + epsilon)
                    self.Layers[ln].b = self.Layers[ln].b - learningRate * (1/batchSize) * mtbhat /((vtbhat**0.5) + epsilon)

                if (doBatchNorm == True):
                    self.Layers[ln].beta = self.Layers[ln].beta - learningRate * self.Layers[ln].dbeta
                    self.Layers[ln].gamma = self.Layers[ln].gamma - learningRate * self.Layers[ln].dgamma
