import numpy as np
from BatchNormMode import BatchNormMode
from LROptimizerType import LROptimizerType
from ActivationType import ActivationType

class Layer(object):
    
    def __init__(self, numNeurons, numNeuronsPrevLayer, batchsize,LastLayer = False, dropOut = 0.2, activationType=ActivationType.SIGMOID):
        
        ## initialize all the model parameters declared above, necessary for the NN. ( Weights, Biases, Gradients ).

        self.numNeurons = numNeurons
        self.LastLayer = LastLayer
        self.numNeuronsPrevLayer = numNeuronsPrevLayer
        self.activationFunction = activationType
        self.w = np.random.uniform(low = -0.1, high = 0.1, size=(numNeurons, numNeuronsPrevLayer))
        self.b = np.random.uniform( low = -1, high = 1, size=(numNeurons))
        self.delta = np.zeros((numNeurons, numNeuronsPrevLayer))
        #self.a = np.zeros((numNeurons,1))
        self.df = np.zeros((numNeurons))
        self.wgrad = np.zeros((numNeurons, numNeuronsPrevLayer)) ## Matrix W in Notes
        self.bgrad = np.zeros((numNeurons))
        self.zeroout = None
        self.dropOut = dropOut

        #-------------BATCH NORMALIZATION--------------
        self. mu = np.zeros((numNeurons)) # batch mean
        self.sigma2 = np.zeros((numNeurons)) # sigma^2 for batch
        self.epsilon = 1e-6
        self.gamma = np.random.rand(1)
        self.beta= np.random.rand(1)
        self.S = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.Shat = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.Sb = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.runningmu = np.zeros((numNeurons))
        self.runningsigma2 = np.zeros((numNeurons))
        self.dgamma = np.zeros((numNeurons))
        self.dbeta = np.zeros((numNeurons))
        self.delta = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.deltabn = np.zeros((numNeurons,numNeuronsPrevLayer))
 #----------------------------------------------------------------

 #----------following for implementing ADAM-----------------------
        self.mtw = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.mtb = np.zeros((numNeurons))
        self.vtw = np.zeros((numNeurons,numNeuronsPrevLayer))
        self.vtb = np.zeros((numNeurons))
 #----------------------------------------------------------------



    def sigmoid(self,x):
 
        return 1 / (1 + np.exp(-x))  ##SIGMOID

    def TanH(self, x):
        
        return np.tanh(x) ##TANH

    def Relu(self, x):
        
        return np.maximum(0,x) ##RELU

    def Softmax(self, x):
        
        if (x.shape[0] == x.size):
            ex = np.exp(x)
            return ex/ex.sum()
 
        ex = np.exp(x)

        for i in range(ex.shape[0]):
            denom = ex[i,:].sum()
            ex[i,:] = ex[i,:]/denom
        return ex
    



    def Computation(self,indata, doBatchNorm=False, batchMode = BatchNormMode.TRAIN):

        self.S = np.dot(indata,self.w.T) + self.b

        if( doBatchNorm == True):
            if(batchMode == BatchNormMode.TRAIN):
                self.mu = np.mean(self.S, axis=0) # batch mean
                self.sigma2 = np.var(self.S,axis=0) # batch sigma^2
                self.runningmu = 0.9 * self.runningmu + (1 - 0.9)* self.mu
                self.runningsigma2 = 0.9 * self.runningsigma2 + (1 - 0.9)* self.sigma2
            else:
                self.mu = self.runningmu
                self.sigma2 = self.runningsigma2
            self.Shat = (self.S - self.mu)/np.sqrt(self.sigma2 + self.epsilon)
            self.Sb = self.Shat * self.gamma + self.beta
            sum = self.Sb
        else:
                sum = self.S

        if( self.activationFunction == ActivationType.SIGMOID ) :  ## SIGMOID ACTIVATION FUNCTION
            self.a = self.sigmoid(sum)
            self.df = self.a * (1- self.a)

        if( self.activationFunction==ActivationType.RELU) : ## RELU ACTIVATION FUNCTION 
            self.a = self.Relu(sum)
            #self.derivAF = 1.0 * (self.a > 0)
            epsilon=1.0e-6
            self.df = 1. * (self.a > epsilon)
            self.df[self.df == 0] = epsilon
            
        if( self.activationFunction==ActivationType.SOFTMAX) : ## SOFTMAX ACTIVATION FUNCTION
            self.a = self.Softmax(sum)
            self.df = None


        if( self.activationFunction==ActivationType.TANH) : ## TANH ACTIVATION FUNCTION
            self.a = self.TanH(sum)
            self.df = (1 - self.a * self.a) 

        if (self.LastLayer == False):
            self.zeroout = np.random.binomial(1,self.dropOut,(self.numNeurons))/self.dropOut
            self.a = self.a * self.zeroout
            self.df = self.df * self.zeroout

    def ClearWBGrads(self):  # Used after updating gradients to clear accumulation and restart new batch. 
             
        self.wgrad = np.zeros((self.numNeurons, self.numNeuronsPrevLayer))
        self.bGrad = np.zeros((self.numNeurons,1))
