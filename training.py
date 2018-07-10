# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:58:36 2018

@author: yinz
"""

import data_loader
import numpy as np

class backProp():
    def __init__(self,inputsize,hiddensize,outputsize):

        self.w1 = np.random.randn(inputsize,hiddensize) #weight between neuron and intput data
        self.b1 = np.random.randn(hiddensize) # bias for each neuron in hidden layer
        self.w2 = np.random.randn(hiddensize,outputsize) # wegith between hidden neuron and output neuron
        self.b2 = np.random.randn(outputsize) #bias for each neuron in output layer
        self.hiddensize = hiddensize
        self.outputsize = outputsize
        
    def sigmoid(self,z):
        y = 1/(1+ np.exp(-z)) 
        return y

    def activation(self,weight,bias,img_data):
        z = self.sigmoid((np.dot(weight,img_data))+bias) # sum (x x w for one neuron in full connnect network + its own bias)
        return z
            
    def forwardPath(self,trials,mini_batch_size,datafile):
        w1,w2,b1,b2 = self.w1,self.w2,self.b1,self.b2
        np.random.shuffle(datafile)
        if (mini_batch_size >= len(datafile)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()

        for trials in range(trials):
                
            selector = np.random.randint(len(datafile)-mini_batch_size)
            
            
            mini_batch = datafile[selector:selector+mini_batch_size]
            
    
            img_data,label_data = [i[0] for i in mini_batch],[i[1] for i in mini_batch]
            img_data_reshaped = np.reshape(img_data,(np.size(img_data[0]),len(img_data)))
            
            hOutput = []
            fOutput = []
            for i in range(self.hiddensize): #for each nueuron
                hiddenOutput = self.activation(w1[:,i],b1[i],img_data_reshaped) #
                hOutput.append(hiddenOutput)
    
            for j in range(self.outputsize):
                finalOutput = self.activation(w2[:,j],b2[j],hOutput)
                fOutput.append(finalOutput)
    
            fOutput = np.transpose(np.reshape(fOutput,(10,mini_batch_size)))
    
            print ("trial #%d" %trials)
    
            self.evaluateResult(fOutput,label_data)
    
    def evaluateResult(self,finalOutput,label_data):
        correctAnswer = 0
        for i in range(len(finalOutput)):
            if np.argmax(finalOutput[i]) == label_data[i]:
                correctAnswer += 1
        print (correctAnswer)
        
        print ("accuracy is %f" %(correctAnswer/len(finalOutput)*100))
        
        
        
f = data_loader.data_loader()
datafile = f.loading("training")
datafile = list(datafile)

test = backProp(28*28,10,10)

tests = test.forwardPath(10,10000,datafile)
#print (result)


