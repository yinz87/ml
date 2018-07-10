# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:58:36 2018

@author: yinz
"""

import data_loader
import numpy as np

class backProp():
    def __init__(self,inputsize,hiddensize,outputsize):

        self.w1 = np.random.randn(hiddensize,inputsize) #weight between neuron and intput data shape
        self.b1 = np.random.randn(hiddensize) # bias for each neuron in hidden layer
        self.w2 = np.random.randn(outputsize,hiddensize) # wegith between hidden neuron and output neuron
        self.b2 = np.random.randn(outputsize) #bias for each neuron in output layer
        self.hiddensize = hiddensize
        self.outputsize = outputsize
        
    def sigmoid(self,z):
        y = 1/(1+ np.exp(-z)) 
        return y

    def activation(self,weight,bias,img_data):
        z = self.sigmoid((np.dot(img_data,weight))+bias) # shape.img_data = [intputsize x 1]
        return z
            
    def forwardPath(self,epoch,mini_batch_size,datafile):
        w1,w2,b1,b2 = self.w1,self.w2,self.b1,self.b2
        np.random.shuffle(datafile)
        if (mini_batch_size >= len(datafile)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()

        for epoch in range(epoch):
                
            selector = np.random.randint(len(datafile)-mini_batch_size)
            
            mini_batch = datafile[selector:selector+mini_batch_size] #shape [mini_batch_size,2]

            self.img_data,self.label_data = [i[0] for i in mini_batch],[i[1] for i in mini_batch]#shape.img_data[mini_batch_size,8,8],shape.label_data[mini_batch_size)]
            self.img_data_reshaped = np.reshape(self.img_data,(len(self.img_data),np.size(self.img_data[0]))) #after reshape. shape = [mini_batch_size, 8x8]
            
            hOutput = []
            fOutput = []
            for i in range(self.hiddensize): #for each nueuron
                hiddenOutput = self.activation(w1[i],b1[i],self.img_data_reshaped) # dot product of ith column and img data
                hOutput.append(hiddenOutput)
            hOutput = np.transpose(hOutput) #shape.hOutput = [mini_batch_size,hidden_neuron_size]
                
            for j in range(self.outputsize):
                finalOutput = self.activation(w2[j],b2[j],hOutput)
                fOutput.append(finalOutput)
            fOutput = np.transpose(fOutput)
            return hOutput,fOutput
#            print ("trial #%d" %epoch)
#    
#            correctAnswer, errorTable = self.evaluateResult(fOutput,label_data)
#            
#            print ("accuracy is %0.2f" %(correctAnswer/len(finalOutput)*100))
    
    def evaluateResult(self,Output):
        correctAnswer = 0
        errorTable = []
        for i in range(len(Output)):
            error = self.label_data[i] - np.argmax(Output[i])
            errorTable.append(error)
            if error == 0:
                correctAnswer += 1
        return correctAnswer, errorTable
        
        
    def errorGradientFun(layerOutput,error):
        errorGradient = layerOutput*(1-layerOutput) * error
        return errorGradient
        
    def weightTraining(self,errorTable):
        w1,w2,b1,b2 = np.copy(self.w1),np.copy(self.w2),np.copy(self.b1),np.copy(self.b2)
        
        
        #e2 is the errorGradient at output layer
        e2 = self.errorGradientFun()
        

        #e1 is the errorGraident at hidden layer    
    
        
        
f = data_loader.data_loader()
datafile = f.loading("training")
datafile = list(datafile)

test = backProp(28*28,10,10)

tests = test.forwardPath(2,10000,datafile)
hOutput,fOutput = tests
tests = test.evaluateResult(fOutput)
correctAnswer,errorTable = tests
test.weightTraining(errorTable)
