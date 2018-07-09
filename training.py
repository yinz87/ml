# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:58:36 2018

@author: yinz
"""

import data_loader
import numpy as np

class backProp():
    def __init__(self,inputsize,hiddensize,outputsize):

        self.w1 = np.random.randn(inputsize,hiddensize)
        self.b1 = np.random.randn(hiddensize)
        self.w2 = np.random.randn(hiddensize,outputsize)
        self.b2 = np.random.randn(outputsize)
        self.hiddensize = hiddensize
        self.outputsize = outputsize
        
    def sigmoid(self,z):
        y = 1/(1+ np.exp(-z))
        return y

    def activation(self,weight,bias,img_data):
        z = self.sigmoid((np.dot(weight,img_data))+bias)
        return z
            
    def forwardPath(self,trails,mini_batch_size,datafile):
        w1,w2,b1,b2 = self.w1,self.w2,self.b1,self.b2

        np.random.shuffle(datafile)
        if (mini_batch_size >= len(datafile)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()

        selector = np.random.randint(len(datafile)-mini_batch_size)
        
        
        mini_batch = datafile[selector:selector+mini_batch_size]
        img_data,label_data = [i[0] for i in datafile],[i[1] for i in datafile]
        img_data_reshaped = np.reshape(img_data,(len(img_data),np.size(img_data[0])))




        hOutput = []
        for i in range(self.hiddensize):
            hiddenOutput = self.activation(w1[:,i],b1[i],img)
            hOutput.append(hiddenOutput)
            #print (hOutput)
            
        fOutput = []
        for j in range(self.outputsize):
            finalOutput = self.activation(w2[:,j],b2[j],hOutput)
            fOutput.append(finalOutput)
        #print (len(fOutput))
        #print (fOutput)
        return fOutput
    
    def compareResult(self,dataFile):
        testOutput = np.max(self.forwardPath(self.dataFile))
        return (testOutput)
        
        
        
f = data_loader.data_loader()
datafile = f.loading("training")
datafile = list(datafile)

test = backProp(28*28,10,10)

for i in range(len(datafile)):
    tests = test.forwardPath(2,10000,datafile)
    result.append(np.argmax(tests))
print (result)


