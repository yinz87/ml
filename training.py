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
            
    def forwardPath(self,img,label):
        w1,w2,b1,b2 = self.w1,self.w2,self.b1,self.b2
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
a = f.loading("training")
img,label = 
imgshaped = np.reshape(img,(len(img),28*28))
test = backProp(28*28,10,10)
result = []
for i in range(len(label)):
    tests = test.forwardPath(imgshaped[i])
    result.append(np.argmax(tests))
print (result)


