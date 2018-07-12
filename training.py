# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:58:36 2018

@author: yinz
"""

import data_loader
import numpy as np

class backProp():
    def __init__(self,size):
        self.weight = []
        self.bias = []
        for i in range (1,len(size)):
            self.weight.append(np.random.randn(size[i],size[i-1]))
            bias = np.random.randn(size[i])
            self.bias.append(bias.reshape(len(bias),1))
            
    def sigmoid(self,z):
        y = 1/(1+ np.exp(-z)) 
        return y

    def activation(self,data,w,b):
        z = self.sigmoid(np.dot(w,data)+b)
        return z
            
    def forwardPath(self,mini_batch_size,datafile):
        np.random.shuffle(datafile)
        if (mini_batch_size >= len(datafile)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()
        selector = np.random.randint(len(datafile)-mini_batch_size)
        mini_batch = datafile[selector:selector+mini_batch_size] #shape [mini_batch_size,2]
        self.img_data,self.label_data = [i[0] for i in mini_batch],[i[1] for i in mini_batch]
        self.label_data = np.transpose(self.label_data)
        self.img_data_reshaped = np.reshape(self.img_data,(np.size(self.img_data[0]),len(self.img_data))) 
        output = [self.img_data_reshaped]
        for i in range(len(self.weight)):
            output.append(self.activation(output[i],self.weight[i],self.bias[i]))
        return output
        
    def sigmoid_prime(self,output):
        z = output.dot(1-output)
        return z
        
    def weightTraining(self,output,l_rate):
        last_error = self.label_data - output[-1]
        print (last_error)
        
f = data_loader.data_loader()
datafile = f.loading("training")
datafile = np.array(list(datafile))

test = backProp([28*28,20,10])

tests = test.forwardPath(100,datafile)
output = tests
tests = test.weightTraining(output,0.1)

