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
            a = np.random.randn(size[i],1)
            self.bias.append(a)
            
    def sigmoid(self,z):
        y = 1.0/(1.0 + np.exp(-z)) 
        return y

    def activation(self,y):
        z = self.sigmoid(y)
        return z
            
    def mini_batch(self,mini_batch_size,training_data):
        np.random.shuffle(training_data)
        if (mini_batch_size >= len(training_data)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()
        selector = np.random.randint(len(training_data)-mini_batch_size)
        mini_batch = training_data[selector:selector+mini_batch_size] #shape [mini_batch_size,2]
        img_data,label_data = [i[0] for i in mini_batch],[i[1] for i in mini_batch]
        label_data = np.transpose(label_data)
        img_data_reshaped = np.reshape(img_data,(np.size(img_data[0]),len(img_data)))
        return label_data,img_data_reshaped
    
    def forwardPath(self,img_data):
        activations = []
        activations = [img_data]
        for layer in range(len(self.weight)):
            z = np.dot(self.weight[layer],activations[layer])+self.bias[layer]
            activations.append(self.activation(z))
        return activations
    
    def sigmoid_prime(self,z):
        return (z)*(1-z)
        
    def weightTraining(self,l_rate,label,activations,batch_size):
        output_error = activations[-1] - label#output = [10,m]
        errors = []
        errors.append(output_error)
        for i in range(len(activations)-1):
            error = self.sigmoid_prime(activations[-2-i]) * (np.dot(self.weight[-1-i].T,errors[-1-i]))
            errors.insert(0,error)
        for j in range(len(self.weight)):
            self.weight[j] = self.weight[j] + l_rate/batch_size*np.dot(errors[1+j],activations[j].T)
            self.bias[j] = self.bias[j] + np.sum(errors[1+j],axis=1,keepdims = True)/batch_size

    def evaluate(self,test_data):
        img,label = [i[0] for i in test_data],[i[1] for i in test_data]
        img = np.reshape(img,(np.size(img[0]),len(img)))
        a = self.forwardPath(img)
        a = a[-1].T

        test_results = [(np.argmax(a[i]),np.argmax(label[i])) for i in range(len(label))]
        
        a = sum(test_results[i][0] == test_results[i][1] for i in range(len(label)))
        
        print ("accuracy = %f" %(a/len(label)*100))
        
    def backProps(self,l_rate,mini_batch_size,training_data,test_data):
        label_data,img_data_reshaped = self.mini_batch(mini_batch_size,training_data)
        for i in range(10):
            activations = self.forwardPath(img_data_reshaped)
            self.weightTraining(l_rate,label_data,activations,mini_batch_size)
            
        self.evaluate(test_data)  
        
f = data_loader.data_loader()
training_data = f.loading("training")
training_data = np.array(list(training_data))

test_data = f.loading("testing")
test_data = np.array(list(test_data))

test = backProp([28*28,300,70,10])

test.backProps(0.1,40000 ,training_data,test_data)
