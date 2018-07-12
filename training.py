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
        y = 1.0/(1.0+ np.exp(-z)) 
        return y

    def activation(self,y):
        z = self.sigmoid(y)
        return z
            
    def mini_batch(self,mini_batch_size,datafile):
        np.random.shuffle(datafile)
        if (mini_batch_size >= len(datafile)):
            raise ValueError("mini_batch_size should be less than the datafile size")
            exit()
        selector = np.random.randint(len(datafile)-mini_batch_size)
        mini_batch = datafile[selector:selector+mini_batch_size] #shape [mini_batch_size,2]
        img_data,label_data = [i[0] for i in mini_batch],[i[1] for i in mini_batch]
        label_data = np.transpose(label_data)
        img_data_reshaped = np.reshape(img_data,(np.size(img_data[0]),len(img_data))) 
        return label_data,img_data_reshaped
        
    def forwardPath(self,img_data):
        activations = [img_data]
        zs = []
        for layer in range(len(self.weight)):
            z = np.dot(self.weight[layer],activations[layer])+self.bias[layer]
            zs.append(z)
            activations.append(self.activation(z))
        return activations,zs
        
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
        
    def weightTraining(self,l_rate,label,output,zs):
        output_error = output[-1] - label#output = [10,m]
        errors = []
        errors.append(output_error)
        error_gradients = []
        

        for i in range(len(output)-1):
            error_gradient = self.sigmoid_prime(zs[-1-i]) * errors[i]
            error_gradients.append(error_gradient)
            delta_w = l_rate*error_gradients[i].dot(output[-2-i].T)
            errors.append(self.weight[-1-i].T.dot(errors[i]))
            self.weight[-1-i] = self.weight[-1-i] - delta_w
            self.bias[-1-i] = self.bias[-1-i]*error_gradient[i]*(-1)
        
    def evaluate(self,label,output):
        output,label = output.T,label.T
        correct_label = 0
        for i in range(len(label)):
            if np.argmax(label[i]) == np.argmax(output[i]):
                correct_label += 1
       # print ("accuracy is %f" %(correct_label/(len(label))*100))
        
    def backProps(self,l_rate,mini_batch_size,datafile):
        label_data,img_data = self.mini_batch(mini_batch_size,datafile)
        outputs = []
        for i in range(10):
            activations,zs = self.forwardPath(img_data)
            self.evaluate(label_data,activations[-1])
            self.weightTraining(l_rate,label_data,activations,zs)
            outputs.append(activations)
        return outputs,label_data
            
f = data_loader.data_loader()
datafile = f.loading("training")
datafile = np.array(list(datafile))

test = backProp([28*28,25,10])

tests,label_data = test.backProps(0.1,2,datafile)

#a = np.round(tests[5][-1].T,decimals=1)
#b = np.round(tests[6][-1].T,decimals=1)
#c = label_data.T

#print (a)
#print ( )
#print (b)
#print ( )
#print (c)
#print (np.argmax(b[1]))
#print (np.argmax(label_data.T[1]))