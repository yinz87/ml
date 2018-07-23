# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 14:02:19 2018

@author: yinz
"""
import data_order
import data_loader
import numpy as np


class backProp():
    def __init__(self,size):
        self.weight = []
        self.bias = []
        for i in range (1,len(size)):
            self.weight.append(np.random.randn(size[i],size[i-1])*np.sqrt(2.0/size[i-1])) #setup the weight base on size
            self.bias.append(0) #setup the bias base on size

    def mini_batch(self,mini_batch_size,training_data): #random sampling size
       #np.random.shuffle(training_data)
        n = len(training_data)
        mini_batches = []
        
        for i in range(0,n,mini_batch_size):
            mini_batch = training_data[i:i+mini_batch_size] 
            img,label = [i[0] for i in mini_batch],[j[1] for j in mini_batch]
            img,label = np.transpose(img),np.transpose(label) #get label in [10,size]
            mini_batches.append((img,label))
        return mini_batches
    
    def softMax(self,z):
        #softmax activation function exp(z)/sum(exp(z))
        c = np.amax(z,axis=0) #max output for numeric stability
        z = (np.exp(z-c))
        z_sum = np.sum(z,axis=0,keepdims=True)
        z = z/ (z_sum)
        return z
    
    def Sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def Sigmoid_prime(self,activation):
        return activation*(1-activation)
    
    def forwardPath(self,img):
        activations = [img] #a = activation, a[0] is the input img data
        #activaiton for hidden layer
        for hiddenLayer in range(len(self.weight)-1): #hidden network uses .ReLu
            z = np.dot(self.weight[hiddenLayer],activations[hiddenLayer])+self.bias[hiddenLayer]
            activation = self.Sigmoid(z)
            activations.append(activation)
            
        #output layer
        z = np.dot(self.weight[-1],activations[-1])+self.bias[-1]
        activation = self.softMax(z)
        activations.append(activation)
        return activations

    def cost (self,activation,label,reg_strength):
        p = np.sum((-label*np.log(activation)) + ((1-label)*np.log((1-activation))))
        reg_loss = 1/2 * reg_strength*np.sum(self.weight[-1]*self.weight[-1]) #1/2*lambda* w^2
        loss = np.sum(p)/len(label[0]) + reg_loss
        return loss
    
    def weightTraining(self,l_rate,label,activations,batch_size,reg_strength):
        # for output layer
        dZ = activations[-1] - label
        dZs = []
        dWs = []
        dBs = []
        reg = self.weight[-1]*reg_strength 
        dZs.append(dZ)
        dWs.append(reg+1/batch_size * np.dot(dZs[-1],activations[-2].T)) #dW = dot(dZ(L),dA(L-1))/m
        dBs.append(1/batch_size * np.sum(dZs[-1],axis=1,keepdims=True))
        
#        #for hidden layers
        for i in range(len(activations)-2):
            reg = self.weight[-2-i]*reg_strength
            dZ = np.dot(self.weight[-1-i].T,dZs[-1-i]) # dZ(L-1) = dot(w(L),dZ(L))
            dZ = self.Sigmoid_prime(activations[-2-i]) #ReLu backProp, using -2 becuase, last layer which is the output layer has been computed, we start from the layer before the last layer
            dZs.insert(0,dZ)
            dWs.insert(0,reg*np.dot(dZs[-2-i],activations[-3-i].T,)/batch_size) #dW = dot(dZ(L),dA(L-1)
            dBs.insert(0,np.sum(dZs[-2-i],axis=1,keepdims=True)/batch_size) #dB = 1-D vector 

        for j in range(len(dWs)): #update all weights and bias
            self.weight[j] = self.weight[j] - l_rate*dWs[j]
            self.bias[j] = self.bias[j] - l_rate*dBs[j]

    def prediction(self,test_data):
        img,label = [i[0] for i in test_data],[i[1] for i in test_data]
        img = np.transpose(img)
        a = self.forwardPath(img)
        a = a[-1].T
        test_results = [(np.argmax(a[i]),np.argmax(label[i])) for i in range(len(label))]
        total_corrects = sum(test_results[i][0] == test_results[i][1] for i in range(len(label)))  
        print ("accuracy = %f" %(total_corrects/len(label)*100))
        
    # warp function to run the whole NN
    def backProps(self,epochs,l_rate,mini_batch_size,training_data,test_data,reg_strength = 0.1):
        for i in range(epochs):
            mini_batches = self.mini_batch(mini_batch_size,training_data)
            print ("#%d epoch:" %i)
            for j in range(len(mini_batches)):  
                img,label = mini_batches[j]
                activations = self.forwardPath(img)
                self.weightTraining(l_rate,label,activations,mini_batch_size,reg_strength)
            loss_error = self.cost(activations[-1],label,reg_strength)
            print ("loss = %f" %loss_error)
            self.prediction(test_data)  
            
f = data_loader.data_loader()
training_data = f.loading("training")
training_data = np.array(list(training_data)) #get training data
test_data = f.loading("testing")
test_data = np.array(list(test_data)) # get testing data

g = data_order.data_order()
training_data_ordered = g.sort(training_data)
test = backProp([28*28,400,10]) #setup the network size in [i,x,o], where i is the input size, o is the ouput size, x is the hidden layer size. Multiple X can be added so [i,x1,x2,xn,o] is possible
test.backProps(10,0.5,10000,training_data,test_data) 
