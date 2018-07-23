# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 00:58:38 2018

@author: yinz
"""
import struct
import numpy as np
import matplotlib as mp
from matplotlib import pyplot as plt
np.set_printoptions(edgeitems = 100)

class data_loader():

    def loading(self,sets = "training"):
        if (sets == "training"):
            Image_File_Name = "train-images.idx3-ubyte"
            Label_File_Name ="train-labels.idx1-ubyte"
        elif (sets == "testing"):
            Image_File_Name = "t10k-images.idx3-ubyte"
            Label_File_Name = "t10k-labels.idx1-ubyte"
        else:
            raise ValueError
        
        with open (Label_File_Name,'rb') as label_data:
            magic, size = struct.unpack(">ll",label_data.read(8))
            label = np.fromfile(label_data,dtype="int8")
            
            
        with open (Image_File_Name,'rb') as img_data:
            magic,size,row,col = struct.unpack(">llll",img_data.read(16))
            img = np.fromfile(img_data,dtype="uint8").reshape(len(label),row*col)
            img = img/np.float64(256)
            
        label_like_shape = np.zeros((len(label),10)) #make a temp label where all value is 0

        for i in range(len(label)):
            label_like_shape[i][label[i]] = 1.0 #replace 1 for corresponding label value

        label = label_like_shape
        Image_Label = lambda i: (img[i],label[i]) 
        for j in range(len(label)):
            yield Image_Label(j)

    def drawing(self,img):
        plt.imshow(img,cmap=mp.cm.Greys)
        plt.show()