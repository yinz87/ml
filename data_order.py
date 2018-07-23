# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:42:49 2018

@author: yinz
"""
import numpy as np
import pandas as pd

class data_order():
    def __init__(self):
        pass
    
    def label_unpack(self,data):    
        img,label = [i[0] for i in data],[i[1] for i in data]
        label = [np.argmax(label[i]) for i in range (len(data))]
        Image_Label = lambda i: (img[i],label[i]) 
        datas = []
        for j in range(len(label)):
            datas.append(Image_Label(j))
        return datas
    
    def sort(self,data):
        d = self.label_unpack(data)
        d = pd.DataFrame(d)
        d = d.sort_values([1])
        d = d.reset_index(drop=True)
        #d = d.to_records(index=False)
        return self.label_treatment(d)
        
    def label_treatment(self,data):
        label_like_shape = np.zeros((len(data),10)) #make a temp label where all value is 0
        for i in range(len(data)):
            label_like_shape[i][data[1][i]] = 1.0 #replace 1 for corresponding label value
        replacement = pd.DataFrame({"new_label":list(label_like_shape)})
        data[1] = replacement["new_label"]
        data = data.to_records(index=False)
        return data
#    
    
    
#f = data_loader.data_loader()
#training_data = f.loading("training")
#training_data = list(training_data)
#
#g = data_order()
#h = g.unpack(training_data)

#h = g.sort(training_data)
#print (h)
