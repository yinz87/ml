# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:24:59 2018

@author: yinz
"""
import numpy as np
a = [1,2,3,4]
b = []
b.append(a)
b.append(a)
print (b)

c = np.array([1,2])
c = c.reshape(2,1)

d = b + c
print (d)