# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 21:53:34 2018

@author: Jakaria
"""

from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import classalgorithms as algs

trainsize = 5000
testsize = 5000
XT = list()

trainset, testset = dtl.load_susy(trainsize,testsize)

X = trainset[0]
Y = trainset[1]

Z = np.c_[ X, Y ]  

idx = Z[:, 9] == 0
X0 = Z[idx]

idx = Z[:, 9] == 1
X1 = Z[idx]

percentage0 = int(np.rint((X0.shape[0]/X.shape[0])*100))
percentage1 = int(np.rint((X1.shape[0]/X.shape[0])*100))


for i in range(5):
    XT.append(np.concatenate((X0[i*percentage0:percentage0*(i+1)],X1[i*percentage1:percentage1*(i+1)]),axis = 0))
    print(XT[i].shape) 
    
print(XT[0][:,0:9])





"""
unique = np.unique(X[:, 0:1])
answer = []
for element in unique:
    present = a[:,0]==element
    answer.append(np.extract(present,a[:,-1]))
print (answer)
"""