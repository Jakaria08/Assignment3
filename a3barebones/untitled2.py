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

trainset, testset = dtl.load_susy(trainsize,testsize)

X = trainset[0]
Y = trainset[1]

print(X.shape) 
print(Y.shape) 