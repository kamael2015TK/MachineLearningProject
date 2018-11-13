##
# Author Janus Bastian Lansner S145349
# contributors Taras Karpin S153067 30%

#!/usr/bin/python
import numpy as np
import xlrd
import math
from scipy import stats

def standardize(dataSet):
    handledData = handleMissingData(dataSet)
    return stats.zscore(handledData)

#
# this function takes missing values marked with -9 and replace those with mean
#
def handleMissingData(data):
    observations = len(data)
    features = len(data[0])
    for j in range(0, features):
        mean = 0
        count = 0
        for i in range(0, observations):
            if(data[i][j] != -9):
                mean = mean + data[i][j]
                count += 1
        mean = mean/count
        for i in range(0, observations):
            if(data[i][j] == -9):
                data[i][j] = mean
    return data

def arrayToBinary(data) : 
    for obj in range(0,len(data)) :
        if(data[obj] != 0) :
            data[obj] = 1
    return data