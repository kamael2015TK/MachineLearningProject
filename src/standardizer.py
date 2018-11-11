#!/usr/bin/python
import numpy as np
import xlrd
import math
from scipy import stats

def standardize(dataSet):
    handledData = handeleMissingData(dataSet)
    return stats.zscore(handledData)

#
# this function takes missing values marked with -9 and replace those with mean
#
def handeleMissingData(data):
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