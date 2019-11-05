# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 02:40:20 2019

@author: PaulGraham
"""
import numpy as np
class StumpClass:
    def __init__(self, data, labels, weights , midPoint):
        self.data = data #Data in R^[dim] and with a number of size "dataSize"
        self.labels = labels #The labels are either -1 or 1 and has size "dataSize"
        self.weights = weights #The weights are 
        self.components = []
        self.dataSize , self.dim = data.shape
        self.params = [0,True,0]#The first component is the index, the second is true if reater than or equal to and the third component is the ineq condition
    def predict(self, data):
        final_loss = -1
        for index in range(self.dim):
            final_pred = []
            data_s = np.sort(data[:,index])#sort the data and looks at only the component labeled by "indez"
            if midPoint == True:
                data_s = 0.5 * (data_s[0:self.dataSize-1] + data_s[1:self.dataSize])
            for check in data_s:#Runs over all of the pieces of data
                temp_pred = 2 * (data_s >= check) - 1#This is a labeling with 1 if componenet is greater than datum 
                temp_loss = (1/self.dataSize)*np.sum(np.multiply(self.weights , np.multiply(temp_pred , self.labels)))#This will add one 
                #if label is accurate and subtract one if label is wrong
                if temp_loss >= final_loss:
                    final_loss = temp_loss
                    final_pred = temp_pred
                    self.params = [index , True , check]
                if -1 * temp_loss >= final_loss:
                    final_loss = -1 * temp_loss
                    final_pred = -1 * temp_pred
                    self.params = [index , False , check]
        return final_pred
    def getParams(self):
        return self.params
    
