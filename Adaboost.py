# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 02:16:32 2019

@author: PaulGraham
"""

import numpy as np
import matplotlib.pyplot as plt


import Stump
from Stump import StumpClass

#from TestFunction import funct2d
def funct2D(Data):
    return (2*( (-1*np.abs(np.sin(Data[:,0])) + np.exp(Data[:,0]*Data[:,1]) - Data[:,1]**2) >= 0.5 )-1)

def adaboost(data, labels, n_classifiers):
    classifiers = []
    classifier_weights = []
    N = np.size(labels)
    weights = (1/N) * np.ones(N)
    for t in range(n_classifiers):
        stumpObj = StumpClass(data , labels , weights)
        predictions = stumpObj.predict(data)
        z = np.sum(np.multiply(weights , np.multiply(predictions , labels)))
        alpha = 0.5 * np.log( (1+z) / (1-z) )
        weights = weights * np.exp(-1 * alpha * np.multiply(labels,predictions))
        weights = 1/np.sum(weights) * weights
        classifiers.append(stumpObj)
        classifier_weights.append(alpha)
    return classifiers, classifier_weights

def adaboostPredictions(classifiers , classifier_weights , data, labels, n_classifiers, dataSize):
    Predictions = np.zeros(dataSize)
    Sum = 0
    for i in range(n_classifiers):
        Stump = classifiers[i]
        Sum += classifier_weights[i] * Stump.predict(data)
    for j in range(dataSize):
        if Sum[j] > 0:
            Predictions[j] = 1
        else:
            Predictions[j] = -1
    return Predictions
    
#def dataCreation(dataSize , dim):
#    Labels = np.zeros(dataSize)
#    Data = np.random.rand(dataSize,dim)
#    for i in range(dataSize):
#        condition = -1*np.abs(np.sin(Data[i][0])) + np.exp(Data[i][0]*Data[i][1]) - Data[i][1]**2
#        if condition >= 0.5:
#            Labels[i] = 1
#        else:Labels[i] = -1
#    return Data , Labels

def dataCreation(dataSize , dim):
    Data = np.random.rand(dataSize,dim)
    Labels = funct2D(Data)
    return Data , Labels

def pointPlotter(Data , Labels , dataSize):
    positivePoints = []
    negativePoints = []
    for i in range(dataSize):
        if Labels[i] == 1:
            positivePoints.append(Data[i])
        else:
            negativePoints.append(Data[i])
    positivePoints = np.array(positivePoints)
    negativePoints = np.array(negativePoints)
    
    positiveX = np.transpose(positivePoints)[0]
    positiveY = np.transpose(positivePoints)[1]
    negativeX = np.transpose(negativePoints)[0]
    negativeY = np.transpose(negativePoints)[1]
    plt.figure()
    plt.scatter(positiveX , positiveY, c = 'r', s = 2500, marker = 'X')
    plt.scatter(negativeX , negativeY, c = 'b', s = 2500, marker = 'P')

def inequalityPlotter(classifiers, classifier_weights, n_classifiers):
    plt.figure()
    plt.hsv()
    for i in range(n_classifiers):
        points = []
        Stump = classifiers[i]
        [index, geqTrue, condition] = Stump.getParams()
        print(condition)
        if index == 0:
            pointsX_1 = [1,1,condition,condition]
            pointsY_1 = [0,1,1,0]
            pointsX_2 = [0,0,condition,condition]
            pointsY_2 = [0,1,1,0]
            p
        else:
            pointsX_1 = [0,1,1,0]
            pointsY_1 = [0,0,condition,condition]
            pointsX_2 = [0,1,1,0]
            pointsY_2 = [1,1,condition,condition]
        colors_1 = [0 , (1-classifier_weights[i]) , 0.1]
        colors_2 = [0.4 , (1-classifier_weights[i]) , 0.1]
        if geqTrue:    
            plt.fill(pointsX_1 , pointsY_1 , colors_1 , pointsX_2 , pointsY_2 , colors_2)
        else:
            plt.fill(pointsX_1 , pointsY_1 , colors_2 , pointsX_2 , pointsY_2 , colors_1)
        
dataSize = 10
n_classifiers = 4
Data , Labels = dataCreation(dataSize,2)
print(Data)
print("apple")
classifiers, classifier_weights = adaboost(Data, Labels, n_classifiers)
Predictions = adaboostPredictions(classifiers , classifier_weights , Data, Labels, n_classifiers, dataSize)
pointPlotter(Data , Labels , dataSize)
pointPlotter(Data , Predictions , dataSize)
inequalityPlotter(classifiers, classifier_weights, n_classifiers)
    
        