# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 23:49:12 2019
@author: PaulGraham
"""
import numpy as np


def funct2D(Data):
    return (2*( (-1*np.abs(np.sin(Data[:,0])) + np.exp(Data[:,0]*Data[:,1]) - Data[:,1]**2) >= 0.5 )-1)