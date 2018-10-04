#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 11:35:21 2018

@author: eimearnolan
"""
''' 

creating  create, predict and evaluate functions for  linear models
testing data 
        
'''

import pandas as pd

# calling functions
from train_generation_model import train_generation_model
from predict_generation import predict_generation
from evaluate_model import evaluate_model

# generation data 
weather_generation = pd.read_pickle('weather_generation_data.pkl')

# testing model generation function ...
house_to_keep = ['0','1','2','3', '4','5', '6', '7','8']

# trains linear models for all remaining houses
train_generation_model(weather_generation, house_to_keep)

# small test on function
timestamp =weather_generation.time[10]  # has to be datetime format !!!
timestamp = pd.to_datetime(timestamp, errors='coerce')
house = 1

# this assumes a function weather!! has dummy variables now ...
pred = predict_generation(timestamp, house)

# dummy test value
actual = 1300
error = evaluate_model(actual,pred)
