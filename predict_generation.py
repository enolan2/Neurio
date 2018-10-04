#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:58:34 2018

@author: eimearnolan
"""
# This function takes in a dataset 'training_ds' with features as columns
#    and takes in the target variable 'target'; 

# weather function that takes in a timestamp 
#from weather import weather
import pickle
import pandas as pd
import datetime as dt
import pandas as pd
from train_generation_model import train_generation_model


def predict_generation(time, house_num):
    
    """
        This function takes in:
        
         -  timestamp (datetime format)
         - the house number of the house you wish to predict solar generation
         
         It imports the regression model for the specific house number 
         and predicts and returns generation         
     """    
    
    print('Predicting generation for house #'+str(house_num))


    hour = time.hour # to capture day and night 
    month = time.month  # to capture seasonality 

    # area and house number dataframe
    locations = [(0,2),
             (1,1),
             (2,2),
             (3,3),
             (4,1),
             (5,0),
             (6,0),
             (7,1),
             (8,4)]
        
    labels = ["House", "Location"]
    house_locations = pd.DataFrame.from_records(locations, columns=labels)
    
    # getting location of house  for weather function
    location_row = house_locations.loc[house_locations['House'] == house_num]
    location = location_row.iloc[0]['Location']

    # get weather ; returns dictionary of weather 
    #weather_hr = weather(location, hour)
    # dummy variable
    cloudcover =98 # weather_hr['cloudiness']
    
    # import linear regresion
    # load the model from disk
    pkl_filename = "pickle_model_lm"+str(house_num)+".pkl"  
    lr_model = pickle.load(open(pkl_filename, 'rb'))    
    
    # predict
    #y_pred_test= reg.predict(test[['cloudcover','hour', 'month']])
    pred = lr_model.predict([[cloudcover, hour, month]])  

    #return trained model
    return pred










