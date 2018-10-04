#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:58:34 2018

@author: eimearnolan
"""
# This function takes in a dataset 'training_ds' with features as columns
#    and takes in the target variable 'target'; 

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pickle

from sklearn.metrics import mean_squared_error
from math import sqrt




def train_generation_model(weather_generation, house_to_keep):

    """
This function takes in:

 - a training dataset 'weather_generation' with timestamp;'time', 'cloudcover', 'house' and generation;'gen_sum' as columns
 - the house number of the house you wish to model
 - list of houses you want to model
 
 It trains a linear regression model with features hour of the day,
 month as a numeric and cloud cover as a percentage.
 Works using k-fold validation and plot trinaing and test error for those k folds
 
 """    
    
    # training function, house number and time 
    rms_train = []
    rms_test = []
    
    for house in house_to_keep:
        print('Training linear regression model for house #'+str(house)+ '...' )

        wtest = weather_generation.loc[weather_generation['house'] == house]
        data= wtest[['time', 'gen_sum', 'cloudcover']]
        
        data["hour"] = data["time"].dt.hour # to capture day and night 
        data["month"] = data["time"].dt.month  # to capture seasonality
        
        #creating the train and validation set, split on past and future data
        # in future use timeseriessplit()
        train_size = int(len(data) * 0.7)
        train, test = data[0:train_size], data[train_size:len(data)]
        
        #train = target[:int(0.7*(len(target)))]
        #test = target[int(0.3*(len(target))):]
        
        # Define the model by explicitely saying that our data contains an intercept term
        reg = LinearRegression(fit_intercept=False)
        
        # Fit the data
        reg.fit(train[['cloudcover','hour', 'month']], train['gen_sum'])
        
        # Evaluate the model on the given for training
        y_pred_train = reg.predict(train[['cloudcover','hour', 'month']])
        
        # Evaluate the model on the givenfor training
        y_pred_test= reg.predict(test[['cloudcover','hour', 'month']])
        
        # Plot comparison test data
        plt.figure()
        plt.plot( 'time', 'gen_sum', data=train)
        plt.plot( train.time, y_pred_train, color='red', linewidth=2)
        plt.title('Actual vs Predicted Generation - Train Data', fontsize=16)
        plt.xlabel('Time (datetime stamp)', fontsize=14)
        plt.ylabel('Magnitide (MW)', fontsize=14)
        plt.legend('Actual Gen', 'Pred')
        
        
        # Plot comparison test data
        plt.figure()
        plt.plot( 'time', 'gen_sum', data=test)
        plt.plot( test.time, y_pred_test, color='red', linewidth=2)
        plt.title('Actual vs Predicted Generation -Test Data', fontsize=16)
        plt.xlabel('Time (datetime stamp)', fontsize=14)
        plt.ylabel('Magnitide (MW)', fontsize=14)
        plt.legend('Actual Gen', 'Pred')
        
        # evaluation 
        
        # training/test error
        error_train = train.gen_sum - y_pred_train
        error_test = test.gen_sum - y_pred_test
        plt.figure()
        plt.plot(train.time, error_train, color='green', linewidth=2)
        plt.plot(test.time, error_test, color='blue', linewidth=2)
        plt.title('Actual vs Predicted Generation Error', fontsize=16)
        plt.xlabel('Time (datetime stamp)', fontsize=14)
        plt.ylabel('Generation Error', fontsize=14)
        plt.legend('Training Error', 'Test Error')
        
        # root mean sqaured error
        rms_train.append(sqrt(mean_squared_error(train.gen_sum , y_pred_train)))
        rms_test.append(sqrt(mean_squared_error(test.gen_sum , y_pred_test))) 
        
        # saving model
        pkl_filename = "pickle_model_lm"+str(house)+".pkl"  
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(reg, file)
        
    # rms across houses
    plt.figure()
    
    plt.plot(house_to_keep, rms_train, color='green', linewidth=2)
    plt.plot(house_to_keep, rms_test, color='blue', linewidth=2)
    plt.title('RMS error per house', fontsize=16)
    plt.xlabel('House #', fontsize=14)
    plt.ylabel('RMS Error', fontsize=14)
    plt.legend('Training Error', 'Test Error')
    
    return print('models created and saved')
