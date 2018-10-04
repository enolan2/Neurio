#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 16:58:34 2018

@author: eimearnolan
"""
# Evaulates prediction error 


def evaluate_model(y_pred, target):
    
    # with 30% test data 
    
    """
        This function takes in:pred , actual
         
     """    
    # simple error (use RMS for array)
    error = target- y_pred

    return error
    