#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:05:46 2023

@author: xiaohui
"""

import numpy as np
import torch

#Apply this before unnormalize
#for the grid points where the input nca is greater than 0, the nca and tendency is not changed.
def mskf_constraint(feature, predict, ListofVar, config_wrf):
    
    trigger_input = feature[:, config_wrf.index_trigger_input, 0]
    
    index = torch.where(trigger_input > 0.0001)[0]
    
    #This ensure pratec to be non negative.    
    for i, varName in enumerate(ListofVar):        
        
        if varName  == 'pratec':
            
            if len(predict.shape) == 2:                
                predict[:, i][predict[:, i] < 0] = 0
                
            elif len(predict.shape) == 3:
                predict[:, i, :][predict[:, i, :] < 0] = 0            
    

    if len(index) > 0:
    
        for i, varName in enumerate(ListofVar):        
            if varName  == 'trigger':
                if len(predict.shape) == 2:
                    predict[index, i] = trigger_input[index]
                    
                elif len(predict.shape) == 3:
                    predict[index, i, 0] = trigger_input[index]
                      
            elif varName  == 'pratec':
                index_feature = config_wrf.feature_all_variable.index(varName)
                
                if len(predict.shape) == 2:
                    predict[index, i] = feature[index, index_feature]     
                                        
                elif len(predict.shape) == 3:
                    for level in range(predict.shape[2]):                    
                        predict[index, i, level] = feature[index, index_feature, level]     
                                        
            else:
                index_feature = config_wrf.feature_all_variable.index(varName)
                
                for level in range(predict.shape[2]):
                    predict[index, i, level] = feature[index, index_feature, level]
        
    return predict
    
#Apply this before unnormalize
#for the grid points where the predicted trigger is 0, all the other output is 0
def trigger_consistency(predict, predicts_cf, config_wrf):
    
    trigger_predict = predicts_cf[:, 0] > 0.5
    
    #For those grid points where it is predicted as not triggered, make sure they are 0.
    index = torch.where(trigger_predict < 0.0001)[0]

    for i, varName in enumerate(config_wrf.label_all_variable_reg):
        if len(predict.shape) == 2:
            predict[index, i] = 0
            
        elif len(predict.shape) == 3:
            for level in range(predict.shape[2]):
                predict[index, i, level] = 0            

    return predict

#Apply this after unnormalize
def nca_constraint(predicts_unnorm, ListofVar):
    
    index_nca = ListofVar.index('nca')
    
    #nca must be equal to integer multiples of time step
    predicts_unnorm[:, index_nca, :][predicts_unnorm[:, index_nca, :]<0.01] = 0.0
    predicts_unnorm[:, index_nca, :] = predicts_unnorm[:, index_nca, :].to(torch.int).to(torch.float32)

    return predicts_unnorm

#saturation vapor pressure calculation
#Input unit is K, output unit is Pa
def calculate_p_saturated(t, option = 1):  
    
    # Saturation vapor pressure (ES) is calculated following Buck (1981)
    # New equations for computing vapor pressure and enhancement factor, J. Appl. Meteorol., 20, 1527-1532, 1981)
    
    #The constants below are from module_model_constants.F of WRFV4.3
    SVP1=0.6112 # constant for saturation vapor pressure calculation (dimensionless)
    SVP2=17.67  # constant for saturation vapor pressure calculation (dimensionless)
    SVP3=29.65  # constant for saturation vapor pressure calculation (K)
    SVPT0=273.15 #constant for saturation vapor pressure calculation (K)
    
    aliq = SVP1*1000.
    bliq = SVP2
    cliq = SVP2*SVPT0
    dliq = SVP3
    
    if option == 1:
        p_sat = aliq*np.exp((bliq*t - cliq)/(t - dliq))
        # p_sat = 6.112*np.exp(17.67*(t-273.16)/(243.5+t-273.16)) * 100
        
    elif option == 2: 
        p_sat = 6.112*np.exp(21.87*(t-273.16)/(t-7.66)) * 100
        
    return p_sat

           # if(T0(NK).LE.273.16) then
           #   envEsat = 6.112*exp(21.87*(T0(NK)-273.16)/(T0(NK)-7.66))
           # else
           #   envEsat = 6.112*exp(17.67*(T0(NK)-273.16)/(243.5+T0(NK)-273.16))
           # end if

#Input unit is Pa, output unit is kg/kg
def calculate_qv_saturated(p, t):
            
    p_sat = calculate_p_saturated(t)
    
    qv_sat = 0.622*p_sat/(p-p_sat)    
    
    return qv_sat

def calculate_raincv(pratec, dt):
    #dt unit has to be seconds
    raincv = pratec * dt
    
    return raincv
    
def calculate_rh(qv, qv_sat):
    q = np.minimum(qv, qv_sat)
    
    q = np.maximum(0.000001, q)
    
    rh = q/qv_sat
    
    return rh
    