#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 16:55:03 2023

@author: xiaohui
"""

import os

import numpy as np

#%%
dirName_data_gt = '/home/data2/RUN/DAMO/train'       
dirName_data_model = '/home/data2/RUN/DAMO/model_outputs/LSTMClassifier_Reg_32_3/all_abs-max_mae_trigger_specified'       
dirName_data = '/home/data2/RUN/DAMO/xiaohui'

pathName_test_dict_in = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_in.npy'
pathName_test_dict_out = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_out.npy'

def Save_var_all(varName):
        
    np_source_out = np.load(pathName_test_dict_out, allow_pickle = True).item()

    pathName = os.path.join(dirName_data, varName + '_gt_all.npy')
    if os.path.isfile(pathName) == 0:
    
        ### Save targets    
        Var_all = []    
        for t in range(len(np_source_out)):
        
            fileName_out = np_source_out[t]
            
            #Get initialization time and forecast time
            time_str = {}
            temp = fileName_out.split('/')
            time_str['initTime_str'] = temp[-2]
            time_str['forecastTime_str'] = temp[-1][0:15]
            
            print('Load gt {} data from {}\n'.format(varName, fileName_out))
            target = np.load(fileName_out)
            
            Var_all = Var_all + list(target[varName])                  
            
        Var_all = np.array(Var_all)
        
        np.save(pathName, Var_all)
        
        print('{} is saved\n'.format(pathName))

    ### Save model outputs
    pathName = os.path.join(dirName_data, varName + '_model_all.npy')
    if os.path.isfile(pathName) == 0:
        
        Var_all = []
        for t in range(len(np_source_out)):
        
            fileName_out = np_source_out[t]
            
            #Get initialization time and forecast time
            time_str = {}
            temp = fileName_out.split('/')
            time_str['initTime_str'] = temp[-2]
            time_str['forecastTime_str'] = temp[-1][0:15]
            
            pathName_model = os.path.join(dirName_data_model, time_str['initTime_str'] , temp[-1].replace('_output', ''))
    
            print('Load model {} data from {}\n'.format(varName, pathName_model))        
            model_outputs = np.load(pathName_model)
            
            Var_all = Var_all + list(model_outputs[varName])
            
        Var_all = np.array(Var_all)
        
        np.save(pathName, Var_all)
            
#%%
        
ListofVar = ['trigger', 'nca', 'pratec', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']
ListofVar = ['nca', 'pratec', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']    
        
for varName in ListofVar:
    
    Save_var_all(varName)
                    
