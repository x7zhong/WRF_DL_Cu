#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 17:45:39 2023

@author: xiaohui

This function is used to derive whether a grid point is triggered for convection
based on the values of output variables
"""

import numpy as np

from config_wrf import config_wrf

def derive_trigger(var):
    trigger = np.zeros(var.shape[0])
    
    temp = np.sum(np.abs(var),axis=1)
    
    trigger[ np.where(temp!=0)[0] ] = 1
    
    return trigger

if __name__ == "__main__":
    
    dict_all = {}
    dict_all['train'] = np.load(config_wrf.pathName_train_dict, allow_pickle = True).item()    
    dict_all['test'] = np.load(config_wrf.pathName_test_dict, allow_pickle = True).item()
    
    # for flag in ['train', 'test']:
        # for fileName in dict_all[flag].values():        
    for flag in ['test']:        
        for fileName in dict_all[flag].values():        

            print('Read data from {}'.format(fileName))
            np_source = dict(np.load(fileName))
            
            if 'trigger' not in np_source:
                                    
                dtt = np_source['dtt']                    
    
                trigger = derive_trigger(dtt)
                
                np_source['trigger'] = trigger
                                
                np.savez(fileName, **np_source)    
                
                print('{} is saved\n'.format(fileName))    
            
    
    