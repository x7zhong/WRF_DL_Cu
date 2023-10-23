#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:40:53 2023

@author: xiaohui

this script is used to analyze whether a grid point is triggered for convection
after implementing derive_trigger
"""

import numpy as np

from config_wrf import config_wrf




if __name__ == "__main__":
    
    dict_all = {}
    dict_all['train'] = np.load(config_wrf.pathName_train_dict, allow_pickle = True).item()    
    dict_all['test'] = np.load(config_wrf.pathName_test_dict, allow_pickle = True).item()
    
    for flag in ['train', 'test']:
        
        for fileName in dict_all[flag].values():        

            print('Read data from {}'.format(fileName))
            
            

