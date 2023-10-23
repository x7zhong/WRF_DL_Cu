#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:53:23 2023

@author: xiaohui
"""

import os
import numpy as np

# from config_wrf import config_wrf
from utils.config_wrf import config_wrf

# ListofVar = ['xland', 'u', 'v', 'w', 't', 'q', 'p', 'rho', 'pi', 'th', 'dz8w', \
#              'dtt', 'dqv', 'dqc', 'dqr', 'dqi', 'dqs', 'pratec']
  
ListofVar = config_wrf.feature_all_variable + config_wrf.label_all_variable 
        
norm_mapping_standard = {}
norm_mapping = {}
for varName in ListofVar:
    norm_mapping_standard[varName]=  {'mean': 0.0, 'scale': 1.0, 'max': 1.0, 'min': 0.0}
                                     
norm_mapping = {}

if os.path.isfile(config_wrf.pathName_norm) != 0:
    norm_mapping = np.load(config_wrf.pathName_norm, allow_pickle = True).item()
else:
    print('{} does not exist\n'.format(config_wrf.pathName_norm))

norm_mapping['trigger'] = {'mean': 0.0, 'scale': 1.0, 'max': 1.0, 'min': 0.0, \
                           'diff_max': 1, 'diff_min': -1}
    
if 'w0avg' in norm_mapping.keys():
    norm_mapping['w0avg_output'] = norm_mapping['w0avg']
        
# norm_mapping['htop'] = {'mean': 0.0, 'scale': 1.0, 'max': 1.0, 'min': 0.0, \
#                        'diff_max': config_wrf.vertical_layers, 'diff_min': -config_wrf.vertical_layers}    
# norm_mapping['hbot'] = norm_mapping['htop'].copy()
    
for varName in norm_mapping.keys():
    if ('max' in norm_mapping[varName].keys()) & ('diff_max' not in norm_mapping[varName].keys()):
        norm_mapping[varName]['diff_max'] = np.max((np.abs(norm_mapping[varName]['max']), \
                                                   np.abs(norm_mapping[varName]['min'])))
            
        norm_mapping[varName]['diff_min'] = -norm_mapping[varName]['diff_max']
        
print(norm_mapping)