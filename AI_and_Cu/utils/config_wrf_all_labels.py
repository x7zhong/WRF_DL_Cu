#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 10:50:45 2023

@author: xiaohui
"""

import numpy as np
# import torch 

class Configs:
    def __init__(self):
        pass
    
config_wrf = Configs()

#Number of files per simulation for training
config_wrf.num_train = 48

#Number of files per simulation for testing
config_wrf.num_test = 24

#Number of simulations
config_wrf.num_sim = 1

#time_length specifies the number of samples per file
config_wrf.time_length = 44000

#Number of vertical layers in WRF data, i.e. number of vertical levels - 1
config_wrf.vertical_layers = 44

#Variable mapping
config_wrf.extract_mapping = {'xland': 'xland',
                           'u': 'u',
                           'v': 'v',
                           'w': 'w',
                           't': 't',
                           'q': 'q',
                           'p': 'p',                                
                           'rho': 'rho',
                           'pi': 'pi',
                           'th': 'th',
                           'dz8w': 'dz8w',
                           'dtt': 'dtt',
                           'dqv': 'dqv',
                           'dqc': 'dqc',
                           'dqr': 'dqr',
                           'dqi': 'dqi',
                           'dqs': 'dqs',
                           'pratec': 'pratec'
                        }

config_wrf.feature_channel = 10

# config_wrf.single_height_variable = ['xland']
config_wrf.single_height_variable = []
#xland here does not change with time, so we only  output it once in a additional file.
config_wrf.additional_variable = ['xland']

config_wrf.multi_height_variable = ['u', 'v', 'w', 't', 'q', 'p', 'rho', \
                                    'pi', 'th', 'dz8w']

# config_wrf.multi_height_cumsum_variable = {'qc': 1}
config_wrf.multi_height_cumsum_variable = {}

### pressure of all the vertical levels ###
###5000 is model top pressure, not included
# config_wrf.auxiliary_variable = ['p']
config_wrf.auxiliary_variable = []

# config_wrf.constant_variable_list = ['dx', 'ptop']
config_wrf.constant_variable_list = ['ptop']

config_wrf.gpu = "cuda:1"
config_wrf.num_gpu = 1
# config_wrf.gpu = "cuda:0,1,2,3"
# config_wrf.num_gpu = 4

#num_orig_feature specifies the number of original variables used as features
config_wrf.num_orig_feature = len(config_wrf.single_height_variable) + \
                              len(config_wrf.multi_height_variable)
               
#num_all_feature specifies the total number of variables used as features, including preprocessed variables.                                
config_wrf.num_all_feature = config_wrf.num_orig_feature + \
                             len(config_wrf.multi_height_cumsum_variable) * 2

#Here, dtdt, dqvdt, and pratec are the most important 
#dqcdt and dqidt are of the secondary importance
#dqrdt and dqsdt are of the last importance
config_wrf.label_single_height_variable = ['trigger', 'pratec']
config_wrf.label_multi_height_variable = ['dtt', 'dqv', 'dqc', 'dqr', 'dqi', 'dqs']

config_wrf.label_all_variable = config_wrf.label_single_height_variable + \
                                config_wrf.label_multi_height_variable

config_wrf.weight_loss = {}
for varName in config_wrf.label_all_variable:
    config_wrf.weight_loss[varName] = 1
    
#output_channel specifies the number of output channels
config_wrf.output_channel = len(config_wrf.label_single_height_variable) + \
                            len(config_wrf.label_multi_height_variable)

# config_wrf.device = torch.device('cuda:0')
# config_wrf.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config_wrf.pathName_additional = '/home/data2/RUN/DAMO/train/mskf_xland.npz'

config_wrf.dirName_data = '/home/data2/RUN/DAMO/train'       

config_wrf.error_train = 'mse'
config_wrf.Listof_error = ['rmse', 'mbe', 'mae']

config_wrf.if_plot = False
# config_wrf.if_plot = True

config_wrf.pathName_train_dict = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict.npy'   
config_wrf.pathName_test_dict = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict.npy'
config_wrf.num_train = 48
config_wrf.num_test = 24

# config_wrf.pathName_train_dict = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict_test.npy'   
# config_wrf.pathName_test_dict = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_test.npy'
# config_wrf.num_train = 1
# config_wrf.num_test = 1
    