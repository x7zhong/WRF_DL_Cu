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

#%%
#WRF configuration

#time_length specifies the number of samples per file
config_wrf.time_length = 44000

#Number of vertical layers in WRF data, i.e. number of vertical levels - 1
config_wrf.vertical_layers = 44

config_wrf.dims = [220, 200]

#cudt is 5 minutes
config_wrf.cudt = 5 * 60 

#dt is 15 seconds
config_wrf.dt = 15

STEPCU = config_wrf.cudt/config_wrf.dt
config_wrf.TST = STEPCU * 2

#WRF output information
config_wrf.output_interval = 5 #unit is minutes
config_wrf.total_hours = 36 #unit is hours
config_wrf.train_hours = 24
config_wrf.test_hours = 12
   
#Number of simulations
config_wrf.num_sim = 9
# config_wrf.num_sim = 5

#Number of files per simulation for training
config_wrf.num_train = int((config_wrf.train_hours * 60)/config_wrf.output_interval)
#Number of files per simulation for testing
config_wrf.num_test = int((config_wrf.test_hours * 60)/config_wrf.output_interval)

#%%

#flag_constraint specifies whether add constraint when training model 
config_wrf.flag_constraint = 1

# config_wrf.single_height_variable = ['xland']
config_wrf.single_height_variable = ['pratec', 'nca', 'hfx', 'ust', 'pblh'] #zol is removed as it is always 0
# config_wrf.single_height_variable = ['pratec', 'nca', 'zol', 'hfx', 'ust', 'pblh']
# config_wrf.single_height_variable = ['pratec', 'zol', 'hfx', 'ust', 'pblh']
# config_wrf.single_height_variable = ['raincv', 'pratec', 'nca', 'cu_act_flag', \
#                                      'zol', 'hfx', 'ust', 'pblh']

#xland here does not change with time, so we only  output it once in a additional file.
config_wrf.additional_variable = ['xland']

#note that w is level variable, equal to one plus the number of vertical_layers
config_wrf.multi_height_variable = ['u', 'v', 'w', 't', 'q', 'p', 'th', 'dz8w', \
                                    'rho', 'pi', 'w0avg', \
                                    'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', \
                                    'rqicuten', 'rqscuten']      

# config_wrf.multi_height_cumsum_variable = {'qc': 1}
config_wrf.multi_height_cumsum_variable = {}

### pressure of all the vertical levels ###
###5000 is model top pressure, not included
config_wrf.auxiliary_variable = []
# config_wrf.auxiliary_variable = ['trigger', 'w0avg_output']
config_wrf.auxiliary_variable = ['trigger', 'w0avg_output', 'p_diff', 'qv_sat', 'rh']
# config_wrf.auxiliary_variable = ['trigger', 'w0avg_output', 'p_diff', 'qv_sat', 'rh', 'height', 'cape', 'cin']
# config_wrf.auxiliary_variable = ['p_diff', 'qv_sat', 'rh', 'height', 'cape', 'cin']

# config_wrf.constant_variable_list = ['dx', 'ptop']
config_wrf.constant_variable = {'ptop': 5000}

#num_orig_feature specifies the number of original variables used as features
config_wrf.num_orig_feature = len(config_wrf.single_height_variable) + \
                              len(config_wrf.multi_height_variable)
               
#num_all_feature specifies the total number of variables used as features, including preprocessed variables.                                
config_wrf.num_all_feature = config_wrf.num_orig_feature + \
                             len(config_wrf.auxiliary_variable)

config_wrf.feature_all_variable = config_wrf.single_height_variable + \
                                  config_wrf.multi_height_variable + \
                                  config_wrf.auxiliary_variable 

#Here, dtt, dqvdt, and pratec are the most important 
#dqc and dqi are of the secondary importance
#dqr and dqs are of the last importance
config_wrf.label_single_height_variable = []
config_wrf.label_multi_height_variable = []
config_wrf.label_single_height_variable_possible = ['trigger', 'nca', 'pratec', 'hbot', 'htop']
# config_wrf.label_single_height_variable = ['nca']
config_wrf.label_single_height_variable = ['trigger', 'nca', 'pratec']
config_wrf.label_multi_height_variable = ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']
# config_wrf.label_multi_height_variable = ['rthcuten']
config_wrf.label_all_possible_variable = ['trigger', 'nca', 'pratec', 'hbot', 'htop', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']
config_wrf.label_classification_variable_possible = ['trigger']

#classification label must be the first varaible in label_all_variable
config_wrf.label_all_variable = config_wrf.label_single_height_variable + \
                                config_wrf.label_multi_height_variable

#flag_trigger_consistency specifies whether apply consistency bewteen trigger 
#and all the other target when training model 
config_wrf.flag_trigger_consistency = 0
config_wrf.label_all_variable_reg = config_wrf.label_all_variable.copy()
config_wrf.output_channel_cf = 0
if 'trigger' in config_wrf.label_all_variable:
    config_wrf.flag_trigger_consistency = 1
    config_wrf.label_all_variable_reg.remove('trigger')

    config_wrf.output_channel_cf = len(config_wrf.label_classification_variable_possible)

config_wrf.weight_loss = {}
for varName in config_wrf.label_all_possible_variable:
    config_wrf.weight_loss[varName] = 1

    config_wrf.index_trigger_input = config_wrf.feature_all_variable.index('trigger')        
    
#output_channel specifies the number of output channels
config_wrf.output_channel = len(config_wrf.label_all_variable_reg)
config_wrf.pathName_additional = '/home/data2/RUN/DAMO/train/mskf_xland.npz'

# config_wrf.lat = 
# config_wrf.lon = 

config_wrf.dirName_data = '/home/data2/RUN/DAMO/train'       

config_wrf.error_train = ['mse']
config_wrf.num_class = 2

config_wrf.Listof_error = ['rmse', 'mbe', 'mae']
# config_wrf.Listof_error = ['rmse', 'mbe', 'mae', 'cc']
# config_wrf.error_train = ['mse_trigger']
if 'trigger' in config_wrf.label_all_variable:
    config_wrf.Listof_error_class = ['BCE', 'accuracy']        
    
    if len(config_wrf.label_all_variable) == 1:
        config_wrf.error_train = ['BCE']        
        config_wrf.num_class = 2

        # config_wrf.error_train = 'CrossEntropyLoss'    
        # config_wrf.num_class = 3
                
    else:
        config_wrf.error_train = ['BCE' , 'mse']    
        # config_wrf.error_train = ['BCE' , 'mse_trigger']            
        

#filter specify whether filter variables based on the values of trigger
config_wrf.flag_filter = 0
# config_wrf.flag_filter = 1
    
config_wrf.filter_variable = 'trigger'
config_wrf.filter_threshold = 0.001

if len(config_wrf.label_all_variable) > 1:
    # config_wrf.Listof_error.append( 'accuracy_trigger' )
    # config_wrf.Listof_error.append( 'mse_trigger' )
    config_wrf.Listof_error.append( 'rmse_trigger' )
    config_wrf.Listof_error.append( 'mbe_trigger' )
    config_wrf.Listof_error.append( 'mae_trigger' )
    # config_wrf.Listof_error.append( 'cc_trigger' )
    
# config_wrf.norm_method = 'min-max'
config_wrf.norm_method = 'abs-max'
# config_wrf.norm_method = 'z-score'

config_wrf.if_plot = False
# config_wrf.if_plot = True

# config_wrf.if_plot_2d = False
config_wrf.if_plot_2d = True
config_wrf.plot_variable = config_wrf.label_all_possible_variable

config_wrf.pathName_norm = '/home/yux/Code/Python/AI_and_Cu/utils/pathName_{}.npy'.format(config_wrf.norm_method)   
if config_wrf.norm_method == 'abs-max':
    config_wrf.pathName_norm = '/home/yux/Code/Python/AI_and_Cu/utils/pathName_min-max.npy'
    
config_wrf.pathName_train_dict_in = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict_in.npy'   
config_wrf.pathName_train_dict_out = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict_out.npy'   
config_wrf.pathName_test_dict_in = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_in.npy'
config_wrf.pathName_test_dict_out = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_out.npy'
# config_wrf.num_sim = 1

#For debugging code
# config_wrf.pathName_train_dict_in = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict_in_test.npy'   
# config_wrf.pathName_train_dict_out = '/home/yux/Code/Python/AI_and_Cu/utils/train_dict_out_test.npy'   
# config_wrf.pathName_test_dict_in = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_in_test.npy'
# config_wrf.pathName_test_dict_out = '/home/yux/Code/Python/AI_and_Cu/utils/test_dict_out_test.npy'
# config_wrf.num_train = 2
# config_wrf.num_test = 1
# #Number of simulations
# config_wrf.num_sim = 1

# print(config_wrf)    