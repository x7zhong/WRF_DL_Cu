#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Sun Mar 19 12:43:16 2023

@author: xiaohui
'''

import os
import numpy as np
import torch
import sys
import wrf

sys.path.append('..')
from torch.utils.data import Dataset, DataLoader

from utils.config_wrf import config_wrf
from utils.config_norm import norm_mapping
from utils.cu_utils import calculate_qv_saturated, calculate_rh
# from config_wrf import config_wrf
# from config_norm import norm_mapping
# from cu_utils import calculate_qv_saturated, calculate_rh

train_dict = np.load(config_wrf.pathName_train_dict, allow_pickle = True).item()

test_dict = np.load(config_wrf.pathName_test_dict, allow_pickle = True).item()

def unnormalized_feature(temp_value, varName, norm_method, norm_mapping):
    if norm_method == 'z-score':
        temp_value_new = temp_value * norm_mapping[varName]['scale'] + \
        norm_mapping[varName]['mean']  
        
        
    elif norm_method == 'min-max':
        temp_value_new = temp_value * norm_mapping[varName]['max']
          
    return temp_value_new
        
class WrfCuDataset(Dataset):
    def __init__(self, vertical_layers, type, norm_mapping={}, time_length = 44000, \
                 flag_get_stat = 0, evaluate = "False", flag_filter = 0):
        self.time_length = time_length
        self.type = type
        self.flag_get_stat = flag_get_stat
        self.evaluate = evaluate
        self.flag_filter = flag_filter        
        
        self.norm_mapping = norm_mapping

        self.vertical_layers = vertical_layers

        self.label_multi_feature = np.zeros(
            [time_length, len(config_wrf.label_multi_height_variable), self.vertical_layers])

        self.label_single_feature = np.zeros(
            [time_length, len(config_wrf.label_single_height_variable), self.vertical_layers])

        self.single_feature = np.zeros(
            [time_length, len(config_wrf.single_height_variable), self.vertical_layers])

        self.multi_feature = np.zeros(
            [time_length, len(config_wrf.multi_height_variable), self.vertical_layers])

        self.multi_cumsum_feature = np.zeros(
            [time_length, 2*len(config_wrf.multi_height_cumsum_variable), self.vertical_layers])

        self.auxiliary_feature = np.zeros(
            [time_length, len(config_wrf.auxiliary_variable), self.vertical_layers])

        self.filter_feature = np.ones(
            [time_length, 1, self.vertical_layers])
        
        # self.constant_value = {}
        # for variable in config_wrf.constant_variable_list:
        #     self.constant_value[variable] = self.norm_mapping[variable]["mean"]
                                           
        if(self.type == 'train'):
            self.index_dict = train_dict
        elif(self.type == 'test'):
            self.index_dict = test_dict
    
    def __len__(self):
        if(self.type == 'train'):
            #Number of files for training: number of files per simulation * number of simulations
            return config_wrf.num_train * config_wrf.num_sim
        elif(self.type == 'test'):
            #Number of files for testing: number of files per simulation * number of simulations
            return config_wrf.num_test * config_wrf.num_sim

    def __getitem__(self, index):
        #Get fileName through index
        # fileName = '/home/data2/RUN/DAMO/train/2022052012/20220520_123000.npz'
        fileName = self.index_dict[index]
        self.np_source = np.load(fileName)        
            
        if os.path.isfile(config_wrf.pathName_additional):
            self.np_additional = np.load(config_wrf.pathName_additional)

        for variable_index, variable_name in enumerate(config_wrf.single_height_variable):

            if variable_name in config_wrf.additional_variable:
                temp_value = self.np_additional[variable_name]
                
            else:                
                temp_value = self.np_source[variable_name]
      
                if config_wrf.norm_method == 'z-score':
                    temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                    self.norm_mapping[variable_name]['scale']
                    
                elif config_wrf.norm_method == 'min-max':
                    temp_value = temp_value / self.norm_mapping[variable_name]['max']
                    
            for layer in range(self.vertical_layers):
                self.single_feature[:, variable_index, layer] = temp_value            

        for variable_index, variable_name in enumerate(config_wrf.multi_height_variable):

            temp_value = self.np_source[variable_name]

            if config_wrf.norm_method == 'z-score':                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                    
            self.multi_feature[:, variable_index, :] = temp_value

            if(variable_name in config_wrf.multi_height_cumsum_variable):
                # Get index
                variable_index = config_wrf.multi_height_cumsum_variable[variable_name]
                
                temp_value_cumsum_forward = np.cumsum(temp_value, axis=1)/20.0
                temp_value_cumsum_backward = np.cumsum(temp_value[:, ::-1], axis=1)/20.0

                self.multi_cumsum_feature[:, variable_index, :] = temp_value_cumsum_forward

                self.multi_cumsum_feature[:, len(config_wrf.multi_height_cumsum_variable) + variable_index,
                                          :] = temp_value_cumsum_backward  

        for variable_index, variable_name in enumerate(config_wrf.label_multi_height_variable):
            temp_value = self.np_source[variable_name]

            if config_wrf.norm_method == 'z-score':                                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                if variable_name not in self.norm_mapping.keys():
                    print('{} does not have norm_mapping\n'.format(variable_name))
                    
                else:
                    temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                       
            self.label_multi_feature[:, variable_index, :] = temp_value

        for variable_index, variable_name in enumerate(config_wrf.label_single_height_variable):
            temp_value = self.np_source[variable_name]

            if config_wrf.norm_method == 'z-score':                                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                if variable_name not in self.norm_mapping.keys():
                    print('{} does not have norm_mapping\n'.format(variable_name))
                
                else:
                    temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                                  
            if self.flag_get_stat == 1:
                #Change to this before running get_mean_std.py
                for layer in range(self.vertical_layers): 
                    if len(temp_value.shape) == 2:
                        self.label_single_feature[:, variable_index, layer] = temp_value[:, 0]
                    elif len(temp_value.shape) == 1:
                        self.label_single_feature[:, variable_index, layer] = temp_value[:]                        
                    
            else:
                #Change to this before training
                if len(temp_value.shape) == 2:
                    self.label_single_feature[:, variable_index, 0] = temp_value[:, 0]

                elif len(temp_value.shape) == 1:
                    self.label_single_feature[:, variable_index, 0] = temp_value[:]
                
        for variable_index, variable_name in enumerate(config_wrf.auxiliary_variable):
            if variable_name == 'p_diff':
                temp_value_p = self.np_source['p']
                temp_value_p = np.concatenate((temp_value_p, \
                np.ones((config_wrf.time_length,1))*config_wrf.constant_variable['ptop']),axis=1)
                    
                temp_value = temp_value_p[:, 0:-1] - temp_value_p[:, 1:]
                
            elif variable_name == 'qv_sat':
                temp_value_t = self.np_source['t']
                                                                
                #pressure here is level pressure, convert to layer pressure            
                temp_value_p = self.np_source['p']
                temp_value_p = np.concatenate((temp_value_p, \
                np.ones((config_wrf.time_length,1))*config_wrf.constant_variable['ptop']),axis=1)                                    
                temp_value_p = (temp_value_p[:, 0:-1] + temp_value_p[:, 1:])/2
                
                temp_value =  calculate_qv_saturated(temp_value_p, temp_value_t)   

            elif variable_name == 'rh':      
                qv = self.np_source['q']

                temp_value_t = self.np_source['t']
                temp_value_p = self.np_source['p']
                temp_value_p = np.concatenate((temp_value_p, \
                np.ones((config_wrf.time_length,1))*config_wrf.constant_variable['ptop']),axis=1)                                    
                temp_value_p = (temp_value_p[:, 0:-1] + temp_value_p[:, 1:])/2                
                qv_sat =  calculate_qv_saturated(temp_value_p, temp_value_t)   
                       
                # index_var = config_wrf.auxiliary_variable.index('qv_sat')
                # qv_sat = self.auxiliary_feature[:, index_var, :]
                # qv_sat = unnormalized_feature(qv_sat, 'qv_sat', config_wrf.norm_method, self.norm_mapping)
                                      
                temp_value = calculate_rh(qv, qv_sat)

            elif variable_name == 'w0avg':     
                temp_value_w0avg = self.np_source['w0avg']
                temp_value_w = self.np_source['w']
                w0 = 0.5*(temp_value_w[:, 0:-1] + temp_value_w[:, 1:])
                
                W0AVGfctr = config_wrf.TST - 1
                W0fctr = 1 
                W0den = config_wrf.TST
                
                temp_value_w0avg = (temp_value_w0avg * W0AVGfctr + w0 * W0fctr) / W0den
                                
                temp_value = 0.5*(temp_value_w[:, 0:-1] + temp_value_w[:, 1:])            
                
            elif variable_name == 'height':     
                temp_value_dz = self.np_source['dz8w']
                
                temp_value = np.zeros(temp_value_dz.shape)
                temp_value[:, 0] = 0.5*temp_value_dz[:, 0]
                for k in np.arange(1, temp_value.shape[1]):
                    temp_value[:, k] = temp_value[:, k-1] + 0.5*(temp_value_dz[:, k-1] + temp_value_dz[:, k])
                
            elif variable_name in {'cape', 'cin'}:     
                temp_value_plevel = self.np_source['p']
                
                #pressure here is level pressure, convert to layer pressure            
                temp_value_p = self.np_source['p']
                temp_value_p = np.concatenate((temp_value_p, \
                np.ones((config_wrf.time_length,1))*config_wrf.constant_variable['ptop']),axis=1)                                    
                temp_value_p = (temp_value_p[:, 0:-1] + temp_value_p[:, 1:])/2
                temp_value_p = temp_value_p.T
                
                pres_hpa = np.zeros((temp_value_p.shape[0], 1, temp_value_p.shape[1]))
                pres_hpa[:, 0, :] = temp_value_p/100
                
                temp_value_t = self.np_source['t'].T 
                tkel = np.zeros(pres_hpa.shape)
                tkel[:, 0, :] = temp_value_t
                                                  
                temp_value_q = self.np_source['q'].T
                qv = np.zeros(pres_hpa.shape)
                qv[:, 0, :] = temp_value_q
                
                ter_follow = True
                
                index_var = config_wrf.auxiliary_variable.index('height')
                temp_height = self.auxiliary_feature[:, index_var, :]
                temp_height = unnormalized_feature(temp_height, 'height', config_wrf.norm_method, self.norm_mapping)             
                temp_height = temp_height.T
                height = np.zeros(pres_hpa.shape)
                height[:, 0, :] = temp_height
                
                terrain = np.zeros((1, pres_hpa.shape[-1])).T
                
                psfc_hpa = np.zeros(terrain.shape)
                # psfc_hpa[0, :] = temp_value_plevel[:, 0]/100
                psfc_hpa[:,0] = temp_value_plevel[:, 0]/100
                
                cape, cin = wrf.cape_3d(pres_hpa, tkel, qv, height, terrain, psfc_hpa, ter_follow)
                                
                if variable_name == 'cape':
                    temp_value = cape[:, 0, :].T
                elif variable_name == 'cin':
                    temp_value = cin[:, 0, :].T
                    
            else:
                temp_value = self.np_source[variable_name]                
        
            if config_wrf.norm_method == 'z-score':
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']
                
            elif config_wrf.norm_method == 'min-max':
                temp_value = temp_value / self.norm_mapping[variable_name]['max']
                
            self.auxiliary_feature[:, variable_index, :] = temp_value                
            
        if config_wrf.flag_filter == 1:
            temp_value = self.np_source[config_wrf.filter_variable]
            for layer in range(self.vertical_layers): 
                if len(temp_value.shape) == 2:
                    self.filter_feature[:, 0, layer] = temp_value[:, 0]
                elif len(temp_value.shape) == 1:    
                    self.filter_feature[:, 0, layer] = temp_value[:]
                                        
        single_feature_tt = torch.tensor(
        self.single_feature, dtype=torch.float32)

        multi_feature_tt = torch.tensor(
        self.multi_feature, dtype=torch.float32)

        multi_cumsum_feature_tt = torch.tensor(
        self.multi_cumsum_feature, dtype=torch.float32)

        label_multi_feature_tt = torch.tensor(
        self.label_multi_feature, dtype=torch.float32)

        label_single_feature_tt = torch.tensor(
        self.label_single_feature, dtype=torch.float32)        

        filter_feature_tt = torch.tensor(
        self.filter_feature, dtype=torch.float32)   
        
        feature_tf = torch.cat([single_feature_tt, multi_feature_tt], dim = 1)
        
        if len(config_wrf.auxiliary_variable) != 0:
            auxiliary_result_tf = torch.tensor(
            self.auxiliary_feature, dtype=torch.float32)    
            
            feature_tf = torch.cat([feature_tf, auxiliary_result_tf], dim=(1))
        
        #Add more weights to single height label
        label_feature_tt = torch.cat([label_single_feature_tt, label_multi_feature_tt], dim = (1))        
                        
        if self.evaluate == "False":                
            random_index = np.random.choice(np.arange(config_wrf.time_length), \
            size  = self.time_length, replace = False)
                
        else:
            random_index = np.arange(config_wrf.time_length)
            
        return feature_tf[random_index,:,:], label_feature_tt[random_index,:,:], \
        filter_feature_tt[random_index,:,:]

class WrfCu():    
    def __init__(self, np_source, vertical_layers, norm_mapping={}, flag_get_stat = 0):
        time_length = config_wrf.time_length
        
        self.flag_get_stat = flag_get_stat
        
        self.np_source = np_source
        self.norm_mapping = norm_mapping
        
        if os.path.isfile(config_wrf.pathName_additional):
            self.np_additional = np.load(config_wrf.pathName_additional)

        self.vertical_layers = vertical_layers
        
        self.label_multi_feature = np.zeros(
            [time_length, len(config_wrf.label_multi_height_variable), self.vertical_layers])

        self.label_single_feature = np.zeros(
            [time_length, len(config_wrf.label_single_height_variable), self.vertical_layers])

        self.single_feature = np.zeros(
            [time_length, len(config_wrf.single_height_variable), self.vertical_layers])

        self.multi_feature = np.zeros(
            [time_length, len(config_wrf.multi_height_variable), self.vertical_layers])

        self.multi_cumsum_feature = np.zeros(
            [time_length, 2*len(config_wrf.multi_height_cumsum_variable), self.vertical_layers])

        self.auxiliary_feature = np.zeros(
            [time_length, len(config_wrf.auxiliary_variable), self.vertical_layers])
        
        # self.constant_value = {}
        # for variable in config_wrf.constant_variable_list:
        #     self.constant_value[variable] = self.norm_mapping[variable]["mean"]
            
    def forward(self):
        
        for variable_index, variable_name in enumerate(config_wrf.single_height_variable):

            if variable_name in config_wrf.additional_variable:
                temp_value = self.np_additional[variable_name]  
                
            else:                
                temp_value = self.np_source[variable_name]

                if config_wrf.norm_method == 'z-score':                                
                    temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                    self.norm_mapping[variable_name]['scale']
    
                elif config_wrf.norm_method == 'min-max':
                    temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                           
            for layer in range(self.vertical_layers):
                self.single_feature[:, variable_index, layer] = temp_value

        for variable_index, variable_name in enumerate(config_wrf.multi_height_variable):

            temp_value = self.np_source[variable_name]
            
            if config_wrf.norm_method == 'z-score':                                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                                       
            self.multi_feature[:, variable_index, :] = temp_value

            if(variable_name in config_wrf.multi_height_cumsum_variable):
                # Get index
                variable_index = config_wrf.multi_height_cumsum_variable[variable_name]
                
                temp_value_cumsum_forward = np.cumsum(temp_value, axis=1)/20.0
                temp_value_cumsum_backward = np.cumsum(temp_value[:, ::-1], axis=1)/20.0

                self.multi_cumsum_feature[:, variable_index, :] = temp_value_cumsum_forward

                self.multi_cumsum_feature[:, len(config_wrf.multi_height_cumsum_variable) + variable_index,
                                          :] = temp_value_cumsum_backward  

            # else:

            #     temp_value = self.constant_value[variable_name]

            #     if config_wrf.norm_method == 'z-score':                                
            #         temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
            #         self.norm_mapping[variable_name]['scale']
    
            #     elif config_wrf.norm_method == 'min-max':
            #         temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                                 
            #     self.multi_feature[:, variable_index, 0:self.vertical_layers] = temp_value
 
        for variable_index, variable_name in enumerate(config_wrf.label_multi_height_variable):
            temp_value = self.np_source[variable_name]

            if config_wrf.norm_method == 'z-score':                                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                temp_value = temp_value / self.norm_mapping[variable_name]['max']
                                                         
            self.label_multi_feature[:, variable_index, :] = temp_value

        for variable_index, variable_name in enumerate(config_wrf.label_single_height_variable):
            temp_value = self.np_source[variable_name]
            
            if config_wrf.norm_method == 'z-score':                                
                temp_value = (temp_value - self.norm_mapping[variable_name]['mean']) / \
                self.norm_mapping[variable_name]['scale']

            elif config_wrf.norm_method == 'min-max':
                temp_value = temp_value / self.norm_mapping[variable_name]['max']                                                         

            #Change to this before running get_mean_std.py
            if self.flag_get_stat == 1:
                for layer in range(self.vertical_layers): 
                    self.label_single_feature[:, variable_index, layer] = temp_value[:, 0]

            else:                    
                #Change to this before training
                if len(temp_value.shape) == 2:
                    self.label_single_feature[:, variable_index, 0] = temp_value[:, 0]

                elif len(temp_value.shape) == 1:
                    self.label_single_feature[:, variable_index, 0] = temp_value[:]
                    
        for variable_index, variable_name in enumerate(config_wrf.auxiliary_variable):
            temp_value = self.np_source[variable_name]
            self.auxiliary_feature[:, variable_index,:] = temp_value

        single_feature_tt = torch.tensor(
            self.single_feature, dtype=torch.float32)
        
        multi_feature_tt = torch.tensor(
            self.multi_feature, dtype=torch.float32)

        multi_cumsum_feature_tt = torch.tensor(
            self.multi_cumsum_feature, dtype=torch.float32)

        label_multi_feature_tt = torch.tensor(
        self.label_multi_feature, dtype=torch.float32)

        label_single_feature_tt = torch.tensor(
        self.label_single_feature, dtype=torch.float32)

        if len(config_wrf.auxiliary_variable) != 0:
            auxiliary_result_tf = torch.tensor(
            self.auxiliary_feature, dtype=torch.float32)
    
            p_diff = auxiliary_result_tf - torch.roll(auxiliary_result_tf, -1, 2)
            p_diff = torch.cat([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], dim=2)

        feature_tf = torch.cat([single_feature_tt, multi_feature_tt], dim = 1)
        
        if len(config_wrf.auxiliary_variable) != 0:
            feature_tf = torch.cat([feature_tf, 
             (p_diff - 17.2)/9.8,
             multi_cumsum_feature_tt
             ], dim=(1))
        
        #Add more weights to single height label
        label_feature_tt = torch.cat([label_single_feature_tt, label_multi_feature_tt], dim = (1))

        if len(config_wrf.auxiliary_variable) != 0:
            return feature_tf, label_feature_tt, auxiliary_result_tf
        else:
            return feature_tf, label_feature_tt, torch.tensor([np.nan])        

def test1():

    fileName = '/home/data2/RUN/DAMO/train/2022052012/20220520_123000.npz'
    var = np.load(fileName)
    model = WrfCu(var, norm_mapping=norm_mapping)
    feature_tf, label_feature_tt, auxiliary_result_tf = model.forward()

    print(feature_tf.shape, label_feature_tt.shape, auxiliary_result_tf.shape)

    print(torch.mean(feature_tf, dim=(0, 2)))
    print(torch.std(feature_tf, dim=(0, 2)))

    print(torch.mean(label_feature_tt, dim=(0, 2)))
    print(torch.std(label_feature_tt, dim=(0, 2)))

    print(torch.mean(auxiliary_result_tf, dim=(0, 2)))
    print(torch.std(auxiliary_result_tf, dim=(0, 2)))

    return 0

def test2():
    dateset =  WrfCuDataset(vertical_layers = 44, type = 'train', norm_mapping=norm_mapping)
    dataLoader = DataLoader(dataset= dateset)
    for batch_idx, (feature, targets, auxis) in enumerate(dataLoader): 
        print(batch_idx, feature.shape, targets.shape, auxis.shape)


if __name__ == '__main__':
    test2()
