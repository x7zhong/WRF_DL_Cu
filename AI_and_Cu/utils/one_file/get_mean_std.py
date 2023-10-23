#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 19:47:51 2023

@author: xiaohui

"""

import torch
import numpy as np

from config_wrf import config_wrf
from make_wrfCu_data import WrfCuDataset
from config_norm import norm_mapping_standard
from torch.utils.data import DataLoader

def fun_calculate_wrf_statistics(norm_method):
    
    dateset =  WrfCuDataset(vertical_layers = config_wrf.vertical_layers, \
    type = "train", norm_mapping= norm_mapping_standard, flag_get_stat = 1)
    dataLoader = DataLoader(dataset= dateset)
    
    if norm_method == 'min-max':

        for batch_idx, (feature, targets, filters) in enumerate(dataLoader): 
            if( batch_idx ==0):
                print('feature {}'.format(feature.shape))
                print('targets {}'.format(targets.shape))
                feature_max = torch.amax(feature,dim = [0,1,3])
                feature_min = torch.amin(feature,dim = [0,1,3])
                
                targets_max = torch.amax(targets,dim = [0,1,3])
                targets_min = torch.amin(targets,dim = [0,1,3])
                
            else:
                feature_max = torch.max( feature_max, torch.amax(feature,dim = [0,1,3]) )
                feature_min = torch.min( feature_min, torch.amin(feature,dim = [0,1,3]) )
                
                targets_max = torch.max( targets_max, torch.amax(targets,dim = [0,1,3]) )
                targets_min = torch.max( targets_min, torch.amin(targets,dim = [0,1,3]) )
                
        norm_mapping = {}
        for variable_index, variable_name in enumerate(config_wrf.feature_all_variable): 
            norm_mapping[variable_name] = {'max': feature_max[variable_index].cpu().detach().numpy(), \
                                           'min': feature_min[variable_index].cpu().detach().numpy()}
            
        
        for variable_index, variable_name in enumerate(config_wrf.label_all_variable): 
            norm_mapping[variable_name] = {'max': targets_max[variable_index].cpu().detach().numpy(), \
                                           'min': targets_min[variable_index].cpu().detach().numpy()}
                            
        print('{}'.format(norm_mapping))
            
        np.save(config_wrf.pathName_norm, norm_mapping)
        print('{} is saved\n'.format(config_wrf.pathName_norm))
                        
    elif norm_method == 'z-score':
        
        counter = 0
        for batch_idx, (feature, targets, filters) in enumerate(dataLoader): 
            if( batch_idx ==0):
                feature_mean = torch.mean(feature,dim = [0,1,3])
                targets_mean = torch.mean(targets,dim = [0,1,3])
                    
            else:
                feature_mean += torch.mean(feature,dim = [0,1,3])
                targets_mean +=  torch.mean(targets,dim = [0,1,3])
                    
            counter += 1
    
        print('Feature mean:')
        print(feature_mean/counter)
        
        print('\nTarget mean:')
        print(targets_mean/counter)
    
        feature_mean_unsqueeze = (feature_mean/counter).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        targets_mean_unsqueeze = (targets_mean/counter).unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        counter = 0
        for batch_idx, (feature, targets, filters) in enumerate(dataLoader): 
            if( batch_idx ==0):
                feature_mean = torch.mean(torch.square(feature - feature_mean_unsqueeze),dim = [0,1,3])
                targets_mean = torch.mean(torch.square(targets - targets_mean_unsqueeze),dim = [0,1,3])

            else:
                feature_mean +=  torch.mean(torch.square(feature - feature_mean_unsqueeze),dim = [0,1,3])
                targets_mean +=  torch.mean(torch.square(targets - targets_mean_unsqueeze),dim = [0,1,3])

            counter += 1
    
        print('\nFeature std:')
        print(torch.sqrt(feature_mean/counter))
        
        print('\nTarget std:')    
        print(torch.sqrt(targets_mean/counter))           
    
if __name__ == "__main__":
    fun_calculate_wrf_statistics(config_wrf.norm_method)
