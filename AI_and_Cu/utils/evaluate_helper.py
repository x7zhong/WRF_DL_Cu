#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:17:42 2023

@author: xiaohui
"""

import torch
import numpy as np
import sys
import os
from sklearn import metrics
import torch.nn.functional as F

sys.path.append("..")
from utils.plot_helper import plot_variable
from utils.plot_helper import plot_variable_2d
from utils.cu_utils import mskf_constraint, trigger_consistency, nca_constraint
from utils.file_helper import FileUtils

class RunningMeter(object):
    """
        Computes and stores the average and current value
        for smoothing of the time series
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.count = 0

    def update(self, value, count):
        self.count = self.count + count
        self.value = self.value + value*count

    def getmean(self):
        return self.value/self.count

    def getsqrtmean(self):
        return np.sqrt(self.value/self.count)

def derive_trigger_from_tendency(tendency):
    trigger = torch.zeros(tendency.shape[0])
    
    if len(tendency.shape) == 2:
        temp = torch.sum(torch.abs(tendency),axis=1)
    
        trigger[ torch.where(temp!=0)[0] ] = 1
    
    else:
        trigger[ torch.where(tendency>0.001)[0] ] = 1
        
    return trigger

def Loss_trigger(loss, predict, true, true_trigger, threshold = torch.tensor(0.001)):
    """
    only for triggered grid points
    """
    trigger_index = torch.where(true_trigger > threshold)[0]
    trigger_length = len(trigger_index)

    if trigger_length == 0:
        print('No grid points are triggered.\n')
        return trigger_length, torch.zeros([0])
    
    else:        
        if loss == 'rmse_trigger':
            if len(predict.shape) == 1:
                return predict[trigger_index].shape.numel(), torch.mean(torch.square(predict[trigger_index] - true[trigger_index]))
            else:
                return predict[trigger_index, ::].shape.numel(), torch.mean(torch.square(predict[trigger_index, ::] - true[trigger_index, ::]))

        elif loss == 'mse_trigger':
            if len(predict.shape) == 1:
                return predict[trigger_index].shape.numel(), torch.mean(torch.square(predict[trigger_index] - true[trigger_index]))
            else:
                return predict[trigger_index, ::].shape.numel(), torch.mean(torch.square(predict[trigger_index, ::] - true[trigger_index, ::]))

        elif loss == 'mae_trigger':
            if len(predict.shape) == 1:
                return predict[trigger_index].shape.numel(), torch.mean(torch.abs(predict[trigger_index] - true[trigger_index]))
            else:
                return predict[trigger_index, ::].shape.numel(), torch.mean(torch.abs(predict[trigger_index, ::] - true[trigger_index, ::]))

        elif loss == 'mbe_trigger':
            if len(predict.shape) == 1:
                return predict[trigger_index].shape.numel(), torch.mean(predict[trigger_index] - true[trigger_index])
            else:
                return predict[trigger_index, ::].shape.numel(), torch.mean(predict[trigger_index, ::] - true[trigger_index, ::])
            
        elif loss == 'cc_trigger':
            if len(predict.shape) == 1:
                return predict[trigger_index].shape.numel(), torch.mean(torch.abs(predict[trigger_index] - true[trigger_index]))
            else:
                return predict[trigger_index, ::].shape.numel(), torch.mean(torch.abs(predict[trigger_index, ::] - true[trigger_index, ::]))
            
def get_accuracy(y_prob, y_true):
    
    # y_prob = torch.nn.functional.Sigmoid(y_prob)
    
    if type(y_prob) == np.ndarray:
        accuracy = metrics.accuracy_score(y_true, y_prob > 0.5)
        
    elif type(y_prob) == torch.Tensor:
        # assert y_true.ndim == 1 and y_true.size() == y_prob.size()        
        
        y_prob = y_prob > 0.5
    
        accuracy = (y_true == y_prob).sum().item() / y_true.size(0)
    
    return accuracy

def Loss_all(loss, predict, true, true_trigger = []):

    if loss == 'rmse': 
        return predict.shape.numel(), torch.mean(torch.square(predict - true))

    elif loss == 'mse': 
        return predict.shape.numel(), torch.mean(torch.square(predict - true))

    elif loss == 'mbe':
        return predict.shape.numel(), torch.mean(predict - true)

    elif loss == 'mae':
        return predict.shape.numel(), torch.mean(torch.abs(predict - true))

    elif loss == 'cc':
        return predict.shape.numel(), torch.mean(predict - true)
    
    elif loss == 'accuracy':        
        return predict.shape.numel(), get_accuracy(predict, true)

    elif loss == 'accuracy_trigger':
        predict_trigger = derive_trigger_from_tendency(predict)
        
        return predict_trigger.shape.numel(), get_accuracy(predict_trigger, true_trigger)
    
    elif loss == 'BCE':
        return predict.shape.numel(), \
        torch.nn.functional.binary_cross_entropy(predict,true)
        # torch.nn.functional.binary_cross_entropy_with_logits(predict,true)

    elif loss in {'rmse_trigger', 'mse_trigger', 'mae_trigger', 'mbe_trigger', 'cc_trigger'}:
        
        length_predict, error = Loss_trigger(loss, predict, true, true_trigger, threshold = 0.001)
        
        return length_predict, error
        
def unnormalized(predicts, targets, norm_mapping, label_all_variable_cf, label_all_variable_reg, norm_method):
    
    predicts_unnorm = torch.zeros(predicts.shape).to(predicts.device)
    targets_unnorm = targets.clone().to(targets.device)
    
    # print('unnormalized predicts shape',predicts.shape,'targets shape',targets.shape)
    
    for index, variable_name in enumerate(label_all_variable_reg):
        
        index_target = index + len(label_all_variable_cf)
        
        if norm_method == 'z-score':                                
            
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            torch.tensor(norm_mapping[variable_name]["scale"], dtype=torch.float32) + \
            torch.tensor(norm_mapping[variable_name]["mean"], dtype=torch.float32)
    
            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            torch.tensor(norm_mapping[variable_name]["scale"], dtype=torch.float32) + \
            torch.tensor(norm_mapping[variable_name]["mean"], dtype=torch.float32)
        
        elif norm_method == 'min-max':                                
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            torch.tensor(norm_mapping[variable_name]["max"], dtype=torch.float32)

            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            torch.tensor(norm_mapping[variable_name]["max"], dtype=torch.float32)

        elif norm_method == 'abs-max':                   
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            torch.tensor(max( abs(norm_mapping[variable_name]['max']), \
            abs(norm_mapping[variable_name]['min']) ), dtype=torch.float32)

            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            torch.tensor(max( abs(norm_mapping[variable_name]['max']), \
            abs(norm_mapping[variable_name]['min']) ), dtype=torch.float32)

            # print('For', variable_name,'targets max',targets[:, index_target, 0].max(),\
            #       'targets_unnorm max',targets_unnorm[:, index_target, 0].max(),\
            #       'max norm',max( abs(norm_mapping[variable_name]['max']), \
            #                       abs(norm_mapping[variable_name]['min']) ))
                
    return predicts_unnorm, targets_unnorm

def check_accuracy_evaluate(loader, model, norm_mapping, device, config_wrf, \
                            args, epoch, if_plot=False, if_plot_2d = False, \
                            target_norm_info = None):
    model.eval()
    
    Var_error = {}
    Var_error_normalized = {}
        
    for variable_name in config_wrf.label_all_variable:
        Var_error[variable_name] = {}
        Var_error_normalized[variable_name] = {}
        
        if variable_name in config_wrf.label_classification_variable_possible:
            for error in config_wrf.Listof_error_class:    
                Var_error[variable_name][error] = RunningMeter()  
                Var_error_normalized[variable_name][error] = RunningMeter()  
                
        else:
            
            for error in config_wrf.Listof_error:    
                Var_error[variable_name][error] = RunningMeter()           
                Var_error_normalized[variable_name][error] = RunningMeter()           

    with torch.no_grad():
        # loop = tqdm(loader)
        for batch_idx, (feature, targets, filters, time_str) in enumerate(loader):
            print("Testing Epoch", epoch, "batch", batch_idx)
            
            if args.dataset_type == "WRF":
                feature_shape = feature.shape
                target_shape = targets.shape
                filters_shape = filters.shape

                # print('feature shape {},\
                #       target shape {}'.format(feature_shape, target_shape))
                
                inner_batch_size = feature_shape[0]*feature_shape[1]
                
                feature = feature.reshape(
                inner_batch_size, feature_shape[2], feature_shape[3]).to(device=device)
                
                targets = targets.reshape(
                inner_batch_size, target_shape[2], target_shape[3]).to(device=device)
                
                filters = filters.reshape(
                inner_batch_size, filters_shape[2], filters_shape[3])
                
            else:
                feature = feature.to(device=device)
                targets = targets.to(device=device)
                
                filters = filters

            trigger_index = torch.where(filters[:,0,0] > config_wrf.filter_threshold)[0]

            feature_shape = feature.shape
            target_shape = targets.shape
            filters_shape = filters.shape

            # Get data to cuda if possible
            predicts = []
            predicts_cf = []
            if 'Classifier_Reg' in args.model_name:
                predicts_cf, predicts = model(feature)
                
                # print('0 predicts_cf shape {}\n'.format(predicts_cf.shape))
                # print('0 predicts shape {}\n'.format(predicts.shape))
                
            elif 'Classifier' in args.model_name:
                predicts_cf = model(feature)                
                                
            else:
                predicts = model(feature)
            
            if len(config_wrf.label_all_variable_reg) > 0:
                # if config_wrf.flag_filter == 1:
                    
                #     if len(predicts) > 0:
                    
                #         #filter prediction based on specified filter variable.
                #         #teacher forcing, using 100% ground truth filter.
                #         if len(predicts.shape) == 3:
                #             for i in range(predicts.shape[1]):
                #                 predicts[:, i, :] = predicts[:, i, :] * filters[:, 0, :].to(device=device)
        
                #         elif len(predicts.shape) == 2:
                #             for i in range(predicts.shape[1]):
                #                 predicts[:, i] = predicts[:, i] * filters[:, 0, 0].to(device=device)      
                            
                #Update prediction based on the input data and constraint
                if config_wrf.flag_constraint == 1:         
                    if 'Classifier_Reg' in args.model_name:                        
                        ListofVar = config_wrf.label_all_variable_reg             
                        predicts = mskf_constraint(feature, predicts, ListofVar, config_wrf)
                                            
                        ListofVar = ['trigger']                
                        predicts_cf = mskf_constraint(feature, predicts_cf, ListofVar, config_wrf)
                        
                    elif 'Classifier' in args.model_name:       
                        ListofVar = ['trigger']                
                        predicts_cf = mskf_constraint(feature, predicts_cf, ListofVar, config_wrf)
                                            
                    else:
                        ListofVar = config_wrf.label_all_variable            
                        predicts = mskf_constraint(feature, predicts, ListofVar, config_wrf)
                                                                
                #Ensure concistency between all predicted variables
                if config_wrf.flag_trigger_consistency == 1:   
                    if len(predicts) != 0:                             
                        predicts = trigger_consistency(predicts, predicts_cf, config_wrf)
                                                                    
                # 此处去归一化
                if (args.dataset_type == "WRF"):
                    if len(predicts) != 0:
                        predicts_unnorm, targets_unnorm = unnormalized(
                        predicts, targets, norm_mapping, \
                        config_wrf.label_all_variable_cf, \
                        config_wrf.label_all_variable_reg, \
                        config_wrf.norm_method)
            
                        #Ensure nca to be integer
                        predicts_unnorm = nca_constraint(predicts_unnorm, config_wrf.label_all_variable_reg)            

                if config_wrf.if_save == True:
                    # print('time_str is {}'.format(time_str))
                    dirName_save = os.path.join(config_wrf.dirName_data, "../model_outputs", args.main_folder, args.sub_folder, time_str['initTime_str'][0])
                    FileUtils.makedir(dirName_save)
                        
                    pathName = os.path.join(dirName_save, time_str['forecastTime_str'][0] + ".npz")       
                        
                    #save model outputs
                    outputs = {}
                    for variable_index, variable_name in enumerate(config_wrf.label_all_variable_cf):
          
                        # single height label
                        if variable_name in config_wrf.label_single_height_variable:     
                            outputs[variable_name] = predicts_cf[:, variable_index].cpu().numpy()
                            outputs[variable_name] = outputs[variable_name].reshape((-1, 1))
                    
                    for index, variable_name in enumerate(config_wrf.label_all_variable_reg):
                        index_target = index + len(config_wrf.label_all_variable_cf)
                        
                        if variable_name in config_wrf.label_single_height_variable:                                                            
                            outputs[variable_name] = predicts_unnorm[:, index, 0].cpu().numpy()
                            outputs[variable_name] = outputs[variable_name].reshape((-1, 1))
                            
                        elif variable_name in config_wrf.label_multi_height_variable:
                            outputs[variable_name] = predicts_unnorm[:, index, :].cpu().numpy()
                                                                
                    np.savez(pathName, **outputs)   
                    print('{} is saved\n'.format(pathName))                                         
                    
            flag_multi = 0            
            for variable_index, variable_name in enumerate(config_wrf.label_all_variable_cf):
  
                # single height label
                if variable_name in config_wrf.label_single_height_variable:                                                            
                                        
                    for error in config_wrf.Listof_error_class:                                
                        valid_length, valid_value = Loss_all(error,
                        predicts_cf[:, variable_index], \
                        targets[:, variable_index, 0], \
                        filters[:, 0, 0])      
                            
                        if error in {'accuracy'}:
                            Var_error[variable_name][error].update(valid_value, valid_length)
                            Var_error_normalized[variable_name][error].update(valid_value, valid_length)
                            
                        else:
                            Var_error[variable_name][error].update(valid_value.item(), valid_length)
                            Var_error_normalized[variable_name][error].update(valid_value.item(), valid_length)
                  
                if(if_plot_2d):
                    if(batch_idx == 0) & (variable_name in config_wrf.plot_variable):                        
                            
                        pathName = os.path.join(
                            "results", args.main_folder, args.sub_folder, \
                            variable_name + '_' + str(batch_idx) +  '_epoch' + str(epoch).zfill(4) + ".png")                                                
                    
                        # print('predicts shape {}\n'.format(predicts.shape))
                        # print('targets shape {}\n'.format(targets.shape))
                        
                        plot_range = [norm_mapping[variable_name]['diff_max'], \
                                     -norm_mapping[variable_name]['diff_max'] ]

                        plot_diff_range = [norm_mapping[variable_name]['diff_min'], \
                                           norm_mapping[variable_name]['diff_max'] ]
                            
                        if variable_name in config_wrf.label_classification_variable_possible:
                            predicts_2d = (predicts_cf[:, variable_index].reshape(\
                            (config_wrf.dims[0], config_wrf.dims[1]))).T
                                
                            targets_2d = (targets[:, variable_index, 0].reshape(\
                            (config_wrf.dims[0], config_wrf.dims[1]))).T
                                                                                          
                        if variable_name == 'trigger':
                            predicts_2d = predicts_2d > 0.5
                            
                        plot_variable_2d(predicts_2d, targets_2d, pathName, \
                                          config_wrf, variable_name, plot_range, plot_diff_range, \
                                          lon = [], lat = [])                        

                        # plot_variable_2d(predicts_2d, targets_2d, pathName, \
                        #                   config_wrf, variable_name, [], [], \
                        #                   lon = [], lat = [])  
                        
            for variable_index, variable_name in enumerate(config_wrf.label_all_variable_reg):
                
                variable_index_target = variable_index + len(config_wrf.label_all_variable_cf)
       
                # single height label
                if variable_name in config_wrf.label_single_height_variable:                                                            
                                                                        
                    for error in config_wrf.Listof_error:
                        valid_length, valid_value = Loss_all(error,
                        predicts_unnorm[:, variable_index, 0], \
                        targets_unnorm[:, variable_index_target, 0], \
                        filters[:, 0, 0])                                                

                        valid_length_normalized, valid_value_normalized = Loss_all(error,
                        predicts[:, variable_index, 0], \
                        targets[:, variable_index_target, 0], \
                        filters[:, 0, 0])   
                                                  
                        if error in {'accuracy_trigger'}:
                            Var_error[variable_name][error].update(valid_value, valid_length)
                            Var_error_normalized[variable_name][error].update(valid_value_normalized, valid_length_normalized)
                            
                        else:
                            Var_error[variable_name][error].update(valid_value.item(), valid_length)
                            Var_error_normalized[variable_name][error].update(valid_value_normalized.item(), valid_length_normalized)
                
                # multi height label
                elif variable_name in config_wrf.label_multi_height_variable:                    
                        
                    flag_multi = 1
                    
                    for error in config_wrf.Listof_error:
                    
                        if  error == 'accuracy_trigger':
                            valid_length, valid_value = Loss_all(error,
                            predicts_unnorm[:, variable_index, :], \
                            targets_unnorm[:, variable_index_target, :], \
                            filters[:, 0, 0])
  
                            valid_length_normalized, valid_value_normalized = Loss_all(error,
                            predicts[:, variable_index, :], \
                            targets[:, variable_index_target, :], \
                            filters[:, 0, 0])
                                              
                            Var_error[variable_name][error].update(valid_value, valid_length)                                    
                            Var_error_normalized[variable_name][error].update(valid_value_normalized, valid_length_normalized)                                    
                                
                        else:
                            valid_length, valid_value = Loss_all(error,
                            predicts_unnorm[:, variable_index, :], \
                            targets_unnorm[:, variable_index_target, :], \
                            filters[:, 0, 0])

                            valid_length_normalized, valid_value_normalized = Loss_all(error,
                            predicts[:, variable_index, :], \
                            targets[:, variable_index_target, :], \
                            filters[:, 0, 0])
                                
                            Var_error[variable_name][error].update(valid_value.item(), valid_length)                                    
                            Var_error_normalized[variable_name][error].update(valid_value_normalized.item(), valid_length_normalized)                                    

                if(if_plot):
                    if(batch_idx < 50):
                        print("making plot " + str(batch_idx))
                        
                        file_name = os.path.join(
                            "results", args.main_folder, args.sub_folder, "Flux" +
                            str(batch_idx) + ".png")
                                                
                        # plot_variable(flag_multi, predicts_unnorm, targets_unnorm, \
                        #               variable_index, file_name, 0, config_wrf.vertical_layers, \
                        #               variable_name)
                        
                if(if_plot_2d):
                    if(batch_idx == 0) & (variable_name in config_wrf.plot_variable):                        
                            
                        pathName = os.path.join(
                            "results", args.main_folder, args.sub_folder, \
                            variable_name + '_' + str(batch_idx) +  '_epoch' + str(epoch).zfill(4) + ".png")                                                
                    
                        # print('predicts shape {}\n'.format(predicts.shape))
                        # print('targets shape {}\n'.format(targets.shape))
                        
                        plot_range = [norm_mapping[variable_name]['diff_max'], \
                                     -norm_mapping[variable_name]['diff_max'] ]

                        plot_diff_range = [norm_mapping[variable_name]['diff_min'], \
                                           norm_mapping[variable_name]['diff_max'] ]                            
                            
                        if len(predicts.shape) == 2:
                         
                            predicts_2d = (predicts_unnorm[:, variable_index].reshape(\
                            (config_wrf.dims[0], config_wrf.dims[1]))).T
                                
                            targets_2d = (targets_unnorm[:, variable_index_target, 0].reshape(\
                            (config_wrf.dims[0], config_wrf.dims[1]))).T                                                            

                        else:
                            if variable_name in config_wrf.label_single_height_variable:                                                            
                                predicts_2d = predicts_unnorm[:, variable_index, 0].reshape(\
                                (config_wrf.dims[0], config_wrf.dims[1])).T
                                    
                                targets_2d = targets_unnorm[:, variable_index_target, 0].reshape(\
                                (config_wrf.dims[0], config_wrf.dims[1])).T
                                   
                                print(variable_name,\
                                      'targets_2d max',targets_2d.max(),\
                                      'targets_2d min',targets_2d.min(),\
                                      'predicts_2d max',predicts_2d.max(),\
                                      'predicts_2d min',predicts_2d.min() )
                                
                            elif variable_name in config_wrf.label_multi_height_variable:
                                #sum along the vertical axis
                                print('Plot sum along the vertical axis\n')
                                predicts_2d = (torch.sum(torch.abs(\
                                predicts_unnorm[:, variable_index, :]), axis = -1).reshape(\
                                (config_wrf.dims[0], config_wrf.dims[1]))).T
                                                                                           
                                targets_2d = (torch.sum(torch.abs(\
                                targets_unnorm[:, variable_index_target, :]), axis = -1).reshape(\
                                (config_wrf.dims[0], config_wrf.dims[1]))).T
                            
                        plot_variable_2d(predicts_2d, targets_2d, pathName, \
                                          config_wrf, variable_name, plot_range, plot_diff_range, \
                                          lon = [], lat = [])                        

                        # plot_variable_2d(predicts_2d, targets_2d, pathName, \
                        #                   config_wrf, variable_name, [], [], \
                        #                   lon = [], lat = [])  
                            
    #Calculate average error
    for variable_name in config_wrf.label_all_variable:
        
        if variable_name in config_wrf.label_classification_variable_possible:
            for error in config_wrf.Listof_error_class: 
                Var_error[variable_name][error] = Var_error[variable_name][error].getmean()                
                Var_error_normalized[variable_name][error] = Var_error_normalized[variable_name][error].getmean()                
            
        else:
            
            for error in config_wrf.Listof_error:   
                if error in {'rmse', 'rmse_trigger'}:
                    Var_error[variable_name][error] = Var_error[variable_name][error].getsqrtmean()
                    Var_error_normalized[variable_name][error] = Var_error_normalized[variable_name][error].getsqrtmean()
                    
                elif error in {'mbe', 'mae', \
                               'accuracy', 'cc', \
                               'mbe_trigger', 'mae_trigger', 'mse_trigger', 'accuracy_trigger', \
                               'cc_trigger'}:
                    
                    Var_error[variable_name][error] = Var_error[variable_name][error].getmean()                
                    Var_error_normalized[variable_name][error] = Var_error_normalized[variable_name][error].getmean()                
     
    return Var_error, Var_error_normalized


def check_accuracy(loader, model, norm_mapping, device, config_wrf, \
                   args, epoch, if_plot = False, if_plot_2d = False, target_norm_info = None):
    """
    summarize accuracy
    """
    Var_error, Var_error_normalized = check_accuracy_evaluate(
    loader, model, norm_mapping, device, config_wrf, args, epoch, \
    if_plot, if_plot_2d, target_norm_info = target_norm_info)
        
    return Var_error, Var_error_normalized
