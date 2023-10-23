#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:04:49 2023

@author: xiaohui
"""

import os
import numpy as np
import torch
import torch.nn as nn
import onnxruntime

from utils.config_wrf import config_wrf
from utils.make_wrfCu_data import WrfCu
from utils.config_norm import norm_mapping
from dl_inference.cu_utils import mskf_constraint, trigger_consistency, nca_constraint

model_name = 'LSTMClassifier_Reg_32_3'
varName = 'all_abs-max_mae_trigger_specified'
dirName_checkpoints = '/home/yux/Code/Python/AI_and_Cu/scripts/checkpoints'

dirName_model = os.path.join(dirName_checkpoints, model_name, varName)

# file_name = 'LSTMClassifier_Reg_32_3_all0000model.pth'
file_name = 'LSTMClassifier_Reg_32_3_all0005model.pth'
file_name = 'LSTMClassifier_Reg_32_3_all0130model.pth'

pathName_model = os.path.join(dirName_model, file_name)
                              
pathName_onnx = os.path.join(dirName_model, 'model_mskf.onnx')

device = torch.device('cpu')

model = torch.load(pathName_model, map_location=device)

# model = torch.load(pathName_model)

class UnsqueezeModule(nn.Module):
    def __init__(self, model):
        super(UnsqueezeModule, self).__init__()
        self.model = model
    def forward(self, input_):
        input_ = input_.squeeze(-1)
        return self.model(input_)

def unnormalized(predicts, targets, norm_mapping, label_all_variable_cf, label_all_variable_reg, norm_method):
        
    predicts_unnorm = np.zeros(predicts.shape)
    targets_unnorm = targets.copy()
    
    # logger.info("label_all_variable {}".format(label_all_variable))
    # logger.info("norm_mapping {}".format(norm_mapping))
    
    for index, variable_name in enumerate(label_all_variable_reg):
        
        index_target = index + len(label_all_variable_cf)
        
        # logger.info("start denormalize variable_name {}".format(variable_name))
        # logger.info("max {}".format(norm_mapping[variable_name]))
        
        if norm_method == 'z-score':                                
            
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            norm_mapping[variable_name]["scale"] + \
            norm_mapping[variable_name]["mean"]
            
            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            norm_mapping[variable_name]["scale"] + \
            norm_mapping[variable_name]["mean"]
            
        elif norm_method == 'min-max':                                
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            norm_mapping[variable_name]["max"]

            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            norm_mapping[variable_name]["max"]
            
        elif norm_method == 'abs-max':   
            predicts_unnorm[:, index, :] = predicts[:, index, :] * \
            max( abs(norm_mapping[variable_name]['max']), \
            abs(norm_mapping[variable_name]['min']) )
                
            targets_unnorm[:, index_target, :] = targets[:, index_target, :] * \
            max( abs(norm_mapping[variable_name]['max']), \
            abs(norm_mapping[variable_name]['min']) )                        
                
    return predicts_unnorm, targets_unnorm

model.eval()
# unsqueeze_model = UnsqueezeModule(model)
unsqueeze_model = model

# exit()

# set the model to inference mode 
unsqueeze_model.eval() 
feature_channel = config_wrf.num_all_feature
batch_size = 1
# Let's create a dummy input tensor  
# set batch_size to -1 to test if dynamic batch_size works
#dummy_input = torch.randn(1, input_size, 57, 1, requires_grad=False)  
dummy_input = torch.randn(batch_size, feature_channel, config_wrf.vertical_layers)  
# dummy_input = torch.randn(batch_size, feature_channel, config_wrf.vertical_layers, 1)  

# Export the model   
torch.onnx.export(unsqueeze_model,         # model being run 
     dummy_input,       # model input (or a tuple for multiple inputs) 
     pathName_onnx,       # where to save the model
     export_params=True,  # store the trained parameter weights inside the model file 
     opset_version=10,    # the ONNX version to export the model to 
     do_constant_folding=True,  # whether to execute constant folding for optimization 
     input_names = ['input'],   # the model's input names 
     output_names = ['output'], # the model's output names 
     dynamic_axes = {'input': {0: 'batch_size'}, 
                     'output': {0: 'batch_size'}}
     ) 

print(" ") 
print('Model has been converted to ONNX') 

#%%

pathName_input = '/home/data2/RUN/DAMO/train/2022052412/20220524_120000_input.npz'
pathName_output = '/home/data2/RUN/DAMO/train/2022052412/20220524_120000_output.npz'

# pathName_input = '/home/data2/RUN/DAMO/train/2022052412/20220524_120500_input.npz'
# pathName_output = '/home/data2/RUN/DAMO/train/2022052412/20220524_120500_output.npz'

np_source_in = np.load(pathName_input)
np_source_out = np.load(pathName_output)

batch_size = config_wrf.time_length

model = torch.load(pathName_model, map_location = device)

print(model)

dateset =  WrfCu(np_source_in, np_source_out, config_wrf.vertical_layers, norm_mapping)
feature, targets, filters = dateset.forward()
feature = feature.cpu().numpy()
targets = targets.cpu().numpy()
np.save('feature.npy', feature)

###For checking if wrf mskf_data_preprocess is correct
def reshape(Var):
    b = Var.copy()
    for i in range(b.shape[1]):
        for level in range(Var.shape[2]):
            temp = Var[:,i,level]
            temp = temp.reshape((220, 200)).T
            temp = temp.reshape((-1,))
            b[:,i,level] = temp
    return b     
       
feature = reshape(feature)
feature2 = np.load('/home/data2/RUN/JIJIN/Cu-main2/WRFV4/test/em_real/feature.npy')
d = feature - feature2
print('###For checking if wrf mskf_data_preprocess is correct###')
for i in range(27):
    print(config_wrf.feature_all_variable[i],'difference max',d[:,i,:].max(),'min',d[:,i,:].min())

###For checking if wrf mskf_data_preprocess is correct

print('feature shape',feature.shape)

with torch.no_grad():
    
    # Compare model prediction by original model and onnx model
    model.eval()

    predicts_cf, predicts = model.forward(torch.tensor(feature2))

print('predicts shape',predicts.shape,'predicts_cf shape',predicts_cf.shape)
print('predicts_cf max',predicts_cf.max(),'min',predicts_cf.min())
       
label_all_variable_cf = ['trigger']
label_single_height_variable = ['nca', 'pratec']
label_multi_height_variable = ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']

label_all_variable_reg = label_single_height_variable + label_multi_height_variable

norm_method = 'abs-max'

predicts_unnorm, targets_unnorm = unnormalized(
predicts, targets, norm_mapping, label_all_variable_cf, label_all_variable_reg, norm_method)    

predicts = predicts.cpu().numpy()
predicts_cf = predicts_cf.cpu().numpy()
trigger_predict = (predicts_cf > 0.5).flatten()
trigger_gt = targets_unnorm[:,0,0]

accuracy = (trigger_gt == trigger_predict).sum() / trigger_gt.shape[0]
print('trigger percent',np.sum(trigger_gt)/trigger_gt.shape[0],'accuracy',accuracy)

for index, variable_name in enumerate(label_all_variable_reg):
    index_target = index + len(label_all_variable_cf)
    if index < 2:
        print(variable_name,\
              'predicts_unnorm max',predicts_unnorm[:,index,0].max(), \
              'predicts_unnorm min', predicts_unnorm[:,index,0].min(), \
              'targets_unnorm max',targets_unnorm[:,index_target,0].max(), \
              'targets_unnorm min', targets_unnorm[:,index_target,0].min())
    else:
        print(variable_name,\
              'predicts_unnorm max',predicts_unnorm[:,index,:].max(), \
              'predicts_unnorm min', predicts_unnorm[:,index,:].min(), \
              'targets_unnorm max',targets_unnorm[:,index_target,:].max(), \
              'targets_unnorm min', targets_unnorm[:,index_target,:].min())
        
###apply constraint

ListofVar = config_wrf.label_all_variable_reg             
predicts = mskf_constraint(feature, predicts, ListofVar, config_wrf)
                    
ListofVar = ['trigger']                
predicts_cf = mskf_constraint(feature, predicts_cf, ListofVar, config_wrf)

predicts = trigger_consistency(predicts, predicts_cf, config_wrf)

predicts_unnorm, targets_unnorm = unnormalized(
predicts, targets, norm_mapping, config_wrf.label_all_variable_cf, \
config_wrf.label_all_variable_reg, config_wrf.norm_method)

#Ensure nca to be integer
predicts_unnorm = nca_constraint(predicts_unnorm, config_wrf.label_all_variable_reg)

for index, variable_name in enumerate(label_all_variable_reg):
    index_target = index + len(label_all_variable_cf)
    if index < 2:
        print(variable_name,\
              '2 predicts_unnorm max',predicts_unnorm[:,index,0].max(), \
              '2 predicts_unnorm min', predicts_unnorm[:,index,0].min(), \
              '2 targets_unnorm max',targets_unnorm[:,index_target,0].max(), \
              '2 targets_unnorm min', targets_unnorm[:,index_target,0].min())
    else:
        print(variable_name,\
              '2 predicts_unnorm max',predicts_unnorm[:,index,:].max(), \
              '2 predicts_unnorm min', predicts_unnorm[:,index,:].min(), \
              '2 targets_unnorm max',targets_unnorm[:,index_target,:].max(), \
              '2 targets_unnorm min', targets_unnorm[:,index_target,:].min())
np.save('predicts_unnorm.npy', predicts_unnorm)   
     
####For checking if wrf build output is correct
output = np.load('/home/data2/RUN/JIJIN/Cu-main2/WRFV4/test/em_real/output.npz')
label_variable = ['nca', 'pratec', 'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']
print('###For checking if wrf output is correct###')
for i in range(len(label_variable)):
    if i == 0:
        d = output[label_variable[i]] - predicts_unnorm[:,i,0].flatten()*15
    elif i == 1:
        d = output[label_variable[i]] - predicts_unnorm[:,i,0].flatten()
    else:
        d = output[label_variable[i]] - predicts_unnorm[:,i,:].flatten()
        
    print(label_variable[i],'difference max',d.max(),'min',d.min())

####For checking if wrf build output is correct
            
###          
  
# session = onnxruntime.InferenceSession(pathName_onnx)
session = onnxruntime.InferenceSession(pathName_onnx, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# compute ONNX Runtime output prediction
print('output length is',len(session.get_outputs()))

predicts_cf_onnx, predicts_onnx = session.run(None, {input_name: feature})

print('predicts_onnx shape',predicts_onnx.shape,'predicts_cf_onnx shape',predicts_cf_onnx.shape) 

difference = np.abs(predicts_onnx - predicts)
difference_cf = np.abs(predicts_cf_onnx - predicts_cf)

print('difference max:{}, min: {}'.format(difference.max(), difference.min()))
print('difference_cf max:{}, min: {}'.format(difference_cf.max(), difference_cf.min()))
print('predicts_cf_onnx max:{}, min: {}'.format(predicts_cf_onnx.max(), predicts_cf_onnx.min()))

''
