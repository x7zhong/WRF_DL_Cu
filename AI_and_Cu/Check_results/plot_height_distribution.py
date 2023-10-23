#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:47:57 2023

@author: xiaohui
"""

import os
import  numpy as np
import torch
import matplotlib.pyplot as plt

font1 = {'weight' : 'normal',
          'size'   :  20 }

ListofVar = ['nca', 'pratec', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']  
ListofVar = ['rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten']      
    
# Var_ticks = {'nca': list(np.arange(0, 400, 100)), \
#              'pratec': list(np.arange(0, 0.0020, 0.0005)), \
#              'rthcuten': list(np.arange(-1e-5, 1.5e-5, 0.5e-5)), \
#              'rqvcuten': list(np.arange(-6e-9, 9e-9, 3e-9)), \
#              'rqccuten': list(np.arange(-1e-09, 1.5e-09, 0.5e-09)), \
#              'rqrcuten': list(np.arange(-3e-09, 4e-09, 1.0e-9))                  
#                  }
    
# unit = {'rthcuten': 'K s-1', \
#         'rqvcuten': 'kg kg-1 s-1', \
#         'rqccuten': 'kg kg-1 s-1', \
#         'rqrcuten': 'kg kg-1 s-1'
#                  }   

Var_ticks = {'rthcuten': list(np.arange(-1, 1.5, 0.5)), \
             'rqvcuten': list(np.arange(-1, 1.5, 0.5)), \
             'rqccuten': list(np.arange(-1, 1.5, 0.5)), \
             'rqrcuten': list(np.arange(-1, 1.5, 0.5))                  
                 }
    
unit = {'pratec': 'mm d-1',
        'rthcuten': 'K d-1', \
        'rqvcuten': 'g kg-1 d-1', \
        'rqccuten': 'g kg-1 d-1', \
        'rqrcuten': 'g kg-1 d-1'
                 }      
    
Var_conv = {'pratec': 86400, \
            'rthcuten': 86400, \
            'rqvcuten': 86400*10**3, \
            'rqccuten': 86400*10**3, \
            'rqrcuten': 86400*10**3}   
    
# num_rows = 4
num_rows = 1
num_cols = 4

# num_rows = 1
# num_cols = 2

dirName_plot = '/home/yux/Code/Python/AI_and_Cu/Check_results/Plot'
dirName_data = '/home/data2/RUN/DAMO/xiaohui'    

#%%

h_lay_vector = np.array(
[98425.31  , 97831.97  , 97146.55  , 96414.305 , 95624.375 ,
       94767.28  , 93826.21  , 92801.35  , 91665.09  , 90454.48  ,
       89147.375 , 87725.96  , 86191.555 , 84552.734 , 82761.63  ,
       80867.61  , 78802.62  , 76622.78  , 74275.234 , 71792.055 ,
       69152.03  , 66372.77  , 63438.41  , 60382.215 , 57179.777 ,
       53854.73  , 50410.13  , 46892.562 , 43286.83  , 39648.33  ,
       35994.85  , 32345.367 , 28766.432 , 25264.475 , 21888.26  ,
       18796.238 , 16083.358 , 13765.426 , 11783.763 , 10087.849 ,
        8639.959 ,  7392.824 ,  6333.0923,  5422.8877])/100

#%%

fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows*8))
# fig, axs = plt.subplots(1, 2, figsize=(1 * 6, 1*8))

for i in range(num_rows):
    
    for j in range(num_cols):
        
        index = i*num_cols + j
        
        if num_rows > 1:
            index_ax = (i, j)
        else:
            index_ax = j
        
        varName = ListofVar[index]      
        
        Var_gt = []
        pathName_gt = os.path.join(dirName_data, varName + '_gt_all.npy')        
        print('Load gt {} from {}\n'.format(varName, pathName_gt))
        Var_gt = np.load(pathName_gt)
        # Var_gt = np.load('/home/data2/RUN/DAMO/train/2022052012/20220522_000000_output.npz')[varName]

        Var_model = []
        pathName_model = os.path.join(dirName_data, varName + '_model_all.npy')
        print('Load predicted {} from {}\n'.format(varName, pathName_model))        
        Var_model = np.load(pathName_model)        
        # Var_model = np.load('/home/data2/RUN/DAMO/model_outputs/LSTMClassifier_Reg_32_3/all_abs-max_mae_trigger_specified/2022052012/20220522_000000.npz')[varName]
    
        if varName in Var_conv.keys():
            Var_gt = Var_gt * Var_conv[varName]
            Var_model = Var_model * Var_conv[varName]
            
        result = Var_model - Var_gt
        
        result_bias = result.mean(axis = 0)
        result_mae = np.abs(result).mean(axis = 0)
        result_std = result.std(axis = 0)
    
        height_vector = h_lay_vector
        
        axs[index_ax].plot(result_mae.flatten(),  height_vector, "-", c = "g", label = "MAE")
        axs[index_ax].plot(result_bias.flatten(), height_vector, '--', c = "g",label = "MBE")
        axs[index_ax].fill_betweenx(height_vector, np.quantile(result, 0.05, axis = 0), 
                          np.quantile(result, 0.95, axis = 0), 
                          alpha = 0.3, color = "g")    
                
        # axs[i].legend(fontsize = 20)
        axs[index_ax].invert_yaxis()
        axs[index_ax].set_xlim([Var_ticks[varName][0], Var_ticks[varName][-1]])
        axs[index_ax].set_xticks(Var_ticks[varName])
        axs[index_ax].set_ylim([1000, 50.0])
        
        if j == 0:        
            axs[index_ax].set_ylabel("Pressure (hPa)", font = font1)
    
        axs[index_ax].tick_params(axis='both', labelsize = 25)                        
            
        if j > 0:
            axs[index_ax].set_yticklabels([])         
            
        # axs[0,i].grid("--",c = "k", )
        axs[index_ax].grid(linestyle = "--",c = "k", )
        # axs[index_ax].set_title(varName, font = font1, y=1.05 )                          

        if varName in unit.keys():
            axs[index_ax].set_title("{} ({})".format(varName, unit[varName]), y=1.05,font = font1)
        else:
            axs[index_ax].set_title("{}".format(varName), y=1.05,font = font1)
            
if num_rows > 1:
    index_ax = (0, 0)
else:
    index_ax = 0
axs[index_ax].legend(fontsize = 25 , loc = "upper left")

pathName = os.path.join(dirName_plot, 'height_distribution.png')
plt.tight_layout()
fig.savefig(pathName)
