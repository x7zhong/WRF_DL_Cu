#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 20:34:52 2023

@author: xiaohui
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from numpy import meshgrid

font1 = {'weight' : 'normal',
          'size'   :  30 }

colormap = ''
# colormap = 'Reds'
colormap = 'jet'

ListofVar = ['nca', 'pratec', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']  
ListofVar = ['nca', 'pratec', \
'rthcuten', 'rqvcuten', 'rqccuten', 'rqrcuten']  
    
# ListofVar = ['rqccuten', 'rqrcuten', 'rqicuten', 'rqscuten']  
# ListofVar = ['rqicuten', 'rqscuten']  
    
# Var_ticks = {'nca': list(np.arange(0, 400, 100)), \
#              'pratec': list(np.arange(0, 0.0020, 0.0005)), \
#              'rthcuten': list(np.arange(-6e-4, 18e-4, 6e-4)), \
#              'rqvcuten': list(np.arange(-3e-07, 3e-07, 1e-07)), \
#              'rqccuten': list(np.arange(-1e-08, 4e-08, 1e-08)), \
#              'rqrcuten': list(np.arange(0, 6e-08, 1e-08)), \
#              'rqicuten': list(np.arange(-2e-09, 3e-09, 1e-09)), \
#              'rqscuten': list(np.arange(-2e-012, 3e-012, 1e-012)),                  
#                  }
# unit = {'rthcuten': 'K s-1', \
#         'rqvcuten': 'kg kg-1 s-1', \
#         'rqccuten': 'kg kg-1 s-1', \
#         'rqrcuten': 'kg kg-1 s-1'
#                  }      

Var_ticks = {'nca': list(np.arange(0, 400, 100)), \
             'pratec': list(np.arange(0, 140, 20)), \
             'rthcuten': list(np.arange(-40, 120, 20)), \
             'rqvcuten': list(np.arange(-0.6, 0.6, 0.2)), \
             'rqccuten': list(np.arange(-0.02, 0.08, 0.02)), \
             'rqrcuten': list(np.arange(0, 0.08, 0.02)), \
             'rqicuten': list(np.arange(-2e-09, 3e-09, 1e-09)), \
             'rqscuten': list(np.arange(-2e-012, 3e-012, 1e-012)),                  
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
num_rows = 3
num_cols = 2

# num_rows = 1
# num_cols = 2

dirName_plot = '/home/yux/Code/Python/AI_and_Cu/Check_results/Plot'
dirName_data = '/home/data2/RUN/DAMO/xiaohui'       

#%%

def generate_correlation_contourmap(x_val, y_val, min_value, max_value, grid_number = 300):
    print(min_value, max_value, grid_number)
    x_vector = np.linspace(min_value, max_value, grid_number)
    spacing = x_vector[1] - x_vector[0]

    result_np = np.zeros( (len(x_vector) , len(x_vector)) )
    
    x_index = np.array( (x_val - min_value)//spacing, dtype = np.int32 )
    y_index = np.array( (y_val - min_value)//spacing, dtype = np.int32 )
    
    x_index_vector = x_index[(x_index >0) & (x_index< grid_number) & (y_index >0) & (y_index< grid_number)]
    y_index_vector = y_index[(x_index >0) & (x_index< grid_number) & (y_index >0) & (y_index< grid_number)]
    

    for x_index_, y_index_ in zip(x_index_vector,y_index_vector):
        result_np[x_index_, y_index_] += 1
    
    result_np = result_np/np.sum(result_np)
    result_np_plot = result_np[:]
    return x_vector, result_np_plot

def calcuate_stat(x,y):
    bias = np.mean(x - y)
    correlation_coefficient = np.corrcoef(x, y)[0,1]
    # rmse = np.std(x - y)
    rmse = np.sqrt(np.mean(np.square(x - y)))
    print(f"bias:{bias}, correlation:{correlation_coefficient},rmse:{rmse} ")
    return bias, correlation_coefficient, rmse

#%%
fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows*5.5))
# fig, axs = plt.subplots(2, 2, figsize=(2 * 7, 2*5.5))

for i in range(num_rows):
    
    for j in range(num_cols):
        
        index = i*num_cols + j
        
        index_ax = (i, j)
        
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
        
        if varName == 'nca':
            #dt is 15
            # Var_model = Var_model*15
            Var_gt = Var_gt/15
            
            #smaller than 0 all set to 0.
            Var_gt[Var_gt < 0.01] = 0
            
        if varName == 'pratec':
            Var_gt[Var_gt < 0] = 0
            Var_model[Var_model < 0] = 0            
            
        if varName in Var_conv.keys():
            Var_gt = Var_gt * Var_conv[varName]
            Var_model = Var_model * Var_conv[varName]
            
        x_vector, y_vector = generate_correlation_contourmap( \
        x_val = Var_gt.flatten(), y_val = Var_model.flatten(),        
        min_value = Var_ticks[varName][0], max_value = Var_ticks[varName][-1], grid_number = 300)                                        
        
        bias, correlation, rmse =  calcuate_stat(\
        np.array(Var_gt.flatten()),
        np.array(Var_model.flatten()) )        
            
        Var_model = []
        Var_gt = []
        
        y_vector_plot = np.log10(y_vector)
        x_mesh, y_mesh = meshgrid(x_vector, x_vector) 
        # fig0_contourf = axs[index_ax].contourf(x_mesh, y_mesh, y_vector_plot, cmap = colormap)            
        fig0_contourf = axs[index_ax].contourf(x_mesh, y_mesh, y_vector_plot)            
        
        chartBox = axs[index_ax].get_position()
        
        # axs[index_ax].text(\
        # 0.05*(x_vector.max() - x_vector.min()) + x_vector.min(), \
        # 0.9*(x_vector.max() - x_vector.min()) + x_vector.min(), 
        # f"R\u00b2:{correlation:.5f}\nRMSE:{rmse:.5f}",font = font1)
        
        axs[index_ax].grid()
        axs[index_ax].axis('equal')
        axs[index_ax].set_aspect('equal', 'box')
        axs[index_ax].set_xticks(Var_ticks[varName])
        axs[index_ax].set_yticks(Var_ticks[varName])
        
        # if i > 0:
        #     axs[index_ax].set_yticklabels([])      

        axs[index_ax].plot([Var_ticks[varName][0], Var_ticks[varName][-1]], \
                           [Var_ticks[varName][0], Var_ticks[varName][-1]] ) 

        fig0_contourf.set_clim(-8,-1)    
        clb = fig.colorbar(fig0_contourf, ax=axs[index_ax], ticks = [-8,-7,-6,-5,-4,-3,-2,-1],
        fraction = 0.04)   

        clb.ax.tick_params(labelsize=25)
    
        axs[index_ax].tick_params(axis='both', labelsize = 25) 
        
        if varName in unit.keys():
            axs[index_ax].set_title("{} ({})".format(varName, unit[varName]), y=1.05,font = font1)
        else:
            axs[index_ax].set_title("{}".format(varName), y=1.05,font = font1)
                
        pathName = os.path.join(dirName_plot, 'correlation.png')
        plt.tight_layout()
        fig.savefig(pathName)  

pathName = os.path.join(dirName_plot, 'correlation.png')
plt.tight_layout()
fig.savefig(pathName)
