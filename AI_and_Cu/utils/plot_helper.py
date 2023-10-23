#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 23:18:27 2023

@author: xiaohui
"""

import torch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_variable(flag_multi, predicts, targets, filename, variable_index, \
                  sample_index, vertical_layers, varName):

    predicts = predicts.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()

    fig = plt.figure(tight_layout=True, figsize=(12, 5),)
    gs = gridspec.GridSpec(3, 1)
    
    # ref: https://matplotlib.org/stable/gallery/subplots_axes_and_figures/align_labels_demo.html#sphx-glr-gallery-subplots-axes-and-figures-align-labels-demo-py
    
    ax = fig.add_subplot(gs[0, 0])
    
    if flag_multi == 1:
        #multi height variable
        ax.plot(predicts[sample_index, variable_index, :], label="predict")
        ax.plot(targets[sample_index, variable_index, :], label="true")
        
    else:
        #single height variable        
        ax.plot(predicts[sample_index, variable_index, 0], label="predict")
        ax.plot(targets[sample_index, variable_index, 0], label="true")
                
    ax.set_xticks(np.arange(1, vertical_layers + 1, 20))
    
    ax.set_ylabel(varName)
    ax.legend()

    plt.savefig(filename, dpi=500, bbox_inches='tight')
    print('{} is saved\n'.format(filename))
    plt.close(fig)
    plt.clf()
    plt.cla()
    
def plot_variable_2d(predicts, targets, pathName, config_wrf, \
                     varName, plot_range, plot_diff_range, \
                     lon = [], lat = []):

    font1 = {'weight' : 'normal',
             'size'   :  30 }
    
    fig, axs = plt.subplots(1,2,figsize=(40, 15))        
    fig_plot = [None] * 2
    
    if type(predicts) == torch.Tensor:
        predicts = predicts.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
    
    print('model {}, shape {}\n'.format(predicts.shape, targets.shape))
        
    for i in range(2):
        
        index_ax = i
        index_fig = i
        
        if i == 0:
            if len(lat) == 0:
                fig_plot[index_fig] = axs[index_ax].pcolor(targets)
                axs[index_ax].set_title('GT', font = font1, y=1.05 )      
                
            else:
                fig_plot[index_fig] = axs[index_ax].pcolor(lon, lat, targets)
                axs[index_ax].set_title('GT', font = font1, y=1.05 )                         

        else:
            if len(lat) == 0:
                fig_plot[index_fig] = axs[index_ax].pcolor(predicts - targets)
                axs[index_ax].set_title('prediction - GT', font = font1, y=1.05 )      
                
            else:
                fig_plot[index_fig] = axs[index_ax].pcolor(lon, lat, predicts - targets)
                axs[index_ax].set_title('prediction - GT', font = font1, y=1.05 )                         
                                    
        axs[index_ax].set_xlabel("longitude", font = font1)                
    
        axs[index_ax].set_ylabel("latitude", font = font1)
        
        axs[index_ax].tick_params(axis='both', labelsize = 25) 
            
        if i == 0:
            if len(plot_range) != 0:
                fig_plot[index_fig].set_clim(plot_range[0], plot_range[-1])    
            
        else:
            if len(plot_diff_range) != 0:            
                fig_plot[index_fig].set_clim(plot_diff_range[0], plot_diff_range[-1])    
            
        clb = plt.colorbar(fig_plot[index_fig], ax = axs[index_ax], label = varName)
             
        if i == 0:           
            clb.set_label(varName, fontsize = 25, font=font1)
            
        else:
            clb.set_label('Difference in ' + varName, fontsize = 25, font=font1)            
            
        clb.ax.tick_params(labelsize=25)                                            
    
    # plt.tight_layout()
    fig.savefig(pathName, bbox_inches='tight')
    print('{} is saved\n'.format(pathName))
        

