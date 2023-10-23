#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:21:45 2023

@author: xiaohui
"""

# 做一个空间分布比较的图，不同的row代表不同变量，不同column代表不同的预报时刻

import os

import datetime
import netCDF4 as nc
import  numpy as np
import matplotlib.pyplot as plt
import shapefile
# from mpl_toolkits.basemap import Basemap

# font1 = {'family'  : 'Times New Roman Bold',
#           'weight' : 'normal',
#           'size'   :  25 }
font1 = {'weight' : 'normal',
          'size'   :  30 }

colormap = ''
# colormap = 'Reds'
colormap = 'jet'
colormap_diff = 'jet'

#%%
#长江流域shp文件
shape_file = './CHN_province.shp'
this_shapefile = shapefile.Reader(shape_file) # whichever file
points = {}
intervals = {}
for j in range(32):
    shape_temp = this_shapefile.shape(j) # whichever shape
    points[j] = np.array(shape_temp.points)
    intervals[j] = list(shape_temp.parts) + [len(shape_temp.points)] 

#%%

# dx = '5km'
dx = '10km'

if dx == '5km':
    projectName = ['5kmbase', '5km_ml']
else:
    projectName = ['10kmbase', '10km_ml']

ListofVar = ['RAINC', 'RAINNC', 'T2']
Var_ticks = {'RAINC': list(np.arange(0, 35, 5)), \
             'RAINNC': list(np.arange(0, 200, 40)), \
             'T2': list(np.arange(290, 310, 5))}
Var_ticks_error = {'RAINC': list(np.arange(-15, 20, 5)), \
                   'RAINNC': list(np.arange(-60, 80, 20)), \
                   'T2': list(np.arange(-2, 3, 1))}
    
VarName_accum = ['RAINC', 'RAINNC']
ListofVar_label = {'RAINC': '12 hour Accumulated RAINC', \
                   'RAINNC': '12 hour Accumulated RAINNC', \
                   'T2': 'Temperature at 2 meter'}

ListofVar_label = {'RAINC': '12 hour RAINC', \
                   'RAINNC': '12 hour RAINNC', \
                   'T2': 'T2M'}
    
unit = {'RAINC': 'mm', 'RAINNC': 'mm', 'T2': 'K'}

startTime_list = '2022061212'
endTime_list = '2022061812'

start_hour = 12
end_hour = 36

hour_interval = 12

startTime = [str(i) for i in startTime_list.split(',')] 
endTime = [str(i) for i in endTime_list.split(',')] 

dirName = '/home/data2/RUN/MSKF/workdir'
dirName_plot = '/home/yux/Code/Python/AI_and_Cu/Check_results/Plot'
        
#%%

time_interval = datetime.timedelta(hours = 1)
#day_interval = datetime.timedelta(days = 1)
day_interval = datetime.timedelta(days = 2)

forecast_horizon = np.arange(start_hour, end_hour + hour_interval, hour_interval)      
    
time_start = []
time_end = []
for m in range(len(startTime)):
    time_start.append( datetime.datetime.strptime(startTime[m], "%Y%m%d%H") )
    time_end.append( datetime.datetime.strptime(endTime[m], "%Y%m%d%H") )

#Get the number of days between time_start and time_end    
day_len = []
for m in range(len(time_start)):
    day_len.append( int((time_end[m] - time_start[m]).total_seconds() / day_interval.total_seconds()) + 1 )
    
initTime = [] #The initialization time for each of the forecast to be evaluated.
initTime_str = []
timeRange = []
timeRange_day = {}
count = 0
for m in range(len(day_len)):
    for t in range(day_len[m]):
        initTime.append(time_start[m] + day_interval*t)
        initTime_str.append(initTime[-1].strftime("%Y%m%d%H"))
        
        timeRange_day[count] = []
        for n in forecast_horizon:
            timeRange.append(time_start[m] + day_interval*t + time_interval*n)
            timeRange_day[count].append(time_start[m] + day_interval*t + time_interval*n) 
                   
        timeRange_day[count] = np.array(timeRange_day[count])

        count = count + 1
        
#%%

def Read_NWP_gridded_data(initTime_str_temp, timeRange_temp):

#==============================================================================
# Read NWP data
#==============================================================================
            
    
    #Var_NWP is NWP prediction.
    Var_NWP = {}                                            
    Var_NWP_diff = {}                                            
    for varName in ListofVar:
        Var_NWP[varName] = {}
        Var_NWP_diff[varName] = {}
        
        for n in range(len(projectName)):
                                        
            print('==============================================================================')
            print('Reading ' + projectName[n] + ' ' + varName + ' data: \n')                                                               
                          
            Var_NWP[varName][n] = np.zeros( (lat[dx].shape[0], lat[dx].shape[1], len(timeRange_temp)) )
            Var_NWP_diff[varName][n] = []
            
            for t, time in enumerate(timeRange_temp):
            
                fileName = 'wrfout_d01_' + time.strftime("%Y-%m-%d_%H:%M:%M")
                pathName = os.path.join(dirName, initTime_str_temp, projectName[n], 'WRFV4', fileName)
                
                file = nc.Dataset(pathName)
                
                Var_temp = np.array(file[varName][0, ::])
                
                if varName in VarName_accum:
                    fileName_previous = 'wrfout_d01_' + (time - time_interval*hour_interval).strftime("%Y-%m-%d_%H:%M:%M")
                    pathName_previous = os.path.join(dirName, initTime_str_temp, projectName[n], 'WRFV4', fileName_previous)
                    
                    file_previous = nc.Dataset(pathName_previous)                    
                    Var_temp_previous = np.array(file_previous[varName][0, ::])                
                
                    Var_temp = Var_temp - Var_temp_previous
                            
                Var_NWP[varName][n][:, :, t] = Var_temp.copy()
            
            if n > 0:
                Var_NWP_diff[varName][n] = Var_NWP[varName][n] - Var_NWP[varName][0]

    return Var_NWP, Var_NWP_diff

#%%    
def Plot_NWP_gridded_data(Var_NWP, Var_NWP_diff, initTime_str_temp, timeRange_temp, pathName_plot):

    for n in np.arange(1, len(projectName)):
    
        fig, axs = plt.subplots(len(ListofVar) * 2, len(timeRange_temp), figsize=(12.5*len(timeRange_temp), len(ListofVar)*8.5 * 2))
    
        for i, varName in enumerate(ListofVar):
            
            for m in range(2):
                #Plot gt and difference respectively.
                
                for t, time in enumerate(timeRange_temp):
        
                    index_ax = (i*2 + m, t)
                
                    title_temp = ''
                    
                    for j in range(32):
                        for (m1, n1) in zip(intervals[j][:-1], intervals[j][1:]):
                            axs[index_ax].plot(*zip(*points[j][m1:n1]), 'k')
                                        
                    if m == 0:
                        Var_plot = Var_NWP[varName][0][:, :, t]                        
                        
                        if i == 0:
                            title_temp = '{} hour forecast'.format(forecast_horizon[t])                            
                            
                    elif m == 1:
                        #Plot difference        
                        Var_plot = Var_NWP_diff[varName][n][:, :, t]
                        
                        MAD = np.mean(np.abs(Var_NWP_diff[varName][n][::, t]))
                        MAD_str = str(round(MAD, 3))
                                
                        title_temp = 'MAD {}'.format(MAD_str)                    
                        
                    axs[index_ax].set_title(title_temp, font = font1)                                                              
                    
                    # fig_plot = axs[index_ax].pcolor(lon, np.flip(lat, axis = 0), np.flip(Var_plot, axis = 0))
                    if colormap != '':
                        if m == 0:
                            fig_plot = axs[index_ax].pcolor(lon[dx], lat[dx], Var_plot, cmap = colormap)
                            
                        elif m == 1:
                            fig_plot = axs[index_ax].pcolor(lon[dx], lat[dx], Var_plot, cmap = colormap_diff)
                            
                    else:
                        fig_plot = axs[index_ax].pcolor(lon[dx], lat[dx], Var_plot)
                        
                    axs[index_ax].set_xlim(lon[dx].min(), lon[dx].max())        
                    axs[index_ax].set_ylim(lat[dx].min(), lat[dx].max())
                    
                    if (i == (len(ListofVar) - 1)) & (m==1):      
                        axs[index_ax].set_xlabel("longitude", font = font1)                
                    else:
                        axs[index_ax].set_xticklabels([])  
                            
                    if t == 0:  
                        if m == 0:
                            axs[index_ax].set_ylabel("{}\nlatitude".format(ListofVar_label[varName]), font = font1)   
                        elif m == 1:
                            axs[index_ax].set_ylabel("{}\nlatitude".format(ListofVar_label[varName] + ' Diff'), font = font1)                               

                    else:
                        axs[index_ax].set_yticklabels([])  
                            
                        # axs[index_ax].set_ylabel(title_list[i] + "\nlatitude", font = font1)                        
                        # plt.ylabel("latitude", font = font1, fontsize = 25)

                    if m == 0:                           
                        if varName in Var_ticks.keys():
                            fig_plot.set_clim(Var_ticks[varName][0], Var_ticks[varName][-1])    
                            
                    elif m == 1:
                        if varName in Var_ticks_error.keys():
                            fig_plot.set_clim(Var_ticks_error[varName][0], Var_ticks_error[varName][-1])    
                                                    
                    axs[index_ax].tick_params(axis = 'both', labelsize = 25) 
                                         
                    ###
                    #Test
                
                plt.subplots_adjust(right=0.9)
                
                chartBox = axs[index_ax].get_position()
                
                cax = plt.axes([0.92, chartBox.y0, 0.015, chartBox.height])
                
                if m == 0:
                    
                    if varName in Var_ticks.keys():                        
                        clb = fig.colorbar(fig_plot, cax = cax, orientation='vertical', 
                        ticks = Var_ticks[varName])        
                        
                    else:
                        clb = fig.colorbar(fig_plot, cax = cax, orientation='vertical')
                     
                elif m == 1:
                    
                    if varName in Var_ticks_error.keys():
                        # clb = fig.colorbar(fig_plot, orientation = 'vertical', 
                        # label = varName)
                        
                        # clb = fig.colorbar(fig_plot, orientation='vertical', 
                        # label = varName, ticks = Var_ticks_error[varName])   
                        
                        clb = fig.colorbar(fig_plot, cax = cax, orientation='vertical', 
                        ticks = Var_ticks_error[varName])        
                        
                    else:
                        clb = fig.colorbar(fig_plot, cax = cax, orientation='vertical')
                  
                    ###
                    
                    # clb = fig.colorbar(fig_plot, ax = axs[index_ax], orientation='vertical', 
                    #         location = "right")
                
                clb.set_label(unit[varName], fontsize = 25, font = font1) 
                              
                clb.ax.tick_params(labelsize=25)                                                            
                
        # plt.tight_layout()
        fig.savefig(pathName_plot, bbox_inches='tight')
        print(pathName_plot, ' is saved\n')
        
#%%
lat = {}
lon = {}    
fileName_geog = 'lonlathgt_{}.npz'.format(dx)
geog = np.load(os.path.join(dirName, fileName_geog))
lat[dx] = geog['lat']
lon[dx] = geog['lon']

Var_NWP_avg = {}
Var_NWP_diff_avg = {}
for varName in ListofVar:
    Var_NWP_avg[varName] = {}
    Var_NWP_diff_avg[varName] = {}
    
    for n in range(len(projectName)):
        Var_NWP_avg[varName][n] = []
        Var_NWP_diff_avg[varName][n] = []
        
for t in range(len(initTime_str)):
    
    pathName_plot = os.path.join(dirName_plot, 'pcolor_{}_{}.png'.format(dx, initTime_str[t]))
    
    Var_NWP_temp, Var_NWP_diff_temp = Read_NWP_gridded_data(initTime_str[t], timeRange_day[t])
    
    Plot_NWP_gridded_data(Var_NWP_temp, Var_NWP_diff_temp, initTime_str[t], timeRange_day[t], pathName_plot)
    
    for varName in ListofVar:            
        for n in range(len(projectName)):
            if t == 0:            
                Var_NWP_avg[varName][n] = Var_NWP_temp[varName][n]
                
                if n > 0:
                    Var_NWP_diff_avg[varName][n] = Var_NWP_diff_temp[varName][n]
    
            else:
                Var_NWP_avg[varName][n] = Var_NWP_avg[varName][n] + Var_NWP_temp[varName][n]
                
                if n > 0:                
                    Var_NWP_diff_avg[varName][n] = Var_NWP_diff_avg[varName][n] + Var_NWP_diff_temp[varName][n]
    
#Calculate average over all the initTime_str:
for varName in ListofVar:            
    for n in range(len(projectName)):
        Var_NWP_avg[varName][n] = Var_NWP_avg[varName][n]/len(initTime_str)
        
        if n > 0:                        
            Var_NWP_diff_avg[varName][n] = Var_NWP_diff_avg[varName][n]/len(initTime_str)
    
pathName_plot = os.path.join(dirName_plot, 'pcolor_{}_{}_{}_avg.png'.format(dx, initTime_str[0], initTime_str[-1]))
Plot_NWP_gridded_data(Var_NWP_avg, Var_NWP_diff_avg, initTime_str[t], timeRange_day[t], pathName_plot)    
    
