#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 10:50:48 2023

@author: Xiaohui
"""

#%%
from datetime import datetime
from datetime import timedelta
import os
import numpy as np
from config_wrf import config_wrf

def plustime(start_time_str, addminus = 30, dateformat = '%Y%m%d_%H%M%S'):
    """
    start_time_str: "20200104_120000" 
    addminus = 30
    end_time_str : "20200104_123000" 
    """
    start_time = datetime.strptime(start_time_str, dateformat)
    final_time = start_time + timedelta(minutes = addminus)
    end_time_str = final_time.strftime(dateformat)
    return end_time_str


#%%

startTime = ['2022052012'] 
# endTime = ['2022061812'] 
# endTime = ['2022052012'] 
# endTime = ['2022052612'] 
endTime = ['2022060512'] 

time_start = []
time_end = []
for m in range(len(startTime)):
    time_start.append( datetime.strptime(startTime[m], "%Y%m%d%H") )
    time_end.append( datetime.strptime(endTime[m], "%Y%m%d%H") )

# day_interval = timedelta(days = 1)
day_interval = timedelta(days = 2)

#Get the number of days between time_start and time_end    
day_len = []
for m in range(len(time_start)):
    day_len.append( int((time_end[m] - time_start[m]).total_seconds() / day_interval.total_seconds()) + 1 )
        
dateformat = '%Y%m%d%H'
start_list = []
for m in range(len(startTime)):
    time_start = startTime[m]
    
    for n in range(day_len[m]):
        time_str = plustime(time_start, 60*24*2*n, dateformat)
        start_list.append( time_str )

# start_list = ["2022052012"]

#wrf output interval, unit is minute
# output_interval = 30
output_interval = 5

#%%

#The first 48 files, i.e. 24 hours are used for training.
pathName_train_dict_in = 'train_dict_in.npy'
pathName_train_dict_out = 'train_dict_out.npy'
if (os.path.isfile(pathName_train_dict_in) == 0) & \
   (os.path.isfile(pathName_train_dict_in) == 0):
    
    num_train = config_wrf.num_train
    train_file_dict_in = {}
    train_file_dict_out = {}
    counter = 0
    for start_list_detail in  start_list:
        print('Process data for day',start_list_detail)
        for i in range(num_train):
            end_time_str = plustime(start_list_detail[0:8] + '_' + \
                                    start_list_detail[8:10] + '0000', output_interval* i)                      
                                    # start_list_detail[8:10] + '0000', output_interval* (i+1))
            
            print(end_time_str)
            
            file_name_in = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_input.npz")
            
            file_name_out = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_output.npz")
            
            if (os.path.isfile(file_name_in) != 0) & \
               (os.path.isfile(file_name_out) != 0):
                train_file_dict_in[counter] = file_name_in
                train_file_dict_out[counter] = file_name_out
                counter += 1
            
    print(train_file_dict_in)

    np.save(pathName_train_dict_in, train_file_dict_in) 
    np.save(pathName_train_dict_out, train_file_dict_out) 

#The later 24 files, i.e. 12 hours are used for testing.
pathName_test_dict_in = 'test_dict_in.npy'
pathName_test_dict_out = 'test_dict_out.npy'
if (os.path.isfile(pathName_test_dict_in) == 0) & \
   (os.path.isfile(pathName_test_dict_in) == 0):
    
    num_test =  config_wrf.num_test
    test_file_dict_in = {}
    test_file_dict_out = {}
    counter = 0
    for start_list_detail in  start_list:
        for i in range(num_test):
            end_time_str = plustime(start_list_detail[0:8] + '_' + \
                                    start_list_detail[8:10] + '0000', output_interval* (i+num_train))
                                    # start_list_detail[8:10] + '0000', output_interval* (i+1+num_train))
                
                
            print(end_time_str)
            file_name_in = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_input.npz")
            
            file_name_out = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_output.npz")   
            
            if (os.path.isfile(file_name_in) != 0) & \
               (os.path.isfile(file_name_out) != 0):
                test_file_dict_in[counter] = file_name_in
                test_file_dict_out[counter] = file_name_out
                counter += 1
            
    print(test_file_dict_in)
                                 
    np.save(pathName_test_dict_in, test_file_dict_in) 
    np.save(pathName_test_dict_out, test_file_dict_out) 

#%%
#Generate files for debuging code.

#The first 48 files, i.e. 24 hours are used for training.
pathName_train_dict_in = 'train_dict_in_test.npy'
pathName_train_dict_out = 'train_dict_out_test.npy'

if (os.path.isfile(pathName_train_dict_in) == 0) & \
   (os.path.isfile(pathName_train_dict_in) == 0):
       
    num_train = 2       
    train_file_dict_in = {}
    train_file_dict_out = {}
    counter = 0
    for start_list_detail in  start_list:
        for i in range(num_train):
            end_time_str = plustime(start_list_detail[0:8] + '_' + \
                                    start_list_detail[8:10] + '0000', output_interval* (i))
                                    # start_list_detail[8:10] + '0000', output_interval* (i+1))
            
            print(end_time_str)
            
            file_name_in = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_input.npz")
            
            file_name_out = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_output.npz")
            
            if (os.path.isfile(file_name_in) != 0) & \
               (os.path.isfile(file_name_out) != 0):
                train_file_dict_in[counter] = file_name_in
                train_file_dict_out[counter] = file_name_out
                counter += 1
            
    print(train_file_dict_in)

    np.save(pathName_train_dict_in, train_file_dict_in) 
    np.save(pathName_train_dict_out, train_file_dict_out) 

#The later 24 files, i.e. 12 hours are used for testing.
pathName_test_dict_in = 'test_dict_in_test.npy'
pathName_test_dict_out = 'test_dict_out_test.npy'

if (os.path.isfile(pathName_test_dict_in) == 0) & \
   (os.path.isfile(pathName_test_dict_in) == 0):
    
    num_test =  1
    test_file_dict_in = {}
    test_file_dict_out = {}
    counter = 0
    for start_list_detail in  start_list:
        for i in range(num_test):
            end_time_str = plustime(start_list_detail[0:8] + '_' + \
                                    start_list_detail[8:10] + '0000', output_interval* (i+num_train))
                                    # start_list_detail[8:10] + '0000', output_interval* (i+1+num_train))

            print(end_time_str)
            file_name_in = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_input.npz")
            
            file_name_out = os.path.join(config_wrf.dirName_data, 
                                     start_list_detail,
                                     end_time_str + "_output.npz")   
            
            if (os.path.isfile(file_name_in) != 0) & \
               (os.path.isfile(file_name_out) != 0):
                test_file_dict_in[counter] = file_name_in
                test_file_dict_out[counter] = file_name_out
                counter += 1
            
    print(test_file_dict_in)
                                 
    np.save(pathName_test_dict_in, test_file_dict_in) 
    np.save(pathName_test_dict_out, test_file_dict_out) 
