#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 14:54:07 2021

@author: xiaohui
"""

import sys
import os

from pyparsing import col
import torch
import torch.nn as nn
os.environ['HDF5_DISABLE_VERSION_CHECK']='2'
from models.model_prepare import load_model
from utils.model_helper import ModelUtils
from utils.file_helper import FileUtils
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import numpy as np
import argparse
import logging
from utils.make_wrfCu_data import WrfCuDataset
from utils.config_norm import norm_mapping
from utils.config_wrf import config_wrf
from utils.cu_utils import mskf_constraint, trigger_consistency, nca_constraint

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.evaluate_helper import unnormalized, check_accuracy

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train the Cumulus parameterization')

    parser.add_argument('--main_folder', type=str, default='temp')
    parser.add_argument('--sub_folder', type=str, default='temp')
    parser.add_argument('--prefix', type=str, default='temp')

    parser.add_argument('--dataset_type', type=str, default='WRF')
    parser.add_argument('--loss_type', type=str, default='v01')
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch_size')
    parser.add_argument('--model_name', type=str, default="LSTM",
                        help='model name')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='dropout rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='batch_size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='num_epochs')

    parser.add_argument('--save_model', choices=('True', 'False'))
    parser.add_argument('--save_checkpoint_name',
                        type=str, default="test.pth.tar")
    parser.add_argument('--save_per_samples', type=int, default=10000)
    parser.add_argument('--load_model', choices=('True', 'False'))
    parser.add_argument('--load_checkpoint_name',
                        type=str, default="test.pth.tar")
    parser.add_argument('--resume_epoch',
                        type=int, default=-1)
    parser.add_argument('--evaluate', choices=('True', 'False'), default = "False")    
    parser.add_argument('--random_throw', choices=('True', 'False'),default = "False")

    parser.add_argument('--gpu', type = str, default = "0")
    parser.add_argument('--num_gpu', type = int, default = 1)
    
    #specify label variable 
    parser.add_argument('--label_variable', type=str, default='')
    
    args = parser.parse_args()
    return args
                          
args = parse_args()

if args.label_variable != '':    
    
    config_wrf.label_single_height_variable = []
    config_wrf.label_multi_height_variable = []
    if args.label_variable in config_wrf.label_single_height_variable_possible:
        config_wrf.label_single_height_variable = [args.label_variable]
    
    else:
        config_wrf.label_multi_height_variable = [args.label_variable]
        
    config_wrf.label_all_variable = config_wrf.label_single_height_variable + \
                                    config_wrf.label_multi_height_variable
            
    config_wrf.output_channel = len(config_wrf.label_all_variable_reg)
            
    if 'trigger' in config_wrf.label_all_variable:
        config_wrf.num_class = 2
        # config_wrf.num_class = 3
        
        if len(config_wrf.label_all_variable) == 1:
            config_wrf.error_train = ['BCE']
            
            # config_wrf.error_train = ['CrossEntropyLoss']
        else:
            config_wrf.error_train = ['BCE' , 'mse']    
            # config_wrf.error_train = ['BCE' , 'mse_trigger']    
                
        config_wrf.Listof_error_class = config_wrf.error_train + ['accuracy']        
    
FileUtils.makedir(os.path.join("logs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("results", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(config_wrf.dirName_data, "../model_outputs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join("runs", args.main_folder, args.sub_folder))
FileUtils.makedir(os.path.join(
    "checkpoints", args.main_folder, args.sub_folder))
        
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

if(args.random_throw == "True"):
    args.random_throw_boolean = True
else:
    args.random_throw_boolean = False

# set logging
filehandler = logging.FileHandler(os.path.join(
"logs", args.main_folder, args.sub_folder, args.prefix + "log.txt"))
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logging.getLogger('matplotlib.font_manager').disabled = True
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# random variable
random_state = 0
np.random.seed(random_state)
torch.manual_seed(random_state)
torch.set_printoptions(precision=5)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:" + args.gpu[0] if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:" + args.gpu[0])
    
writer = SummaryWriter(f"runs/{args.main_folder}/{args.sub_folder}/")

step = 0

print(f"args.dataset_type:{args.dataset_type}")

if (args.dataset_type == "WRF"):
    train_dataset =  WrfCuDataset(vertical_layers = config_wrf.vertical_layers, 
    type = "train", norm_mapping=norm_mapping, time_length = config_wrf.time_length)
    
    test_dataset = WrfCuDataset(vertical_layers = config_wrf.vertical_layers, 
    type = "test", norm_mapping=norm_mapping, time_length = config_wrf.time_length, \
    flag_get_stat = 0, evaluate = "True")

if (args.dataset_type == "WRF"):
    train_loader = DataLoader(
    dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=False )

    # test_loader = DataLoader(
    # dataset=test_dataset, batch_size=args.batch_size, shuffle=True,
    # num_workers=args.num_workers, pin_memory=False )

    test_loader = DataLoader(
    dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_workers, pin_memory=False )
    
logger.info(
    f"train size:{len(train_dataset)}, test size:{len(test_dataset)}")

if(args.dataset_type == "WRF"):
    model = load_model(args.model_name, device,
    feature_channel=config_wrf.num_all_feature, output_channel = config_wrf.output_channel, \
    signal_length = config_wrf.vertical_layers, dropout = args.dropout, \
    num_class = config_wrf.num_class, output_channel_cf = config_wrf.output_channel_cf)
    
model_info = ModelUtils.get_parameter_number(model)
logger.info(model_info)

criterion_ce = nn.CrossEntropyLoss()
# criterion_be = nn.BCEWithLogitsLoss()
criterion_be = nn.BCELoss()
criterion_mse = nn.MSELoss()
criterion_mae = nn.L1Loss()

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Define Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
optimizer, factor=0.5, patience=5, verbose=True
)

if(args.load_model == "True"):
    pathName_model = os.path.join("checkpoints",
    args.main_folder, args.sub_folder, args.load_checkpoint_name)
    
    print('Load model {}\n'.format(pathName_model))
    
    ModelUtils.load_checkpoint(torch.load(pathName_model, map_location=device),
    model, optimizer, device)
    
# if torch.cuda.is_available():
#     model.cuda()
    
#Here, we use multiple gpu
if args.num_gpu > 1:
    print("Let's use ", str(args.num_gpu), "GPUs!")
    model = nn.DataParallel(model)
    # torch.backends.cudnn.benchmark = True
    
model.to(device)
    
logger.info("start training...")
# Train Network

"""
#for testing the dependence of running time on num_workers

def optimize_num_worker(dataset, batch_size):
    from time import time
    import multiprocessing as mp
    for num_workers in range(2, mp.cpu_count(), 2):  
    #for num_workers in [50]:
        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True )

        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    return  0

optimize_num_worker(train_dataset,args.batch_size)
exit()
"""


previous_time = time.time()

if args.evaluate == "True":
    
    epoch = args.resume_epoch
    
    target_norm_info = None
    
    if (args.dataset_type == "WRF"):
        Var_error_test_unorm, Var_error_test = check_accuracy(
        test_loader, model, norm_mapping, device, config_wrf, \
        args, epoch, config_wrf.if_plot, config_wrf.if_plot_2d, target_norm_info)         
            
    for variable_name in config_wrf.label_all_variable:
        
        if variable_name in config_wrf.label_classification_variable_possible:
            
            for error in config_wrf.Listof_error_class:                   
                logger.info(f"Epoch {epoch} Testing {variable_name} unormalized {error} :{Var_error_test_unorm[variable_name][error]: .3e}")
                                
        else:
            
            for error in config_wrf.Listof_error:                   
                logger.info(f"Epoch {epoch} Testing {variable_name} unormalized {error} :{Var_error_test_unorm[variable_name][error]: .3e}")
                logger.info(f"Epoch {epoch} Testing {variable_name} unormalized {error} :{Var_error_test[variable_name][error]: .3e}")
                    
    print()
            
else:
    
    print('Epochs: ',np.arange(args.resume_epoch+1, args.num_epochs))
    for epoch in np.arange(args.resume_epoch+1, args.num_epochs):
    
        loop = tqdm(train_loader)
        model.train()
    
        Var_error_train_sum = {} #normalized error
        Var_error_train_sum_unit = {} #unormalized error
        for variable_name in config_wrf.label_all_variable:
            Var_error_train_sum[variable_name] = 0.0    
            Var_error_train_sum_unit[variable_name] = 0.0    
    
        num_samples = 0
        schedule_losses = []
    
        logger.info(f"epoch:{epoch}, elapse time:{time.time() - previous_time}")
        previous_time = time.time()
        
        for batch_idx, (feature, targets, filters, time_str) in enumerate(train_loader):
            # if(batch_idx > 2):
            #     break
            logger.info(f"Epoch {epoch}, batch {batch_idx}" )
    
            if(epoch == 0 and batch_idx == 0):
                logger.info(f"feature shape:{feature.shape}, target shape:{targets.shape}, filters shape:{filters.shape}" )
                    
            feature_shape = feature.shape
            target_shape = targets.shape
            filters_shape = filters.shape
            # Get data to cuda if possible
    
            if args.dataset_type == "WRF":
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
    
            if config_wrf.flag_filter == 1:
                #filter prediction based on specified filter variable.
                #teacher forcing, using 100% ground truth filter.
                if len(predicts) > 0:
                    if len(predicts.shape) == 3:                
                        for i in range(predicts.shape[1]):
                            predicts[:, i, :] = predicts[:, i, :] * filters[:, 0, :].to(device=device)
                        
                    elif len(predicts.shape) == 2:
                        for i in range(predicts.shape[1]):
                            predicts[:, i] = predicts[:, i] * filters[:, 0, 0].to(device=device)      
                                            
            # print("Outside: input size", feature.size(), \
            #       "output_size", predicts.size())
                            
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
                
            #No need to do this for training                                            
            # #Ensure concistency between all predicted variables
            # if config_wrf.flag_trigger_consistency == 1:   
            #     if len(predicts) != 0:                             
            #         predicts = trigger_consistency(predicts, predicts_cf, config_wrf)
                            
            # Unnormalize
            if(args.dataset_type == "WRF"):
                if len(predicts) != 0:    
                    predicts_unnorm, targets_unnorm = unnormalized(
                    predicts, targets, norm_mapping, config_wrf.label_all_variable_cf, \
                    config_wrf.label_all_variable_reg, config_wrf.norm_method)
            
                    #Ensure nca to be integer
                    predicts_unnorm = nca_constraint(predicts_unnorm, config_wrf.label_all_variable_reg)
                    
            if(args.loss_type == "v01"):
                """
                v01 loss
                """
                
                total_loss = 0
                loss = {}
                loss_unit = {} #train set loss with unit
                for variable_index, variable_name in enumerate(config_wrf.label_all_variable_cf):
                    
                    # single height label
                    if variable_name in config_wrf.label_single_height_variable:  
                                      
                        if 'CrossEntropyLoss' in config_wrf.error_train:
                            loss[variable_name] = criterion_ce(predicts_cf[:,variable_index], \
                                                               targets[:,variable_index,0])                             
                        
                        elif 'BCE' in config_wrf.error_train:
                            # print('1 predicts_cf shape {}\n'.format(predicts_cf.shape))
                            # print('1 predicts shape {}\n'.format(predicts.shape))
                            # print('targets shape {}\n'.format(targets[:,variable_index,0].shape))
                            loss[variable_name] = criterion_be(predicts_cf[:,variable_index], \
                                                               targets[:,variable_index,0])                             

                    total_loss = total_loss + loss[variable_name] * config_wrf.weight_loss[variable_name]

                for variable_index, variable_name in enumerate(config_wrf.label_all_variable_reg):
                    
                    variable_index_target = variable_index + len(config_wrf.label_all_variable_cf)

                    # single height label
                    if variable_name in config_wrf.label_single_height_variable:  
                                                
                        if 'mse_trigger' in config_wrf.error_train:
                            loss[variable_name] = criterion_mse(predicts[trigger_index,variable_index,0], \
                                                                targets[trigger_index,variable_index_target,0])                    
                            
                            loss_unit[variable_name]  = criterion_mse(predicts_unnorm[trigger_index,variable_index,0], \
                                                                      targets_unnorm[trigger_index,variable_index_target,0])
                            
                        elif 'mse' in config_wrf.error_train:
                            loss[variable_name] = criterion_mse(predicts[:,variable_index,0], \
                                                                targets[:,variable_index_target,0])                    
                            
                            loss_unit[variable_name]  = criterion_mse(predicts_unnorm[:,variable_index,0], \
                                                                      targets_unnorm[:,variable_index_target,0])

                        elif 'mae_trigger' in config_wrf.error_train:
                            loss[variable_name] = criterion_mae(predicts[trigger_index,variable_index,0], \
                                                                targets[trigger_index,variable_index_target,0])                    
                            
                            loss_unit[variable_name]  = criterion_mae(predicts_unnorm[trigger_index,variable_index,0], \
                                                                      targets_unnorm[trigger_index,variable_index_target,0])
                                              
                        elif 'mae' in config_wrf.error_train:
                            loss[variable_name] = criterion_mae(predicts[:,variable_index,0], \
                                                                targets[:,variable_index_target,0])                    
                            
                            loss_unit[variable_name]  = criterion_mae(predicts_unnorm[:,variable_index,0], \
                                                                      targets_unnorm[:,variable_index_target,0])
                                                                

                    # multi height label                    
                    elif variable_name in config_wrf.label_multi_height_variable:  
                        if 'mse_trigger' in config_wrf.error_train:
                            loss[variable_name] = criterion_mse(predicts[trigger_index,variable_index,:], \
                                                                targets[trigger_index,variable_index_target,:])   
               
                            loss_unit[variable_name] = criterion_mse(predicts_unnorm[trigger_index,variable_index,:], \
                                                                      targets_unnorm[trigger_index,variable_index_target,:])
                        
                        elif 'mse' in config_wrf.error_train:
                            loss[variable_name] = criterion_mse(predicts[:,variable_index,:], \
                                                                targets[:,variable_index_target,:])                                        
    
                            loss_unit[variable_name] = criterion_mse(predicts_unnorm[:,variable_index,:], \
                                                                      targets_unnorm[:,variable_index_target,:])
                 
                        elif 'mae_trigger' in config_wrf.error_train:
                            loss[variable_name] = criterion_mae(predicts[trigger_index,variable_index,:], \
                                                                targets[trigger_index,variable_index_target,:])   
               
                            loss_unit[variable_name] = criterion_mae(predicts_unnorm[trigger_index,variable_index,:], \
                                                                      targets_unnorm[trigger_index,variable_index_target,:])
                                                    
                        elif 'mae' in config_wrf.error_train:
                            loss[variable_name] = criterion_mae(predicts[:,variable_index,:], \
                                                                targets[:,variable_index_target,:])   
               
                            loss_unit[variable_name] = criterion_mae(predicts_unnorm[:,variable_index,:], \
                                                                      targets_unnorm[:,variable_index_target,:])
                    
                    total_loss = total_loss + loss[variable_name] * config_wrf.weight_loss[variable_name]
    
            # print(torch.std(predicts_unnorm), torch.std(targets_unnorm))
            
            num_samples = num_samples + feature_shape[0]
            try:
    
                for variable_name in config_wrf.label_all_variable:                
                    Var_error_train_sum[variable_name] = Var_error_train_sum[variable_name] + \
                    feature_shape[0]*loss[variable_name].item()
            
                    if variable_name not in config_wrf.label_classification_variable_possible:
                        Var_error_train_sum_unit[variable_name] = Var_error_train_sum_unit[variable_name] + \
                        feature_shape[0]*loss_unit[variable_name].item()
                        
            except:
                print("An exception occurred\n")                
                exit()
            
            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
            for variable_name in config_wrf.label_all_variable:
                writer.add_scalar("train " + variable_name, loss[variable_name].item(), global_step=step)
                
            step = step + args.batch_size                    
    
    #    if(epoch % 1 ==0 ):
        if(epoch % 5 ==0 ):
    
            Var_error_train = {}
            Var_error_train_unit = {}
            for variable_name in config_wrf.label_all_variable:
                Var_error_train[variable_name] = Var_error_train_sum[variable_name]/num_samples
                
                if variable_name not in config_wrf.label_classification_variable_possible:
                    Var_error_train_unit[variable_name] = np.sqrt(Var_error_train_sum_unit[variable_name]/num_samples)
    
            if (args.dataset_type == "WRF"):
                target_norm_info = None
                
            if (args.dataset_type == "WRF"):
                Var_error_test_unorm, Var_error_test = check_accuracy(
                test_loader, model, norm_mapping, device, config_wrf, \
                args, epoch, config_wrf.if_plot, config_wrf.if_plot_2d, target_norm_info)
                    
                # Var_error_train_unorm, Var_error_train = check_accuracy(
                # train_loader, model, norm_mapping, device, config_wrf, \
                # args, False, False, target_norm_info)                
        
            loss_test = 0
            loss_train = 0
            for variable_index, variable_name in enumerate(config_wrf.label_all_variable): 
                
                if variable_name in config_wrf.label_classification_variable_possible:
                    error = config_wrf.Listof_error_class[0]
                    loss_test = loss_test + Var_error_test[variable_name][error]
                                    
                else:
                    error = config_wrf.Listof_error[0]
                    loss_test = loss_test + Var_error_test[variable_name][error]
                    
                # loss_train = loss_train + Var_error_train_unorm[variable_name][error]
                
            schedule_losses.append(loss_test)
    
            for variable_name in config_wrf.label_all_variable:             
                if variable_name in config_wrf.label_classification_variable_possible:
                    
                    error_train = config_wrf.error_train[0]                    
                    logger.info(f"Epoch {epoch} Training {variable_name} {error_train}:{Var_error_train[variable_name]: .3e}")        
                    
                else:
                    
                    if len(config_wrf.error_train) > 1:
                        error_train = config_wrf.error_train[1]
                    else:
                        error_train = config_wrf.error_train[0]
                        
                    logger.info(f"Epoch {epoch} Training {variable_name} {error_train}:{Var_error_train[variable_name]: .3e}")        
                    
            print()
    
            # if 'mse' in config_wrf.error_train:
            #     for variable_name in config_wrf.label_all_variable:                
            
            #         logger.info(f"Training {variable_name} unormalized r{config_wrf.error_train}:{Var_error_train_unit[variable_name]: .3e}")        
            #     print()
    
            # for variable_name in config_wrf.label_all_variable:
                
            #     for error in config_wrf.Listof_error:                   
            #         logger.info(f"Epoch {epoch} Training {variable_name} unormalized {error} :{Var_error_train_unorm[variable_name][error]: .3e}")
    
            #     print()
            
            for variable_name in config_wrf.label_all_variable:
                if variable_name in config_wrf.label_classification_variable_possible:
                    
                    for error in config_wrf.Listof_error_class:                   
                        logger.info(f"Epoch {epoch} Testing {variable_name} unormalized {error} :{Var_error_test_unorm[variable_name][error]: .3e}")
                                        
                else:
                    
                    for error in config_wrf.Listof_error:                   
                        logger.info(f"Epoch {epoch} Testing {variable_name} unormalized {error} :{Var_error_test_unorm[variable_name][error]: .3e}")
                        logger.info(f"Epoch {epoch} Testing {variable_name} {error} :{Var_error_test[variable_name][error]: .3e}")
                    
            print()
                
            if (args.save_model == "True"):
                
                if args.num_gpu > 1:
                    checkpoint = {"state_dict": model.module.state_dict(
                    ), "optimizer": optimizer.state_dict()}
                    
                else:
                    checkpoint = {"state_dict": model.state_dict(
                    ), "optimizer": optimizer.state_dict()}
        
                filename = os.path.join("checkpoints", args.main_folder,
                args.sub_folder, args.prefix + str(epoch).zfill(4) + args.save_checkpoint_name + ".pth.tar")
        
                filename_full = os.path.join("checkpoints", args.main_folder,
                args.sub_folder, args.prefix + str(epoch).zfill(4) + args.save_checkpoint_name + ".pth")
        
                ModelUtils.save_checkpoint(
                checkpoint, filename=filename)
                
                if args.num_gpu > 1:
                    torch.save(model.module, filename_full)
                else:
                    torch.save(model, filename_full)    
                    
            mean_loss = sum(schedule_losses) / len(schedule_losses)
            scheduler.step(mean_loss)
            
