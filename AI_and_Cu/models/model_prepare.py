#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:41:17 2023

@author: xiaohui
"""

import sys
sys.path.append("..")
from models.FCNet1D import FCNet1D
from models.ConvLSTM1D import RNN_LSTM
from models.LSTM1D_Classifier import LSTMClassifier, LSTMClassifier_Reg
from models.Transformer import Encoder
from models.Transformer_Classifier import Encoder_Classifier
from models.ResNet1D import ResNet
from models.ResNet1D_Classifier import ResNet_Classifier

def load_model(model_name, device, feature_channel, output_channel, signal_length, \
               dropout, num_class = 2, output_channel_cf = 1):

    if model_name == "FC":
        model = FCNet1D(feature_channel=feature_channel,
                        output_channel=output_channel,
                        hidden_number=10,
                        hidden_size=200,
                        # hidden_number=3,
                        # hidden_size=50,
                        signal_length=signal_length,
                        dim_add=0
                        )
    elif model_name == "FC_M":
        model = FCNet1D(feature_channel=feature_channel,
                        output_channel=4,
                        hidden_number=5,
                        hidden_size=30,
                        # hidden_number=3,
                        # hidden_size=50,
                        signal_length=signal_length,
                        dim_add=0
                        )

    elif "RESClassifier" in model_name:
        if len(model_name.split('_')) == 1:
            num_l = 10
            num_c = 128
            
        else:
            num_l = int(model_name.split('_')[1]) #default 10
            num_c = int(model_name.split('_')[2]) #default 128
            
        model = ResNet_Classifier([num_l], feature_channel=feature_channel,
                       output_channel=output_channel, signal_length=signal_length, \
                       intermediate_channel=num_c, num_class = num_class)

    elif "RES" in model_name:
        model = ResNet([10], feature_channel=feature_channel,
                       output_channel=output_channel, intermediate_channel=128)
                
    elif "LSTMClassifier_Reg" in model_name:
        if model_name == "LSTMClassifier_Reg":
            model = LSTMClassifier_Reg(feature_channel=feature_channel, \
            output_channel_cf=output_channel_cf, output_channel=output_channel, \
            hidden_size=96, num_layers=5, dropout = dropout, num_class = num_class)
                
        else:
            num_h = int(model_name.split('_')[2])
            num_l = int(model_name.split('_')[3])
            
            model = LSTMClassifier_Reg(feature_channel=feature_channel, \
            output_channel_cf=output_channel_cf, output_channel=output_channel, \
            signal_length = signal_length, hidden_size=num_h, num_layers=num_l, \
            dropout = dropout)   
                
    elif "LSTMClassifier" in model_name:
        if model_name == "LSTMClassifier":
            model = LSTMClassifier(feature_channel=feature_channel, output_channel=output_channel_cf, \
            hidden_size=96, num_layers=5, dropout = dropout, num_class = num_class)
                
        else:
            num_h = int(model_name.split('_')[1])
            num_l = int(model_name.split('_')[2])
            
            model = LSTMClassifier(feature_channel=feature_channel, output_channel=output_channel_cf, \
            signal_length = signal_length, hidden_size=num_h, num_layers=num_l, \
            dropout = dropout)                   
                
    elif "LSTM" in model_name:
        if model_name == "LSTM":
            model = RNN_LSTM(feature_channel=feature_channel, output_channel=output_channel, \
                             hidden_size=96, num_layers=5, dropout = dropout)
                
        else:
            num_h = int(model_name.split('_')[1])
            num_l = int(model_name.split('_')[2])
            
            model = RNN_LSTM(feature_channel=feature_channel, output_channel=output_channel, \
                             hidden_size=num_h, num_layers=num_l)                

    elif "Transformer_Classifier" in model_name:

        if model_name == "Transformer_Classifier":
            
            model = Encoder_Classifier(feature_channel=feature_channel,
                            output_channel=output_channel,
                            embed_size=128,
                            num_layers=7,
                            heads=1,
                            forward_expansion=1,
                            seq_length=signal_length,
                            dropout=dropout, 
                            num_class = num_class)                        
        
        else:
            num_embed = int(model_name.split('_')[2])
            num_l = int(model_name.split('_')[3])  
            
            model = Encoder_Classifier(feature_channel=feature_channel,
                            output_channel=output_channel,
                            embed_size=num_embed,
                            num_layers=num_l,
                            heads=1,
                            forward_expansion=1,
                            seq_length=signal_length,
                            dropout=dropout, 
                            num_class = num_class)            
        
    elif model_name == "Transformer":

        model = Encoder(feature_channel=feature_channel,
                        output_channel=output_channel,
                        embed_size=128,
                        num_layers=7,
                        heads=1,
                        forward_expansion=1,
                        seq_length=signal_length,
                        dropout=dropout)
        
    else:
        raise Exception('not implemented model : ' + model_name)

    return model





