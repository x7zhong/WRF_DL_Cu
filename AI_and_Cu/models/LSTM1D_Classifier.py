#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:36:55 2023

@author: xiaohui
"""

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
sys.path.append("..")

class LSTMClassifier(nn.Module):
    def __init__(self, feature_channel, output_channel, hidden_size,
                 signal_length, num_layers, dropout = 0.0, num_class = 2):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.lstm = nn.LSTM(feature_channel, hidden_size,
                            num_layers, batch_first=True, bidirectional=True, \
                            dropout = dropout)
        
        self.conv1d = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)

        self.fc = nn.Linear(signal_length, output_channel)

        self.num_class = num_class
        
        if self.num_class == 2:
            self.pred = torch.nn.Sigmoid()
        elif self.num_class >2:
            self.pred = torch.nn.Softmax()
            
    def forward(self, x):
        # x = torch.permute(x, [0, 2, 1])
        x = x.permute((0, 2, 1))
        # Set initial hidden and cell states

        h0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)
        c0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )
        
        out = out.permute((0, 2, 1))
        out = self.conv1d(out)
        out = torch.squeeze(out)
        
        # print('out shape is {}'.format(out.shape))
        out = self.fc(out)
        # print('out shape is {}'.format(out.shape))
        
        out = self.pred(out)
        
        #Check data distributed using multiple gpu
        # print("\tIn Model: input size", x.size(), \
        #       "output size", out.size())
        
        return out
    
    def predict(self, x):
        
        out = self.forward(x)
                
        threshold = 0.5
        if self.num_class == 2:
            pred = F.sigmoid(out) > threshold
        
        return pred

class LSTMClassifier_Reg(nn.Module):
    def __init__(self, feature_channel, output_channel_cf, output_channel, hidden_size,
                 signal_length, num_layers, dropout = 0.0, num_class = 2):
        super(LSTMClassifier_Reg, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_channel = output_channel
        self.output_channel_cf = output_channel_cf
        self.lstm = nn.LSTM(feature_channel, hidden_size,
                            num_layers, batch_first=True, bidirectional=True, \
                            dropout = dropout)
        
        self.conv1d = nn.Conv1d(
            2 * hidden_size, output_channel_cf, kernel_size=1, padding=0, bias=True)

        self.fc = nn.Linear(signal_length, output_channel_cf)

        self.num_class = num_class
        
        if self.num_class == 2:
            self.pred = torch.nn.Sigmoid()
        elif self.num_class >2:
            self.pred = torch.nn.Softmax()
            
        self.reg_final = nn.Conv1d(
            2 * hidden_size, output_channel, kernel_size=1, padding=0, bias=True)
            
    def forward(self, x):
        # x = torch.permute(x, [0, 2, 1])
        x = x.permute((0, 2, 1))
        # Set initial hidden and cell states

        h0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)
        c0 = torch.zeros(2*self.num_layers, x.shape[0],
                         self.hidden_size, requires_grad=False).to(x.device)

        hidden = (h0, c0)
        (h0, c0) = hidden

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )
        
        out = out.permute((0, 2, 1))
        
        out_cf = self.conv1d(out)
        out_cf = torch.squeeze(out_cf)
        
        # print('out shape is {}'.format(out.shape))
        out_cf = self.fc(out_cf)
        # print('out shape is {}'.format(out.shape))
        
        out_cf = self.pred(out_cf)
        
        out_reg = self.reg_final(out)        
        
        return [out_cf, out_reg]
    
    def predict(self, x):
        
        out = self.forward(x)
                
        threshold = 0.5
        if self.num_class == 2:
            pred = F.sigmoid(out) > threshold
        
        return pred
    
def test():
    hidden_size = 16
    num_layers = 1
    batch_size = 100

    net = LSTMClassifier(feature_channel=34, output_channel=4, hidden_size=hidden_size,
                   num_layers=num_layers)

    y = net(torch.randn(batch_size, 34, 54))
    print(y.size())

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)


if __name__ == "__main__":
    test()
