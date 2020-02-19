#!/bin/python
# -----------------------------------------------------------------------------
# File Name : multilayer.py
# Purpose:
#
# Author: Emre Neftci
#
# Creation Date : 12-03-2019
# Last Modified : Tue 12 Mar 2019 04:51:44 PM PDT
#
# Copyright : (c) 
# Licence : GPLv2
# -----------------------------------------------------------------------------
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from itertools import chain
from collections import namedtuple
import warnings
from decolle.utils import train, test, accuracy, load_model_from_checkpoint, save_checkpoint, write_stats, get_output_shape

dtype = torch.float32

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).float()

    def backward(aux, grad_output):
        # grad_input = grad_output.clone()
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input

smooth_step = SmoothStep().apply
sigmoid = nn.Sigmoid()

class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])

    def __init__(self, layer, alpha=.9, alpharp=.65, wrp=1.0, beta=.85, deltat=1000):
        super(LIFLayer, self).__init__()
        self.base_layer = layer
        self.alpha = torch.tensor(alpha)
        self.beta = torch.tensor(beta)
        self.tau_m = torch.nn.Parameter(1. / (1 - self.alpha), requires_grad=False)
        self.tau_s = torch.nn.Parameter(1. / (1 - self.beta), requires_grad=False)
        self.deltat = deltat
        self.alpharp = alpharp
        self.wrp = wrp
        self.state = None

    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if type(layer) == nn.Conv2d:
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        elif hasattr(layer, 'out_features'): 
            layer.weight.data[:]*=0
            if layer.bias is not None:
                layer.bias.data.uniform_(-1e-3,1e-3)
        else:
            warning.warn('Unhandled data type, not resetting parameters')
    
    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'): 
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'): 
            return layer.get_out_channels()
        else: 
            raise Exception('Unhandled base layer type')
    
    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape, 
                                    kernel_size=layer.kernel_size,
                                    stride = layer.stride,
                                    padding = layer.padding,
                                    dilation = layer.dilation)
        elif hasattr(layer, 'out_features'): 
            return []
        elif hasattr(layer, 'get_out_shape'): 
            return layer.get_out_shape()
        else: 
            raise Exception('Unhandled base layer type')

    def init_state(self, Sin_t):
        device = self.base_layer.weight.device
        input_shape = list(Sin_t.shape)
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

    def init_parameters(self, Sin_t):
        self.reset_parameters(self.base_layer)


    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(Sin_t)

        state = self.state
        Q = self.beta * state.Q + self.tau_s * Sin_t
        P = self.alpha * state.P + self.tau_m * state.Q  # TODO check with Emre: Q or state.Q?
        R = self.alpharp * state.R - state.S * self.wrp
        U = self.base_layer(P) + R
        S = smooth_step(U)
        self.state = self.NeuronState(P=P.detach(), Q=Q.detach(), R=R.detach(), S=S.detach())
        return S, U

    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if not isinstance(layer, nn.Conv2d):
            raise TypeError('You can only get the output shape of Conv2d layers. Please change layer type')
        im_height = input_shape[-2]
        im_width = input_shape[-1]
        height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                      (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
        weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                      (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
        return [height, weight]
    
    def get_device(self):
        return self.base_layer.weight.device


class DECOLLE(nn.Module):
    requires_init = True

    def __len__(self):
        return len(self.LIF_layers)

    def forward(self, input):
        raise NotImplemented('')
    
    @property
    def output_layer(self):
        return self.readout_layers[-1]

    def get_trainable_parameters(self, layer=None):
        if layer is None:
            return chain(*[l.parameters() for l in self.LIF_layers])
        else:
            return self.LIF_layers[layer].parameters()

    def init(self, data_batch, burnin):
        '''
        Necessary to reset the state of the network whenever a new batch is presented
        '''
        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None
        for i in range(max(len(self), burnin)):
            self.forward(data_batch[:, i, :, :])

    def init_parameters(self, data_batch):
        Sin_t = data_batch[:, 0, :, :]
        s_out, r_out = self.forward(Sin_t)[:2]
        ins = [Sin_t]+s_out
        for i,l in enumerate(self.LIF_layers):
            l.init_parameters(ins[i])

    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)
    
    def get_input_layer_device(self):
        if hasattr(self.LIF_layers[0], 'get_device'):
            return self.LIF_layers[0].get_device() 
        else:
            return list(self.LIF_layers[0].parameters())[0].device

    def get_output_layer_device(self):
        return self.output_layer.weight.device


    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 alpha=[.9],
                 beta=[.85],
                 alpharp=[.65],
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 deltat=1000,
                 lc_ampl=.5,
                 lif_layer_type=LIFLayer):

        num_layers = num_conv_layers + num_mlp_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:   kernel_size = kernel_size * num_conv_layers
        if len(stride) == 1:        stride = stride * num_conv_layers
        if len(pool_size) == 1:     pool_size = pool_size * num_conv_layers
        if len(alpha) == 1:         alpha = alpha * num_layers
        if len(alpharp) == 1:       alpharp = alpharp * num_layers
        if len(beta) == 1:          beta = beta * num_layers
        if len(dropout) == 1:       self.dropout = dropout = dropout * num_layers
        if len(Nhid) == 1:          self.Nhid = Nhid = Nhid * num_conv_layers
        if len(Mhid) == 1:          self.Mhid = Mhid = Mhid * num_mlp_layers

        super(DECOLLE, self).__init__()

        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()

        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2  # TODO try to remove padding

        feature_height = input_shape[1]
        feature_width = input_shape[2]

        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        for i in range(num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = lif_layer_type(base_layer,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i],
                                   deltat=deltat)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        Mhid = [mlp_in] + Mhid
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i + 1])
            layer = lif_layer_type(base_layer,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i],
                                   deltat=deltat)
            readout = nn.Linear(Mhid[i + 1], out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[self.num_conv_layers + i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers:
                input = input.view(input.size(0), -1)
            s, u = lif(input)
            u_p = pool(u)
            s_ = smooth_step(u_p)
            sd_ = do(s_)
            r_ = ro(sd_.reshape(sd_.size(0), -1))
            s_out.append(s_)
            r_out.append(r_)
            u_out.append(u_p)
            input = s_.detach()
            i += 1

        return s_out, r_out, u_out


if __name__ == "__main__":
    # Test building network
    net = DECOLLE(Nhid=[1, 8], Mhid=[32, 64], out_channels=10, input_shape=[1, 28, 28])
    d = torch.zeros([1, 1, 28, 28])
    net(d)
