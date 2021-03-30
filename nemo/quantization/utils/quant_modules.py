import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Linear as _linear
from torch.nn import Embedding as _Embedding
from torch.nn import Module, Parameter
from .quant_utils import *

import logging

logger = logging.getLogger(__name__)
# The input quantization needs to use symmetric quantization!

class QuantAct(Module):
    """
    Class to quantize given activations

    Parameters:
    ----------
    activation_bit : int
        Bitwidth for quantized activations.
    act_range_momentum : float, default 0.95
        Momentum for updating the activation quantization range.
    running_stat : bool, default True
        Whether to use running statistics for activation quantization range.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    channel_len : int, default None
        Specify the channel length when using the per_channel mode.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 activation_bit,
                 act_range_momentum=0.95,
                 running_stat=True,
                 per_channel=False,
                 channel_len=None,
                 quant_mode="none",
                 dynamic=False,
                 percentile=None,
                ):
        super(QuantAct, self).__init__()

        self.activation_bit = activation_bit
        self.act_range_momentum = act_range_momentum
        self.running_stat = running_stat
        self.quant_mode = quant_mode
        self.percentile = percentile

        if not per_channel:
            self.register_buffer('x_min', torch.zeros(1))
            self.register_buffer('x_max', torch.zeros(1))
            self.register_buffer('act_scaling_factor', torch.zeros(1))
        else:
            assert channel_len is not None
            self.register_buffer('x_min', torch.zeros(channel_len))
            self.register_buffer('x_max', torch.zeros(channel_len))
            self.register_buffer('act_scaling_factor', torch.zeros(channel_len))

        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.dynamic = dynamic

        if self.quant_mode == "none":
            self.act_function = None
        elif self.quant_mode == "symmetric":
            self.act_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(self.quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        return "{0}(activation_bit={1}, " \
               "quant_mode: {2}, Act_min: {3:.2f}, " \
               "Act_max: {4:.2f})".format(self.__class__.__name__, self.activation_bit,
                                          self.quant_mode, self.x_min.item(), self.x_max.item())
    def fix(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False
        
    def unfix(self):
        """
        unfix the activation range by setting running stat
        """
        self.running_stat = True

    def adjust_range(self, scale):
        self.x_min *= scale
        self.x_max *= scale

    def set_percentile(self, percentile):
        assert not self.per_channel, 'percentile mode is only available for the global quantization mode'
        self.percentile = percentile

    def forward(self, x, 
                pre_act_scaling_factor=None, 
                identity=None, 
                identity_scaling_factor=None,
                specified_min=None,
                specified_max=None):
        # collect runnng stats
        quantile_min, quantile_max = None, None # avoid double computation
        x_act = x if identity is None else identity + x
        if self.running_stat:
            if self.percentile is None:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=-1).values
                    x_max = x_act.data.max(axis=0).values.max(axis=-1).values
            else:
                assert not self.per_channel, 'percentile mode is only available for the global quantization mode'
                quantile_min = torch.quantile(x_act, torch.tensor(1 - self.percentile / 100).cuda())
                quantile_max = torch.quantile(x_act, torch.tensor(self.percentile / 100).cuda())
                x_min = quantile_min.detach()
                x_max = quantile_max.detach()

            # Initialization
            if torch.eq(self.x_min, self.x_max).all():
                self.x_min = self.x_min + x_min
                self.x_max = self.x_max + x_max

            # exponential moving average (EMA)
            # use momentum to prevent the quantized values change greatly every iteration
            elif self.act_range_momentum == -1:
                self.x_min = torch.min(self.x_min, x_min)
                self.x_max = torch.max(self.x_max, x_max)
            else:
                self.x_min = self.x_min * self.act_range_momentum +\
                        x_min * (1 - self.act_range_momentum)
                self.x_max = self.x_max * self.act_range_momentum +\
                        x_max * (1 - self.act_range_momentum)

        if self.quant_mode == 'none':
            return x_act, None

        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)
        
        if self.dynamic:
            if not self.percentile:
                if not self.per_channel:
                    x_min = x_act.data.min()
                    x_max = x_act.data.max()
                else:
                    x_min = x_act.data.min(axis=0).values.min(axis=-1).values
                    x_max = x_act.data.max(axis=0).values.max(axis=-1).values
            else:
                assert not self.per_channel, 'percentile mode is only available for the global quantization mode'
                # avoid quantile computation if already computed above
                if quantile_min is None:
                    x_min = torch.quantile(x_act, torch.tensor(1 - self.percentile / 100).cuda())
                else:
                    x_min = quantile_min
                if quantile_max is None:
                    x_max = torch.quantile(x_act, torch.tensor(self.percentile / 100).cuda())
                else:
                    x_max = quantile_max
        else:
            x_min = self.x_min if specified_min is None else specified_min
            x_max = self.x_max if specified_max is None else specified_max

        x_min = x_min.view(1, -1, 1)
        x_max = x_max.view(1, -1, 1)


        act_scaling_factor = symmetric_linear_quantization_params(
            self.activation_bit, x_min, x_max, 
            per_channel=self.per_channel)
        self.act_scaling_factor = act_scaling_factor.reshape(-1)

        if pre_act_scaling_factor is None:
            # this is for the input quantization 
            quant_act_int = self.act_function(x, self.activation_bit, \
                    self.percentile, act_scaling_factor)

            # This is just a temporal solution
            # Normally, if pre_act_scaling_factor is None, then identity is None as well
            x = quant_act_int * act_scaling_factor
            pre_act_scaling_factor = act_scaling_factor

        quant_act_int = fixedpoint_mul.apply(
                x, pre_act_scaling_factor, 
                self.activation_bit, self.quant_mode, 
                act_scaling_factor, 
                identity, identity_scaling_factor)

        correct_output_scale = act_scaling_factor

        return quant_act_int * correct_output_scale, act_scaling_factor


class QuantConv1d(Module):
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 quant_mode='none',
                 per_channel=False,
                 fix_flag=False,
                 fix_bn = True,
                 weight_percentile=0,
                ):
        super(QuantConv1d, self).__init__()
        self.weight_bit = weight_bit
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = False if bias_bit is None else True
        self.quant_mode = quant_mode
        self.counter = 1
        self.percentile_mode = False
        self.fix_bn = fix_bn
        self.update_bn = True

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = Parameter(conv.weight.data.clone())
        try:
            self.bias = Parameter(conv.bias.data.clone())
            self.register_buffer('bias_integer', torch.zeros_like(self.bias))
        except AttributeError:
            self.bias = None
            self.bias_integer = None
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        self.register_buffer('conv_scaling_factor', torch.zeros(self.out_channels))
        self.conv = conv
        self.bn = None

    def __repr__(self):
        return "{0}(weight_bit={1}, per_channel: {2}, " \
               "quant_mode: {3}".format(self.__class__.__name__, self.weight_bit,
                                        self.per_channel, self.quant_mode)

    def fix(self):
        """
        fix the BN statistics by setting fix_BN to True
        """
        self.fix_bn = True

    def unfix(self):
        """
        change the mode (fixed or not) of BN statistics to its original status
        """
        self.fix_bn = False

    def bn_folding(self, bn):
        self.bn = bn

    def int_conv(self, weight, bias, x, pre_act_scaling_factor):
        if self.per_channel:
            w_min, _ = torch.min(weight, dim=-1, out=None)
            w_max, _ = torch.max(weight, dim=-1, out=None)
            w_min, _ = torch.min(w_min, dim=-1, out=None)
            w_max, _ = torch.max(w_max, dim=-1, out=None)
        else:
            w_min = weight.min()
            w_max = weight.max()

        w_min = w_min.view(-1, 1, 1)
        w_max = w_max.view(-1, 1, 1)

        conv_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, w_min, w_max, self.per_channel)

        self.conv_scaling_factor = conv_scaling_factor.reshape(-1)

        self.weight_integer = self.weight_function(
                weight, self.weight_bit, self.percentile_mode, conv_scaling_factor)

        bias_scaling_factor = conv_scaling_factor * pre_act_scaling_factor

        if bias is not None:
            # self.bias_integer?
            bias_integer = self.weight_function(bias, 
                self.bias_bit, self.percentile_mode, bias_scaling_factor.reshape([-1])).type(torch.double)
        else:
            bias_integer = None

        x_int = (x / pre_act_scaling_factor).type(torch.double)
        w_int = self.weight_integer.type(torch.double)

        conv_int = F.conv1d(x_int, weight=w_int, bias=bias_integer,
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups).type(torch.float)

        correct_scaling_factor = bias_scaling_factor.view(1, -1, 1)
        conv_output = conv_int * correct_scaling_factor
        return conv_output, correct_scaling_factor

    def forward(self, x, pre_act_scaling_factor=None):
        """
        x: the input activation
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer
        """
        if self.quant_mode == 'none':
            conv = F.conv1d(x, weight=self.weight, bias=self.bias, 
                    stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

            if self.bn is not None:
                conv = self.bn(conv)
            return conv, None
        
        assert self.quant_mode == 'symmetric'

        if self.bn is None or not self.fix_bn :

            weight = self.weight.data.detach()
            bias = None if self.bias is None else self.bias.data.detach()
            conv_output, correct_scaling_factor = self.int_conv(weight, bias, x, pre_act_scaling_factor)
            
            if self.bn is None:
                return conv_output, correct_scaling_factor
            
            assert self.bn is not None and not self.fix_bn
            batch_mean = torch.mean(conv_output, dim=(0, 2))
            batch_var = torch.var(conv_output, dim=(0, 2))


            if self.update_bn:
                # update running mean and vairance
                self.bn.running_mean = self.bn.running_mean.detach() * (1 - self.bn.momentum) + self.bn.momentum * batch_mean
                self.bn.running_var = self.bn.running_var.detach() * (1 - self.bn.momentum) + self.bn.momentum * batch_var

            output_factor = self.bn.weight.view(1, -1, 1) / torch.sqrt(self.bn.running_var + self.bn.eps).view(1, -1, 1)
            output = output_factor * (conv_output - self.bn.running_mean.view(1, -1, 1)) + self.bn.bias.view(1, -1, 1)

            return output, output_factor * correct_scaling_factor

        assert self.bn is not None and self.fix_bn
        running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        scaled_weight = self.weight * scale_factor.reshape([-1, 1, 1])

        if self.bias is not None:
            scaled_bias = self.bias
        else:
            scaled_bias = torch.zeros_like(self.bn.running_mean)
        scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias

        # TODO: do refactoring 
        weight = scaled_weight.data.detach()
        bias = scaled_bias.data.detach()
        conv_output, correct_scaling_factor = self.int_conv(weight, bias, x, pre_act_scaling_factor)
        
        return conv_output, correct_scaling_factor



class QuantLinear(Module):
    """
    Class to quantize weights of given Linear layer
    
    Parameters:
    ----------
    weight_bit : int
        Bitwidth for quantized weights.
    bias_bit : int, default None
        Bitwidth for quantized bias.
    per_channel : bool, default False
        Whether to use channel-wise quantization.
    quant_mode : 'none' or 'symmetric', default 'none'
        The mode for quantization. 'none' for no quantization.
    """
    def __init__(self,
                 weight_bit,
                 bias_bit=None,
                 per_channel=False,
                 quant_mode='none'):
        super(QuantLinear, self).__init__()
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.percentile_mode = False

        if self.quant_mode == "none":
            pass
        elif self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError("unsupported quant mode: {}".format(quant_mode))
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quant_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s

    def set_param(self, linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.clone())
        self.register_buffer('fc_scaling_factor', torch.zeros(self.out_features))
        self.register_buffer('weight_integer', torch.zeros_like(self.weight))
        try:
            self.bias = Parameter(linear.bias.data.clone())
        except AttributeError:
            self.bias = None
        self.register_buffer('bias_integer', torch.zeros_like(self.bias))

    def fix(self):
        pass

    def unfix(self):
        pass

    def forward(self, x, prev_act_scaling_factor=None):
        """
        using quantized weights to forward activation x
        """
        if self.quant_mode == 'none':
            return F.linear(x, weight=self.weight, bias=self.bias), None

    	# x / prev_act_scaling_factor = int
        assert self.quant_mode == 'symmetric', \
                "unsupported quant mode: {}".format(quant_mode)

        # assert that prev_act_scaling_factor is a scalar tensor
        # e.g. all input tensors have the same scalar factor
        assert prev_act_scaling_factor is not None and \
              prev_act_scaling_factor.shape == (1,) 

        w = self.weight
        w_transform = w.data.detach()
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        self.fc_scaling_factor = symmetric_linear_quantization_params(
                self.weight_bit, w_min, w_max, self.per_channel)
        self.weight_integer = self.weight_function(
                self.weight, self.weight_bit, self.percentile_mode, 
                self.fc_scaling_factor)

        bias_scaling_factor = self.fc_scaling_factor * prev_act_scaling_factor

        self.bias_integer = self.weight_function(self.bias, 
                self.bias_bit, False, bias_scaling_factor)

        prev_act_scaling_factor = prev_act_scaling_factor.view(1, -1)
        x_int = x / prev_act_scaling_factor

        return F.linear(x_int, weight=self.weight_integer, bias=self.bias_integer) \
                * bias_scaling_factor, bias_scaling_factor
