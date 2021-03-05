# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from nemo.collections.asr.parts.activations import Swish
from nemo.quantization.utils.quant_modules import * 
import pickle

jasper_activations = {"hardtanh": nn.Hardtanh, "relu": nn.ReLU, "selu": nn.SELU, "swish": Swish}


def init_weights(m, mode: Optional[str] = 'xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        if mode is not None:
            if mode == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif mode == 'xavier_normal':
                nn.init.xavier_normal_(m.weight, gain=1.0)
            elif mode == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif mode == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def compute_new_kernel_size(kernel_size, kernel_width):
    new_kernel_size = max(int(kernel_size * kernel_width), 1)
    # If kernel is even shape, round up to make it odd
    if new_kernel_size % 2 == 0:
        new_kernel_size += 1
    return new_kernel_size


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class StatsPoolLayer(nn.Module):
    def __init__(self, feat_in, pool_mode='xvector'):
        super().__init__()
        self.feat_in = 0
        if pool_mode == 'gram':
            gram = True
            super_vector = False
        elif pool_mode == 'superVector':
            gram = True
            super_vector = True
        else:
            gram = False
            super_vector = False

        if gram:
            self.feat_in += feat_in ** 2
        else:
            self.feat_in += 2 * feat_in

        if super_vector and gram:
            self.feat_in += 2 * feat_in

        self.gram = gram
        self.super = super_vector

    def forward(self, encoder_output):

        mean = encoder_output.mean(dim=-1)  # Time Axis
        std = encoder_output.std(dim=-1)

        pooled = torch.cat([mean, std], dim=-1)

        if self.gram:
            time_len = encoder_output.shape[-1]
            # encoder_output = encoder_output
            cov = encoder_output.bmm(encoder_output.transpose(2, 1))  # cov matrix
            cov = cov.view(cov.shape[0], -1) / time_len

        if self.gram and not self.super:
            return cov

        if self.super and self.gram:
            pooled = torch.cat([pooled, cov], dim=-1)

        return pooled


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        heads=-1,
        bias=False,
        use_mask=True,
        quant_mode='none',
        quant_bit=8,
        asymmetric=False,
        name='',
    ):
        super(MaskedConv1d, self).__init__()
        self.quant_mode = quant_mode
        self.asymmetric = asymmetric
        self.name = name

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads

        conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        act_quant_bit = quant_bit
        if self.asymmetric:
            act_quant_bit += 1 # This has the same effect as using the asymmetric quantization w. bias 0

        self.act = QuantAct(act_quant_bit, quant_mode=self.quant_mode, per_channel=False, name=name+'_act')
        self.conv = QuantConv1d(quant_bit, bias_bit=32, quant_mode=self.quant_mode, 
                per_channel=True, name=name+'_conv')
        self.conv.set_param(conv)

        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return (
            lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1
        ) // self.conv.stride[0] + 1

    def bn_folding(self, bn):
        self.conv.bn_folding(bn)

    def set_quant_bit(self, quant_bit, mode='all'):
        if mode in ['all', 'act']:
            act_quant_bit = quant_bit
            if self.asymmetric:
                act_quant_bit += 1
            self.act.activation_bit = act_quant_bit

        if mode in ['all', 'weight']:
            self.conv.weight_bit = quant_bit

    def set_quant_mode(self, quant_mode):
        self.quant_mode = quant_mode
        self.conv.quant_mode = quant_mode
        self.act.quant_mode = quant_mode

    def forward(self, x, lens, scaling_factor=None):
        assert not(float(x.min()) < -1e-5 and self.asymmetric)
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device).expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
            # del mask
            lens = self.get_seq_len(lens)
        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        x, x_scaling_factor = self.act(x, scaling_factor)
        out, out_scaling_factor = self.conv(x, x_scaling_factor)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens, out_scaling_factor


class GroupShuffle(nn.Module):
    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        channels: int,
        reduction_ratio: int,
        context_window: int = -1,
        interpolation_mode: str = 'nearest',
        activation: Optional[Callable] = None,
    ):
        """
        Squeeze-and-Excitation sub-module.

        Args:
            channels: Input number of channels.
            reduction_ratio: Reduction ratio for "squeeze" layer.
            context_window: Integer number of timesteps that the context
                should be computed over, using stride 1 average pooling.
                If value < 1, then global context is computed.
            interpolation_mode: Interpolation mode of timestep dimension.
                Used only if context window is > 1.
                The modes available for resizing are: `nearest`, `linear` (3D-only),
                `bilinear`, `area`
            activation: Intermediate activation function used. Must be a
                callable activation function.
        """
        super(SqueezeExcite, self).__init__()
        self.context_window = int(context_window)
        self.interpolation_mode = interpolation_mode

        if self.context_window <= 0:
            self.pool = nn.AdaptiveAvgPool1d(1)  # context window = T
        else:
            self.pool = nn.AvgPool1d(self.context_window, stride=1)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            activation,
            nn.Linear(channels // reduction_ratio, channels, bias=False),
        )

    def forward(self, x):
        # The use of negative indices on the transpose allow for expanded SqueezeExcite
        batch, channels, timesteps = x.size()[:3]
        y = self.pool(x)  # [B, C, T - context_window + 1]
        y = y.transpose(1, -1)  # [B, T - context_window + 1, C]
        y = self.fc(y)  # [B, T - context_window + 1, C]
        y = y.transpose(1, -1)  # [B, C, T - context_window + 1]

        if self.context_window > 0:
            y = torch.nn.functional.interpolate(y, size=timesteps, mode=self.interpolation_mode)

        y = torch.sigmoid(y)

        return x * y


class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(
        self,
        inplanes,
        planes,
        repeat=3,
        kernel_size=11,
        kernel_size_factor=1,
        stride=1,
        dilation=1,
        padding='same',
        dropout=0.2,
        activation=None,
        residual=True,
        groups=1,
        separable=False,
        heads=-1,
        normalization="batch",
        norm_groups=1,
        residual_mode='add',
        residual_panes=[],
        conv_mask=False,
        se=False,
        se_reduction_ratio=16,
        se_context_window=None,
        se_interpolation_mode='nearest',
        stride_last=False,
        quant_mode='none', 
        quant_bit=8, 
        layer_num=-1,
    ):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        kernel_size_factor = float(kernel_size_factor)
        if type(kernel_size) in (list, tuple):
            kernel_size = [compute_new_kernel_size(k, kernel_size_factor) for k in kernel_size]
        else:
            kernel_size = compute_new_kernel_size(kernel_size, kernel_size_factor)

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.se = se
        self.quant_mode = quant_mode
        self.layer_num = layer_num
        self.name = 'jb%d' % self.layer_num
        self.cnt = 0

        inplanes_loop = inplanes
        conv = nn.ModuleList()
        self.convs_before_bn = []

        for i in range(repeat - 1):
            #print(i, activation)
            # Stride last means only the last convolution in block will have stride
            if stride_last:
                stride_val = [1]
            else:
                stride_val = stride

            conv.extend(
                self._get_conv_bn_layer(
                    inplanes_loop,
                    planes,
                    kernel_size=kernel_size,
                    stride=stride_val,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    heads=heads,
                    separable=separable,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    quant_mode=quant_mode,
                    quant_bit=quant_bit,
                    is_first_layer=(self.layer_num==0 and i==0),
                    name=self.name+('_conv%d' % i)
                )
            )

            conv.extend(self._get_act_dropout_layer(drop_prob=dropout, activation=activation))

            inplanes_loop = planes

        conv.extend(
            self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                quant_mode=quant_mode,
                quant_bit=quant_bit,
                is_first_layer=(self.layer_num==0 and repeat==1),
                name=self.name+('_conv%d' % (repeat-1))
            )
        )

        if se:
            assert quant_mode == 'none', 'SqueezeExcite does not currently support quantization'
            conv.append(
                SqueezeExcite(
                    planes,
                    reduction_ratio=se_reduction_ratio,
                    context_window=se_context_window,
                    interpolation_mode=se_interpolation_mode,
                    activation=activation,
                )
            )

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()

            if residual_mode == 'stride_add':
                stride_val = stride
            else:
                stride_val = [1]

            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for i, ip in enumerate(res_panes):
                res = nn.ModuleList(
                    self._get_conv_bn_layer(
                        ip,
                        planes,
                        kernel_size=1,
                        normalization=normalization,
                        norm_groups=norm_groups,
                        stride=stride_val,
                        quant_mode=quant_mode,
                        quant_bit=quant_bit,
                        is_first_layer=(self.layer_num==0),
                        name=self.name+('_res%d' % i)
                    )
                )

                res_list.append(res)

            self.res = res_list
        else:
            self.res = None

        self.res_act =  QuantAct(quant_bit, quant_mode=self.quant_mode, per_channel=False, name=self.name+('_res_act'))
        self.mout = nn.Sequential(*self._get_act_dropout_layer(drop_prob=dropout, activation=activation))


    def bn_folding(self):
        def _folding(layers, l):
            if isinstance(l, nn.BatchNorm1d):
                assert isinstance(layers[-1], MaskedConv1d)
                layers[-1].bn_folding(l)
            else:
                layers.append(l)

        conv = nn.ModuleList()

        if self.mconv is not None:
            for l in self.mconv:
                _folding(conv, l)

        self.mconv = conv

        if self.res is not None:
            res = nn.ModuleList()
            for _l in self.res: # for all residual connections
                res_list = nn.ModuleList()
                for l in _l:
                    _folding(res_list, l)
                res.append(res_list)
            self.res = res

    def set_quant_bit(self, quant_bit, mode='all'):
        if self.mconv is not None:
            for l in self.mconv:
                if isinstance(l, MaskedConv1d):
                    l.set_quant_bit(quant_bit, mode)
        if self.res is not None:
            for _l in self.res: # for all residual connections
                for l in _l: # for all operations in a residual connection
                    if isinstance(l, MaskedConv1d):
                        l.set_quant_bit(quant_bit, mode)
        self.res_act.activation_bit = quant_bit

    def set_quant_mode(self, quant_mode):
        self.quant_mode = quant_mode
        if self.mconv is not None:
            for l in self.mconv:
                if isinstance(l, MaskedConv1d):
                    l.set_quant_mode(quant_mode)
        if self.res is not None:
            for _l in self.res: # for all residual connections
                for l in _l: # for all operations in a residual connection
                    if isinstance(l, MaskedConv1d):
                        l.set_quant_mode(quant_mode)
        self.res_act.quant_mode = quant_mode

    def _get_conv(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        quant_mode='none',
        quant_bit=8,
        asymmetric=False,
        name='',
    ):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
                heads=heads,
                use_mask=use_mask,
                quant_mode=quant_mode,
                quant_bit=quant_bit,
                asymmetric=asymmetric,
                name=name,
            )
        else:
            assert quant_mode == 'none', 'Quantization mode only supports convolution with mask currently.'
            return nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups,
            )

    def _get_conv_bn_layer(
        self,
        in_channels,
        out_channels,
        kernel_size=11,
        stride=1,
        dilation=1,
        padding=0,
        bias=False,
        groups=1,
        heads=-1,
        separable=False,
        normalization="batch",
        norm_groups=1,
        quant_mode='none',
        quant_bit=8,
        is_first_layer=False,
        name='',
    ):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                    heads=heads,
                    quant_mode=quant_mode,
                    quant_bit=quant_bit,
                    asymmetric=(not is_first_layer),
                    name=name+'_dw',
                ),
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                    groups=groups,
                    quant_mode=quant_mode,
                    quant_bit=quant_bit,
                    asymmetric=False,
                    name=name+'_pw',
                ),
            ]
        else:
            layers = [
                self._get_conv(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=groups,
                    quant_mode=quant_mode,
                    quant_bit=quant_bit,
                    asymmetric=(not is_first_layer),
                    name=name,
                )
            ]


        if normalization == "group":
            layers.append(nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match" f" one of [batch, layer, group, instance]."
            )

        self.convs_before_bn.append((layers[-2], layers[-1]))

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [activation, nn.Dropout(p=drop_prob)]
        return layers

    def forward(self, input_: Tuple[List[Tuple[Tensor, Optional[Tensor]]], Optional[Tensor]]):
        # input: ([(input0, sf0), ..., (inputN, sfN)], length)
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_

        #if self.quant_mode == 'symmetric':
        #    assert len(xs) == 1

        # compute forward convolutions
        out, out_scaling_factor = xs[-1]
        #out_scaling_factor = input_scaling_factor

        lens = lens_orig
        bn_file_path = None
        #bn_file_path = '/rscratch/sehoonkim/workspace/squeezeasr/squeezeasr/bn_stats/bn/'
        conv_file_path = None
        #conv_file_path = '/rscratch/sehoonkim/workspace/squeezeasr/squeezeasr/bn_stats/%s/' % 'real_bs1'
        prev_conv_name = ''
        adj = False
        for i, l in enumerate(self.mconv):
            if isinstance(l, MaskedConv1d):
                out, lens, out_scaling_factor = l(out, lens, out_scaling_factor)
                prev_conv_name = l.name
                # dump convolution outputs
                if conv_file_path is not None:
                    with open(conv_file_path + ('%d/' % self.cnt) + prev_conv_name + '.pkl', 'wb') as f:
                        pickle.dump(out, f)
                adj = True
            else:
                if isinstance(l, nn.BatchNorm1d):
                    assert adj
                    mean = l.running_mean
                    std = l.running_var ** 0.5
                    stats = [mean, std]
                    # dump bn statistics
                    if bn_file_path is not None:
                        #print(prev_conv_name, float(mean[0]), float(std[0]))
                        with open(bn_file_path + prev_conv_name + '.pkl', 'wb') as f:
                            pickle.dump(stats, f)

                out = l(out)
                adj = False

        # compute the residuals
        if self.res is not None:
            if self.quant_mode == 'symmetric':
                #assert len(self.res) == 1
                assert self.residual_mode == 'add' or self.residual_mode == 'stride_add'
            for i, layer in enumerate(self.res):
                res_out, res_out_scaling_factor = xs[i]
                prev_conv_name = ''
                adj = False
                for j, res_layer in enumerate(layer):
                    #print('res', type(res_layer))
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _, res_out_scaling_factor = \
                                res_layer(res_out, lens_orig, res_out_scaling_factor)
                        prev_conv_name = res_layer.name
                        # dump convolution outputs
                        if conv_file_path is not None:
                            with open(conv_file_path + ('%d/' % self.cnt) + prev_conv_name + '.pkl', 'wb') as f:
                                pickle.dump(res_out, f)
                        adj = True
                    else:
                        if isinstance(res_layer, nn.BatchNorm1d):
                            assert adj
                            mean = res_layer.running_mean
                            std = res_layer.running_var ** 0.5
                            stats = [mean, std]
                            # dump bn statistics
                            if bn_file_path is not None:
                                #print(prev_conv_name, float(mean[0]), float(std[0]))
                                with open(bn_file_path + prev_conv_name + '.pkl', 'wb') as f:
                                    pickle.dump(stats, f)
                        res_out = res_layer(res_out)
                        adj = False

                if self.residual_mode == 'add' or self.residual_mode == 'stride_add':
                    #out = out + res_out
                    out, out_scaling_factor = self.res_act(out, out_scaling_factor,
                            res_out, res_out_scaling_factor)

                else:
                    out = torch.max(out, res_out)
        self.cnt += 1
        # compute the output
        out = self.mout(out)

        if self.res is not None and self.dense_residual:
            return xs + [(out, out_scaling_factor)], lens

        return [(out, out_scaling_factor)], lens
