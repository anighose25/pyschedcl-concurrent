import os
import pyschedcl as fw
import logging
import argparse
import json
import sys
import time
import datetime
import plotly.plotly as py
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import networkx as nx
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.utils.model_zoo as model_zoo
import heapq
import faulthandler
import torch.nn as nn
import logging


def parse_arg(args=None):
    parser = argparse.ArgumentParser(
        description='CNN Training Module')

    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='1')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='1')
    parser.add_argument('-l', '--log',
                        help='Flag for turning on LOG',
                        action="store_true")
    parser.add_argument('-g', '--graph',
                        help='Flag for plotting GANTT chart for execution',
                        action="store_true")
    parser.add_argument('-df', '--dump_output_file',
                        help='Flag for dumping output file for a kernel',
                        action="store_true")
    parser.add_argument('-lf', '--layer_forward',
                        help='Test Layer ID Forward',
                        default=0)
    parser.add_argument('-lb', '--layer_backward',
                        help='Test Layer ID Backward',
                        default=0)

    return parser.parse_args(args)


args = parse_arg(sys.argv[1:])
CLHOST = fw.host_initialize(int(args.nGPU), int(args.nCPU))


class LocalController(object):
    def __init__(self):
        pass


class Transforms(object):

    def __init__(self):
        self.macros = dict()
        self.local_work_size = []
        self.global_work_size = []
        self.buffer_sizes = {'input': [], 'output': []}
        self.buffer_chunks = {'input': [], 'output': []}
        self.local_buffer_sizes = []
        self.variable_values = []
        self.local_chunk = 1
        self.input_features = None
        self.weights = None
        self.bias = None
        self.activation = False
        self.data = {'input': []}



class Transpose(Transforms):
    def __init__(self, num_rows, num_cols, input_features=None):
        super(Transpose, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.input_features = input_features
        self.data['input'].append(input_features)

        self.macros["TRANSPOSEX"] = 16
        self.macros["TRANSPOSEY"] = 16

        self.buffer_sizes['input'].append(num_rows * num_cols)
        self.buffer_sizes['output'].append(num_cols * num_rows)

        self.buffer_chunks['input'].append(num_rows)
        self.buffer_chunks['output'].append(num_cols)

        self.local_work_size = [self.macros["TRANSPOSEX"] * self.macros["TRANSPOSEX"]]
        self.global_work_size = [num_rows * num_cols]

        self.variable_values.append(self.num_rows)
        self.variable_values.append(self.num_cols)

        self.local_chunk = self.macros["TRANSPOSEX"] * self.macros["TRANSPOSEX"]


# Calculate dW

class LinearForward(Transforms):
    def __init__(self, batch_size, num_in, num_out, input_features, weights):
        super(LinearForward, self).__init__()

        self.batch_size = batch_size
        self.num_in = num_in
        self.num_out = num_out
        self.data['input'].append(input_features)
        self.data['input'].append(weights)

        self.macros["WPT"] = 1
        self.macros["TS"] = min(batch_size, num_in, num_out)

        if self.macros["TS"] > 32:
            self.macros["TS"] = 32

        self.buffer_sizes['input'].append(self.batch_size * self.num_in)
        self.buffer_sizes['input'].append(self.num_in * self.num_out)
        self.buffer_sizes['output'].append(self.batch_size * self.num_out)

        self.buffer_chunks['input'].append(self.num_in)
        self.buffer_chunks['input'].append(self.num_out)
        self.buffer_chunks['output'].append(self.num_out)

        self.local_work_size = [self.macros["TS"] * self.macros["TS"]]
        self.global_work_size = [self.batch_size * self.num_out]
        self.variable_values.append(self.batch_size)
        self.variable_values.append(self.num_out)
        self.variable_values.append(self.num_in)
        self.local_chunk = self.num_in


class LinearBackwardgradWeight(Transforms):
    def __init__(self, batch_size, num_in, num_out, inputT_features, gradOutput):
        super(LinearBackwardgradWeight, self).__init__()
        self.batch_size = batch_size
        self.num_in = num_in
        self.num_out = num_out
        self.data['input'].append(inputT_features)
        self.data['input'].append(gradOutput)
        self.macros["TS"] = min(batch_size, num_in, num_out)

        self.macros["WPT"] = 1

        if self.macros["TS"] > 32:
            self.macros["TS"] = 32

        self.buffer_sizes['input'].append(self.num_in * self.batch_size)
        self.buffer_sizes['input'].append(self.batch_size * self.num_out)
        self.buffer_sizes['output'].append(self.num_in * self.num_out)

        self.buffer_chunks['input'].append(self.batch_size)
        self.buffer_chunks['input'].append(self.num_out)
        self.buffer_chunks['output'].append(self.num_out)

        self.local_work_size = [self.macros["TS"] * self.macros["TS"]]
        self.global_work_size = [self.num_in * self.num_out]

        self.variable_values.append(self.num_in)
        self.variable_values.append(self.num_out)
        self.variable_values.append(self.batch_size)

        self.local_chunk = self.batch_size


class LinearBackwardgradInput(Transforms):
    def __init__(self, batch_size, num_in, num_out, gradOutput, weightsT):
        super(LinearBackwardgradInput, self).__init__()
        self.batch_size = batch_size
        self.num_in = num_in
        self.num_out = num_out
        self.data['input'].append(gradOutput)
        self.data['input'].append(weightsT)

        self.macros["WPT"] = 1

        self.macros["TS"] = min(batch_size, num_in, num_out)

        if self.macros["TS"] > 32:
            self.macros["TS"] = 32

        self.buffer_sizes['input'].append(self.batch_size * self.num_out)
        self.buffer_sizes['input'].append(self.num_out * self.num_in)
        self.buffer_sizes['output'].append(self.batch_size * self.num_in)

        self.buffer_chunks['input'].append(self.num_out)
        self.buffer_chunks['input'].append(self.num_in)
        self.buffer_chunks['output'].append(self.num_in)

        self.local_work_size = [self.macros["TS"] * self.macros["TS"]]
        self.global_work_size = [self.batch_size * self.num_in]

        self.variable_values.append(self.batch_size)
        self.variable_values.append(self.num_in)
        self.variable_values.append(self.num_out)

        self.local_chunk = self.num_out


class SoftMaxForward(Transforms):
    def __init__(self, batch_size, classes, input_features=None):
        super(SoftMaxForward, self).__init__()
        self.batch_size = batch_size
        self.classes = classes
        self.data['input'].append(input_features)

        self.buffer_sizes['input'].append(self.batch_size * self.classes)
        self.buffer_sizes['output'].append(self.batch_size * self.classes)

        self.buffer_chunks['input'].append(self.batch_size)
        self.buffer_chunks['output'].append(self.batch_size)

        self.local_work_size = [self.batch_size]
        self.global_work_size = [self.batch_size]

        self.variable_values.append(self.batch_size)
        self.variable_values.append(self.classes)
        self.local_chunk = self.batch_size


class SoftMaxBackward(Transforms):
    def __init__(self, batch_size, classes, input_features=None, predicted_output=None):
        super(SoftMaxBackward, self).__init__()
        self.batch_size = batch_size
        self.classes = classes
        self.data['input'].append(input_features)
        self.data['input'].append(predicted_output)

        self.buffer_sizes['input'].append(self.batch_size * self.classes)
        self.buffer_sizes['input'].append(self.batch_size * self.classes)
        self.buffer_sizes['output'].append(self.batch_size * self.classes)

        self.local_work_size = [self.batch_size]
        self.global_work_size = [self.batch_size]

        self.variable_values.append(self.batch_size)
        self.variable_values.append(self.classes)

        self.local_chunk = self.batch_size


class PoolingForward(Transforms):
    def __init__(self, torch_feature, input_size, batch_size, num_planes, input_features=None):
        super(PoolingForward, self).__init__()

        self.input_features = input_features
        self.data['input'].append(input_features)

        pooling_specification = torch_feature
        self.pooling_size = pooling_specification.kernel_size
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.input_size_W, self.input_size_H = input_size
        self.output_size_H = self.input_size_H / self.pooling_size
        self.output_size_W = self.input_size_W / self.pooling_size
        self.output_size = (self.output_size_W, self.output_size_H)

        self.macros["gNumPlanes"] = self.num_planes
        self.macros["gPoolingSize"] = self.pooling_size
        self.macros["gInputSizeSquared"] = self.input_size_H * self.input_size_W
        self.macros["gOutputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gOutputSize"] = self.output_size_H
        self.macros["gInputSize"] = self.input_size_H

        self.buffer_sizes['input'].append(self.batch_size * self.num_planes * self.input_size_H * self.input_size_W)
        self.buffer_sizes['output'].append(self.batch_size * self.macros["gNumPlanes"] * self.macros[
            "gOutputSizeSquared"])
        self.buffer_sizes['output'].append(self.batch_size * self.macros["gNumPlanes"] * self.macros[
            "gOutputSizeSquared"])
        self.local_work_size = [self.macros["gOutputSizeSquared"]]
        self.global_work_size = [self.macros["gNumPlanes"] * self.batch_size * self.local_work_size[0]]
        self.buffer_chunks['input'].append(self.macros["gNumPlanes"] * self.macros["gOutputSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gNumPlanes"] * self.macros["gOutputSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gNumPlanes"] * self.macros["gOutputSizeSquared"])
        self.local_chunk = self.macros["gNumPlanes"]
        self.variable_values.append(self.batch_size)


class PoolingBackward(Transforms):
    def __init__(self, torch_feature, input_size, batch_size, num_planes, input_features=None):
        super(PoolingBackward, self).__init__()

        self.input_features = input_features
        self.data['input'].append(input_features)

        pooling_specification = torch_feature
        self.pooling_size = pooling_specification.kernel_size
        self.batch_size = batch_size
        self.num_planes = num_planes
        self.input_size_W, self.input_size_H = input_size
        self.output_size_H = self.input_size_H / self.pooling_size
        self.output_size_W = self.input_size_W / self.pooling_size
        self.output_size = (self.output_size_W, self.output_size_H)

        self.macros["gNumPlanes"] = self.num_planes
        self.macros["gPoolingSize"] = self.pooling_size
        self.macros["gInputSizeSquared"] = self.input_size_H * self.input_size_W
        self.macros["gOutputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gOutputSize"] = self.output_size_H
        self.macros["gInputSize"] = self.input_size_H

        self.buffer_sizes['input'].append(batch_size * self.macros["gNumPlanes"] * self.macros[
            "gOutputSizeSquared"])
        self.buffer_sizes['input'].append(batch_size * self.macros["gNumPlanes"] * self.macros[
            "gOutputSizeSquared"])
        self.buffer_sizes['output'].append(batch_size * self.macros["gNumPlanes"] * self.macros[
            "gInputSizeSquared"])
        self.local_work_size = [self.macros["gOutputSizeSquared"]]
        self.global_work_size = [self.macros["gNumPlanes"] * self.batch_size * self.local_work_size[0]]
        self.buffer_chunks['input'].append(self.macros["gNumPlanes"] * self.macros["gOutputSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gNumPlanes"] * self.macros["gOutputSizeSquared"])
        self.local_chunk = self.macros["gNumPlanes"]


class Conv2DForward(Transforms):

    def __init__(self, torch_feature, input_size, batch_size, input_features=None, weights=None, bias=None,
                 activation=False):
        super(Conv2DForward, self).__init__()

        self.input_features = input_features
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.data['input'].append(input_features)
        self.data['input'].append(weights)

        self.input_channels = torch_feature.in_channels
        self.output_channels = torch_feature.out_channels
        self.kernel_H, self.kernel_W = torch_feature.kernel_size
        self.stride_H, self.stride_W = torch_feature.stride
        self.padding_H, self.padding_W = torch_feature.padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.input_size_H, self.input_size_W = self.input_size
        self.output_size_H = (self.input_size_H - self.kernel_H + 2 * self.padding_H) / self.stride_H + 1
        self.output_size_W = (self.input_size_W - self.kernel_W + 2 * self.padding_W) / self.stride_W + 1
        self.input_size_H = self.input_size_H + 2 * self.padding_H
        self.input_size_W = self.input_size_W + 2 * self.padding_W
        self.output_size = (self.output_size_H, self.output_size_W)
        self.Even = 0
        if self.kernel_H % 2 == 0:
            self.Even = 1
        self.Pad = 0
        if self.padding_H > 0:
            self.Pad = 1
        self.Margin = 0
        if self.Pad == 1:
            self.Margin = self.kernel_H >> 1
        else:
            self.Margin = 0

        # Setting up Macros

        self.macros["gInputPlanes"] = self.input_channels
        self.macros["gNumFilters"] = self.output_channels
        self.macros["gFilterSizeSquared"] = self.kernel_H * self.kernel_W
        self.macros["gInputSizeSquared"] = self.input_size_H * self.input_size_W
        self.macros["gOutputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gFilterSize"] = self.kernel_H
        self.macros["gHalfFilterSize"] = self.kernel_H / 2
        self.macros["gEven"] = self.Even
        self.macros["gPad"] = self.Pad
        self.macros["gPadZeros"] = self.Pad
        self.macros["gOutputSize"] = self.output_size_H
        self.macros["gInputSize"] = self.input_size_H
        self.macros["DEBUG"] = 0
        # Setting up Global Buffer Sizes

        self.buffer_sizes['input'].append(self.batch_size * self.input_channels * self.input_size_H * self.input_size_W)
        self.buffer_sizes['input'].append(self.input_channels * self.output_channels * self.kernel_H * self.kernel_W)
        self.buffer_sizes['output'].append(
            self.batch_size * self.macros["gNumFilters"] * self.macros["gOutputSizeSquared"])

        self.buffer_chunks['input'].append(self.macros["gInputPlanes"] * self.macros["gInputSizeSquared"])
        self.buffer_chunks['input'].append(self.macros["gInputPlanes"] * self.macros["gFilterSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gNumFilters"] * self.macros["gOutputSizeSquared"])

        # Setting up Local Buffer Sizes

        self.local_buffer_sizes.append(self.macros["gInputSizeSquared"])
        self.local_buffer_sizes.append(self.macros["gFilterSizeSquared"] * self.macros["gInputPlanes"])

        # Setting up work items

        self.local_work_size = [self.macros["gOutputSizeSquared"]]
        self.global_work_size = [self.macros["gNumFilters"] * self.batch_size * self.local_work_size[0]]

        # Setting up variable values

        self.variable_values.append(self.batch_size)

        # Setting up chunk factor

        self.local_chunk = self.macros["gNumFilters"]


class Conv2DBackwardGradWeights(Transforms):

    def __init__(self, torch_feature, input_size, batch_size, input_features=None, weights=None, bias=None,
                 activation=False):
        super(Conv2DBackwardGradWeights, self).__init__()

        self.input_features = input_features
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.data['input'].append(input_features)
        self.data['input'].append(weights)

        self.input_channels = torch_feature.in_channels
        self.output_channels = torch_feature.out_channels
        self.kernel_H, self.kernel_W = torch_feature.kernel_size
        self.stride_H, self.stride_W = torch_feature.stride
        self.padding_H, self.padding_W = torch_feature.padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.input_size_H, self.input_size_W = self.input_size
        self.output_size_H = (self.input_size_H - self.kernel_H + 2 * self.padding_H) / self.stride_H + 1
        self.output_size_W = (self.input_size_W - self.kernel_W + 2 * self.padding_W) / self.stride_W + 1
        self.input_size_H = self.input_size_H + 2 * self.padding_H
        self.input_size_W = self.input_size_W + 2 * self.padding_W
        self.output_size = (self.output_size_H, self.output_size_W)
        self.Even = 0
        if self.kernel_H % 2 == 0:
            self.Even = 1
        self.Pad = 0
        if self.padding_H > 0:
            self.Pad = 1
        self.Margin = 0
        if self.Pad == 1:
            self.Margin = self.kernel_H >> 1
        else:
            self.Margin = 0

        # Setting up Macros

        self.macros["gInputPlanes"] = self.input_channels
        self.macros["gNumFilters"] = self.output_channels
        self.macros["gFilterSizeSquared"] = self.kernel_H * self.kernel_W
        self.macros["gInputSizeSquared"] = self.input_size_H * self.input_size_W
        self.macros["gOutputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gFilterSize"] = self.kernel_H
        self.macros["gHalfFilterSize"] = self.kernel_H / 2
        self.macros["gEven"] = self.Even
        self.macros["gPad"] = self.Pad
        self.macros["gOutputSize"] = self.output_size_H
        self.macros["gInputSize"] = self.input_size_H
        self.macros["gMargin"] = 1
        self.macros["gNumStripes"] = 4
        self.macros["gInputStripeMarginRows"] = self.macros["gHalfFilterSize"]
        self.macros["gInputStripeMarginSize"] = self.macros["gInputStripeMarginRows"] * self.macros["gInputSize"]
        self.macros["gInputStripeInnerNumRows"] = self.macros["gInputSize"] / self.macros["gNumStripes"]
        self.macros["gInputStripeOuterNumRows"] = self.macros["gInputStripeInnerNumRows"] + 2 * self.macros[
            "gHalfFilterSize"]
        self.macros["gInputStripeInnerSize"] = self.macros["gInputStripeInnerNumRows"] * self.macros["gInputSize"]
        self.macros["gInputStripeOuterSize"] = self.macros["gInputStripeOuterNumRows"] * self.macros["gInputSize"]

        self.buffer_sizes['input'].append(
            self.batch_size * self.output_channels * self.output_size_H * self.output_size_W)
        self.buffer_sizes['input'].append(self.macros["gInputPlanes"] * self.macros["gFilterSizeSquared"])
        self.buffer_sizes['output'].append(self.macros["gNumFilters"] * self.macros["gOutputSizeSquared"])

        # TODO : Investigate buffer chunks

        self.buffer_chunks['input'].append(self.macros["gNumFilters"] * self.macros["gOutputSizeSquared"])
        self.buffer_chunks['input'].append(self.macros["gInputPlanes"] * self.macros["gInputSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gInputPlanes"] * self.macros["gFilterSizeSquared"])

        self.local_buffer_sizes.append(self.macros["gOutputSizeSquared"])
        self.local_buffer_sizes.append(self.macros["gInputSizeSquared"])

        self.local_work_size = [self.macros["gFilterSizeSquared"]]
        self.global_work_size = [self.macros["gNumFilters"] * self.macros["gInputPlanes"] * self.local_work_size[0]]

        self.local_chunk = self.macros["gInputPlanes"]

        self.variable_values.append(self.batch_size)


class Conv2DBackwardGradInput(Transforms):

    def __init__(self, torch_feature, input_size, batch_size, input_features=None, weights=None, bias=None,
                 activation=False):
        super(Conv2DBackwardGradInput, self).__init__()

        self.input_features = input_features
        self.weights = weights
        self.bias = bias
        self.activation = activation

        self.data['input'].append(input_features)
        self.data['input'].append(weights)

        self.input_channels = torch_feature.in_channels
        self.output_channels = torch_feature.out_channels
        self.kernel_H, self.kernel_W = torch_feature.kernel_size
        self.stride_H, self.stride_W = torch_feature.stride
        self.padding_H, self.padding_W = torch_feature.padding
        self.input_size = input_size
        self.batch_size = batch_size
        self.input_size_H, self.input_size_W = self.input_size
        self.output_size_H = (self.input_size_H - self.kernel_H + 2 * self.padding_H) / self.stride_H + 1
        self.output_size_W = (self.input_size_W - self.kernel_W + 2 * self.padding_W) / self.stride_W + 1
        self.input_size_H = self.input_size_H + 2 * self.padding_H
        self.input_size_W = self.input_size_W + 2 * self.padding_W
        self.output_size = (self.output_size_H, self.output_size_W)
        self.Even = 0
        if self.kernel_H % 2 == 0:
            self.Even = 1
        self.Pad = 0
        if self.padding_H > 0:
            self.Pad = 1
        self.Margin = 0
        if self.Pad == 1:
            self.Margin = self.kernel_H >> 1
        else:
            self.Margin = 0

        self.macros["gOutputSize"] = self.output_size_H
        self.macros["gInputSize"] = self.input_size_H
        self.macros["gEven"] = self.Even
        self.macros["gFilterSizeSquared"] = self.kernel_H * self.kernel_W
        self.macros["gInputPlanes"] = self.input_channels
        self.macros["gNumFilters"] = self.output_channels
        self.macros["gInputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gOutputSizeSquared"] = self.output_size_H * self.output_size_W
        self.macros["gPadZeros"] = self.Pad
        self.macros["gFilterSize"] = self.kernel_H
        self.macros["gHalfFilterSize"] = self.kernel_H / 2
        self.macros["gMargin"] = self.Margin

        self.buffer_sizes['input'].append(
            self.batch_size * self.output_channels * self.output_size_H * self.output_size_W)

        self.buffer_sizes['input'].append(
            self.output_channels * self.input_channels * self.output_size_W * self.output_size_H)
        self.buffer_sizes['output'].append(
            self.batch_size * self.macros["gInputPlanes"] * self.macros["gOutputSizeSquared"])

        self.buffer_chunks['input'].append(self.macros["gNumFilters"] * self.macros["gOutputSizeSquared"])
        self.buffer_chunks['input'].append(self.macros["gInputPlanes"] * self.macros["gFilterSizeSquared"])
        self.buffer_chunks['output'].append(self.macros["gInputPlanes"] * self.macros["gInputSizeSquared"])

        self.local_buffer_sizes.append(self.macros["gOutputSizeSquared"])
        self.local_buffer_sizes.append(self.macros["gFilterSizeSquared"])

        self.local_work_size = [self.macros["gInputSizeSquared"]]
        self.global_work_size = [self.macros["gInputPlanes"] * self.batch_size * self.local_work_size[0]]

        self.local_chunk = self.macros["gNumFilters"]


class TensorKernel(object):

    def __init__(self, info, transform):
        self.partition = 0.0
        self.thread_coarsening_cpu = 1
        self.thread_coarsening_gpu = 1
        self.stride = 32
        self.info_file = info
        self.kernel = None
        self.cmd_qs, self.ctxs, self.gpus, self.cpus = CLHOST
        self.transform = transform
        self.execution_time = 0.0
        self.initialize()

    def initialize(self):
        info = json.loads(open(self.info_file).read())
        kernel = fw.Kernel(info)

        if self.thread_coarsening_cpu > 1:
            kernel.src_cpu = self.thread_coarsen(self.info_file, self.thread_coarsening_cpu, self.stride)

        if self.thread_coarsening_gpu > 1:
            kernel.src_gpu = self.thread_coarsen(self.info_file, self.thread_coarsening_gpu, self.stride)

        kernel.src = kernel.src_cpu  # Redundant assignment to avoid errors

        kernel.macros = self.transform.macros

        print kernel.local_args
        for index in xrange(0, len(kernel.local_args)):
            print index
            kernel.local_args[index]['size'] = self.transform.local_buffer_sizes[index]

        for buffer_type in ['input', 'output']:
            for index in xrange(len(kernel.buffer_info[buffer_type])):
                kernel.buffer_info[buffer_type][index]['size'] = self.transform.buffer_sizes[buffer_type][index]
                kernel.buffer_info[buffer_type][index]['chunk'] = self.transform.buffer_chunks[buffer_type][index]
        kernel.local_work_size = self.transform.local_work_size
        kernel.global_work_size = self.transform.global_work_size
        kernel.local_chunk = self.transform.local_chunk
        self.kernel = kernel

    def dump(self):
        print "KERNEL SRC", self.kernel.src
        print "LOCAL WORK SIZE CONV", self.kernel.local_work_size
        print "GLOBAL WORK SIZE CONV", self.kernel.global_work_size
        print "MACROS", self.kernel.macros
        print "GLOBAL BUFFER", self.kernel.buffer_info
        print "LOCAL BUFFER", self.kernel.local_args
        print "LOCAL CHUNK SIZE FACTOR", self.kernel.local_chunk

    def compile_kernel(self):
        self.kernel.partition = self.partition
        self.kernel.build_kernel(self.gpus, self.cpus, self.ctxs)

    # TODO: Implement load data functionality

    def load_data(self):
        kernel_data = self.transform.data
        for index in xrange(len(kernel_data['input'])):
            if kernel_data['input'][index] is None:
                kernel_data['input'][index] = np.random.uniform(low=0.1, high=0.9,
                                                                size=(self.transform.buffer_sizes['input'][index],))

        self.kernel.load_data(kernel_data)

    def dispatch_kernel(self):
        start_time, done_events = self.kernel.dispatch(0, 0, self.ctxs, self.cmd_qs,
                                                       C_cpu=self.thread_coarsening_cpu,
                                                       C_gpu=self.thread_coarsening_gpu)
        fw.host_synchronize(self.cmd_qs, done_events)
        end_time = datetime.datetime.now()
        seconds = (end_time - start_time).total_seconds()
        return seconds

    def run_kernel(self):
        self.initialize()
        self.load_data()
        self.compile_kernel()
        t = self.dispatch_kernel()
        print "Span Time: ", t
        self.execution_time = t
        outputs = []
        for output in self.kernel.buffer_info['output']:
            outputs.append(output)
        return tuple(outputs)

    @staticmethod
    def thread_coarsen(info_file, coarsening_factor, stride):
        command = "~/Tools/python/bin/expander.py --eval "
        arguments = "\'C=" + str(coarsening_factor) + ";" + "S=" + str(stride) + "\' "
        command += arguments
        input_file = info_file[:-5] + "_thread_coarsening.txt > "
        output_file = "kernel_src/" + info_file[:-5].split("/")[1] + "_thread_coarsened_" + str(
            coarsening_factor) + "_stride_" + str(stride) + ".cl"
        command += input_file + output_file
        os.system(command)
        logging.debug(output_file)
        return output_file.split("/")[1]


class Layer(object):

    def __init__(self, input_features=None, batch_size=32, torch_features=None, torch_w=None, torch_b=None,
                 trainable=False, activation=False):
        self.cmd_qs, self.ctxs, self.gpus, self.cpus = CLHOST
        self.batch_size = batch_size
        self.torch_features = torch_features
        self.trainable = trainable
        self.activation = activation
        self.input_features = input_features
        self.cache = {'Z': None, 'A': None}
        self.parameters = {'W': torch_w, 'b': torch_b}
        self.grads = {'dW': None, 'db': None}
        self.gradInput = None
        self.gradOutput = None


    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class ConvolutionLayer(Layer):
    def __init__(self, input_size, info_convolution_forward, info_convolution_backward_gradweights,
                 info_convolution_backward_gradinput):
        super(ConvolutionLayer, self).__init__()
        self.convolution_kernel_forward = TensorKernel(info_convolution_forward,
                                                       Conv2DForward(torch_feature, input_size, self.batch_size, None,
                                                                     None, None))

        self.convolution_kernel_backward_dW = TensorKernel(info_convolution_backward_gradweights,
                                                           Conv2DBackwardGradWeights(torch_feature, input_size,
                                                                                     self.batch_size, None,
                                                                                     None, None))

        self.convolution_kernel_backward_gradinput = TensorKernel(info_convolution_backward_gradinput,
                                                                  Conv2DBackwardGradInput(torch_feature, input_size,
                                                                                          self.batch_size, None,
                                                                                          None, None))

    def forward(self, input_features=None):
        if input_features is not None:
            self.input_features = input_features

        self.convolution_kernel_forward.setinput(input_features)


    def backward(self):
        pass


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 6, 5),
                                      nn.Conv2d(6, 16, 5))
        self.classifier = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                        nn.Linear(120, 84),
                                        nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class NeuralNetwork(object):

    def __init__(self, training_dataset, model, args=None, batch_size=32):

        self.layer_weights = []
        self.input_channels = 0
        self.image_height = 0
        self.image_width = 0
        self.num_images = 0
        self.dataset = None
        self.labels = None
        self.nn_model = None
        self.feature_transforms = []
        self.layer_transforms = []
        self.classifier_transforms = []
        self.dataset_name = training_dataset
        self.model_name = model
        self.get_dataset(training_dataset)
        self.get_model(model)
        self.forward_local_controllers = []
        self.backward_local_controllers = []

        self.layer_objects = []
        self.batch_size = batch_size

        self.cmd_qs, self.ctxs, self.gpus, self.cpus = CLHOST
        # print self.cmd_qs
        self.kernels = []

        if args.log:
            f_path = fw.SOURCE_DIR + 'logs/' + 'NN_debug.log'
            logging.basicConfig(filename=f_path, level=logging.DEBUG)
            print "LOG file is saved at %s" % f_path

        if args.dump_output_file:
            fw.dump_output = True

    def print_controller_information(self):
        for lc in self.forward_local_controllers:
            lc.print_information()

    def set_initial_set_points(self):
        for lc in self.forward_local_controllers:
            lc.set_setpoint()

    def normalize_controllers(self):
        for lc in self.forward_local_controllers:
            lc.normalize_set_points()

    def set_batch_size(self, batch):
        self.batch_size = batch

    def get_dataset(self, dataset_folder):
        if "cifar10" in dataset_folder:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            train_set = torchvision.datasets.CIFAR10(dataset_folder, train=True, download=True, transform=transform)
            self.dataset = train_set.train_data
            self.num_images, self.input_channels, self.image_height, self.image_width = self.dataset.shape
            print "DATASET STATS", self.num_images, self.input_channels, self.image_height, self.image_width
            self.labels = train_set.train_labels

    def process_data(self):
        self.dataset = np.pad(self.dataset, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant').flatten()
        # print self.labels
        self.labels = np.eye(10)[self.labels]
        # print self.labels

    def get_minibatch_training_data(self, offset, minibatch_size):
        actual_size = self.num_images * self.image_width * self.image_height * minibatch_size
        data = self.dataset[offset:actual_size]
        return data

    def get_model(self, model_name):
        if model_name == "vgg16":
            self.nn_model = torchvision.models.vgg16()
            # self.nn_model.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth'))
        if model_name == "vgg13":
            self.nn_model = torchvision.models.vgg13()
        if model_name == "vgg11":
            self.nn_model = torchvision.models.vgg11()
        if model_name == "LeNet":
            self.nn_model = LeNet()

        for feature in self.nn_model.features:
            self.feature_transforms.append(feature)
        for classifier in self.nn_model.classifier:
            self.classifier_transforms.append(classifier)
        for name, param in self.nn_model.named_parameters():
            # print name,type(param.data), param.size()
            name_info = name.split(".")
            name_type = name_info[0]
            name_index = int(name_info[1])
            name_param = name_info[2]
            weights = param.data.numpy()
            weights = weights.flatten()
            # print weights.shape
            self.layer_weights.append(weights)
            if name_type == "features":
                self.layer_transforms.append((self.feature_transforms[name_index], name_param))
            else:
                self.layer_transforms.append((self.classifier_transforms[name_index], name_param))

    def print_model_layer_names(self):
        for i in range(0, len(self.layer_transforms)):
            print self.layer_transforms[i], self.layer_weights[i].shape
        for feat in self.feature_transforms:
            print str(type(feat))
        print self.feature_transforms
        print self.classifier_transforms

    def get_filter_cube(self, filter_layer_index):
        return self.layer_weights[filter_layer_index]

    def generate_input_cube(self, minibatch_size, input_size, num_channels, padding=0):
        input_size_W, input_size_H = input_size
        if padding == 1:
            input_size_W += 2
            input_size_H += 2
        total_size = minibatch_size * num_channels * input_size_H * input_size_W
        input_cube = np.random.uniform(low=0.1, high=0.9, size=(total_size,))
        return input_cube

    def get_input_cube(self, minibatch_size, offset=0, padding=0):
        batch_input_data = self.get_minibatch(minibatch_size, offset)
        # print batch_input_data.shape
        if padding == 1:
            batch_input_data = np.pad(batch_input_data, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
        # print batch_input_data.shape
        return batch_input_data.flatten()

    def get_allowable_partition_percentages(self, batch_size):
        allowable_partitions = list(10 * np.arange(0, 1.0, 1.0 / batch_size))
        allowable_partitions.append(10.0)
        return allowable_partitions


if __name__ == '__main__':
    # Initialization

    NN = NeuralNetwork("cifar10_data", "LeNet", args)
    NN.print_model_layer_names()

    # Test Convolution Kernel

    info_convolution_forward = "info/forward3_double.json"
    torch_feature = nn.Conv2d(3, 256, 3, padding=1)
    conv_kernel = TensorKernel(info_convolution_forward, Conv2DForward(torch_feature, (32, 32), 512, None, None, None))
    conv_kernel.dump()
    conv_kernel.run_kernel()

    # Test FFC Kernel

    info_ffc_forward = "info/FFC.json"
    ffc_kernel = TensorKernel(info_ffc_forward, LinearForward(32, 512, 4096, None, None))
    ffc_kernel.dump()
    ffc_kernel.run_kernel()
