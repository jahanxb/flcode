#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import numpy as np
import torch
from models.Nets import *


################################### model setup ########################################
def model_setup(args):
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar().to(args.device)
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = CIFARResNet20().to(args.device)
    elif args.model == 'cnn' and args.dataset == 'fmnist':
        net_glob = CNNFmnist(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'svhn':
        net_glob = CNNSvhn(args=args).to(args.device)
    elif args.model == 'resnet' and args.dataset == 'svhn':
        net_glob = SVHNResNet20().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in args.img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    return args, net_glob


def model_dim(model):
    '''
    compute model dimension
    '''
    flat = [torch.flatten(model[k]) for k in model.keys()]
    s = 0
    for p in flat:
        s += p.shape[0]
    return s


def model_clip(model, clip):
    '''
    clip model update
    '''
    model_norm = []
    for k in model.keys():
        # print(k)
        if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
            continue
        # print(torch.norm(model[k]), k)
        model_norm.append(torch.norm(model[k]))

    total_norm = torch.norm(torch.stack(model_norm))
    # print('total norm', total_norm)
    clip_coef = clip / (total_norm + 1e-8)
    if clip_coef < 1:
        for k in model.keys():
            if 'num_batches_tracked' in k or 'running_mean' in k or 'running_var' in k:
                continue
            model[k] = model[k] * clip_coef
    return model, total_norm


def set_log_path(args):
    '''
    log path for different datasets and methods
    '''
    path = './log/' + args.dataset + '/' + args.method + '/' + args.model + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    if args.hyper_tune != 1:
        path_log = path + 'round_' + str(args.round) + '_tau_' + str(args.tau) \
                   + '_frac_' + str(args.frac)
    else:
        path_log = path + 'round_' + str(args.round) + '_users_' + str(args.num_users) + '_frac_' + str(
            args.frac) + '_clip_' + str(args.clip) \
                   + '_tau_' + str(args.tau) + '_bs_' + str(args.batch_size) + '_llr_' + str(
            args.local_lr) + '_lm_' + str(args.local_momentum) + '_dw_' + str(args.decay_weight)
    return path_log


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


def get_trainable_values(net, mydevice=None):
    ' return trainable parameter values as a vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    N = 0
    for params in paramlist:
        N += params.numel()
    if mydevice:
        X = torch.empty(N, dtype=torch.float).to(mydevice)
    else:
        X = torch.empty(N, dtype=torch.float)
    X.fill_(0.0)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            X[offset:offset + numel].copy_(params.data.view_as(X[offset:offset + numel].data))
        offset += numel

    return X


def put_trainable_values(net, X):
    ' replace trainable parameter values by the given vector (only the first parameter set)'
    trainable = filter(lambda p: p.requires_grad, net.parameters())
    paramlist = list(trainable)
    offset = 0
    for params in paramlist:
        numel = params.numel()
        with torch.no_grad():
            params.data.copy_(X[offset:offset + numel].data.view_as(params.data))
        offset += numel
