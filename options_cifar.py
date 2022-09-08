#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--dataset',
                        type=str,
                        default='cifar',
                        help="name of dataset")
    parser.add_argument('--iid',
                        type=int,
                        default=1,
                        help='whether i.i.d or not')
    parser.add_argument('--num_users',
                        type=int,
                        default=100,
                        help="number of users: K")
    parser.add_argument('--frac',
                        type=float,
                        default=0.1,
                        help="the fraction of clients: C")
    parser.add_argument('--num_data',
                        type=int,
                        default=500,
                        help="number of data per user: m")
    
    
    # model arguments
    parser.add_argument('--method',
                        type=str,
                        default='fedavg',
                        help='method name')
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--local_solver',
                        type=str,
                        default='local_sgd',
                        help='local solver method')
    parser.add_argument('--global_solver',
                        type=str,
                        default='avg',
                        help="aggregation rule")
    ################################################# may need to re-tune tau, batch_size, local_lr
    # local solver hyperparameter
    parser.add_argument('--tau',
                        type=int,
                        default=10,
                        help="num. of local epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=50,
                        help="local batch size")
    parser.add_argument('--local_lr',
                        type=float,
                        default=0.125,
                        help="local learning rate")
    parser.add_argument('--local_momentum',
                        type=float,
                        default=0.8,
                        help="SGD momentum (default: 0.5)")
    parser.add_argument('--decay_weight',
                        type=float,
                        default=0.99,
                        help="learning rate decay weight (default: 0.5)")
    
    # global solver hyperparameter
    parser.add_argument('--round',
                        type=int,
                        default=200,
                        help="rounds of training")
    parser.add_argument('--clip',
                        type=float,
                        default=1.0,
                        help='clipping threshold')
    # other
    parser.add_argument('--gpu',
                        type=int,
                        default=1,
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="seed")
    parser.add_argument('--repeat', type=int, default=1, help='repeat index')
    parser.add_argument('--hyper_tune',
                        type=int,
                        default=0,
                        help=" tuning hyperparameter? ")

    args = parser.parse_args()
    return args


def call_parser():
    args = args_parser()
    return args
