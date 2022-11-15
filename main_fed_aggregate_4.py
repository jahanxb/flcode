#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from asyncore import read
import copy
from fileinput import filename
import sys
import threading
from collections import OrderedDict

import grpc
import numpy as np
import time, math
import torch

from utils.data_utils import data_setup, DatasetSplit
from utils.model_utils import *
from utils.aggregation import *
from options import call_parser
from models.Update import LocalUpdate
from models.test import test_img
from torch.utils.data import DataLoader
from concurrent import futures
# from utils.rdp_accountant import compute_rdp, get_privacy_spent
import warnings
import glob
import statistics

warnings.filterwarnings("ignore")
torch.cuda.is_available()



def serve(args):
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    # pb2_grpc.add_NodeExchangeServicer_to_server(ArgsExchange(), server)
    # server.add_insecure_port("[::]:9999")
    # server.start()
    
    # fsl = lib.FileServer()
    # fsl.start()
    
    try:
        
        
        torch.manual_seed(args.seed + args.repeat)
        torch.cuda.manual_seed(args.seed + args.repeat)
        np.random.seed(args.seed + args.repeat)

        args, dataset_train, dataset_test, dict_users = data_setup(args)
        print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
        print(
            'length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
        print('num. of training data:{}'.format(len(dataset_train)))
        print('num. of testing data:{}'.format(len(dataset_test)))
        print('num. of classes:{}'.format(args.num_classes))
        print('num. of users:{}'.format(len(dict_users)))

        print('arg.num_users:{}'.format(args.num_users))
        
        args, net_glob = model_setup(args)
        nodes = 2
        loss_locals = []
        local_updates = []
        delta_norms = []
        net_glob.train()
        train_local_loss = []
        test_acc = []
        norm_med = []
        log_path = set_log_path(args)
        loss = []
        localupdates = []
        print(log_path)

        # copy weights
        global_model = copy.deepcopy(net_glob.state_dict())
        
       
        if args.dataset == 'fmnist' or args.dataset == 'cifar':
            dataset_test, val_set = torch.utils.data.random_split(
                dataset_test, [9000, 1000])
            print(len(dataset_test), len(val_set))
        elif args.dataset == 'svhn':
            dataset_test, val_set = torch.utils.data.random_split(
            dataset_test, [len(dataset_test)-2000, 2000])
            print(len(dataset_test), len(val_set))
        
        t1 = time.time()
        data_loader_list = []
        fc1_bias = 0
        conv1_weight = 0
        conv1_bias = 0
        conv2_weight= 0
        conv2_bias= 0
        conv3_weight= 0
        conv3_bias= 0
        fc1_weight= 0
        fc1_bias= 0
        fc2_weight= 0
        fc2_bias= 0
        fc3_weight= 0
        fc3_bias= 0
        
        
        #m = max(int(args.frac * args.num_users), 1)
        m = max(int(0.5 * 1), 1)
        for t in range(args.round):
            pass
            #args.local_lr = args.local_lr * args.decay_weight
            # selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
        
        
        for i in range(args.num_users):
            dataset = DatasetSplit(dataset_train, dict_users[i])
            ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            data_loader_list.append(ldr_train)
        ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
        
        for n in range(nodes):
            for t in range(args.round):
                
                args.local_lr = args.local_lr * args.decay_weight
                selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
                print(selected_idxs)
                num_selected_users = len(selected_idxs)
                #selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
                
                print(f'appending node{n}{t}')
                #localupdates.append(torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl'))
                #loss.append(torch.load(f'/mydata/flcode/models/pickles/node{0}-loss[{t}][0].pkl'))
                localupdates = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
                losstorch = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{t}][0].pkl')
                for o in localupdates:
                    localupdates = OrderedDict(o)
                
                
                #local_updates = local_updates + localupdates
                loss = loss + losstorch
                local_updates = localupdates
                
                if t==0 and n==0:
                    pass
                else:
                    for key,item in local_updates.items():
                        print(key)
                        
                        if key == 'conv1.weight':
                            conv1_weight = conv1_weight + local_updates.get('conv1.weight')
                        if key == 'conv1_bias':
                            conv1_bias = conv1_bias + local_updates.get('conv1.bias')
                        if key == "conv2_weight":
                            conv2_weight = conv2_weight + local_updates.get('conv2.weight')
                        if key == 'conv2.bias':
                            conv2_bias = conv2_bias + local_updates.get('conv2.bias')
                        if key== 'conv3.weight':
                            conv3_weight = conv3_weight + local_updates.get('conv3.weight')
                        if key == 'conv3.bias':
                            conv3_bias = conv3_bias + local_updates.get('conv3.bias')
                        if key == 'fc1.weight':
                            fc1_weight = fc1_weight + local_updates.get('fc1.weight')
                        if key == 'fc1.bias':
                            fc1_bias = fc1_bias + local_updates.get('fc1.bias')
                        if key == 'fc2.weight':
                            fc2_weight = fc2_weight + local_updates.get('fc2.weight')
                        if key == 'fc2.bias':
                            fc2_bias = fc2_bias + local_updates.get('fc2.bias')
                        if key == 'fc1.weight':
                            fc3_weight = fc3_weight + local_updates.get('fc3.weight')
                        if key == 'fc3.bias':
                            fc3_bias = fc3_bias + local_updates.get('fc3.bias')
                        
                        # model_up = {
                        #     key: item + item
                        # }
        model_up =  OrderedDict({
          "conv1.weight":conv1_weight,
            "conv1.bias":conv1_bias,
            "conv2.weight":conv2_weight,
            "conv2.bias": conv2_bias,
            "conv3.weight": conv3_weight,
            "conv3.bias": conv3_bias,
            "fc1.weight":fc1_weight,
            "fc1.bias":fc1_bias,
            "fc2.weight":fc2_weight,
            "fc2.bias":fc2_bias,
            "fc3.weight":fc3_weight,
            "fc3.bias":fc3_bias
        })
        
        #print("local_update type: ",type(local_updates))
        local_updates = local_updates        
        count = 0
        for local in local_updates:
            #print("local_updates: ",local_updates)
            #print(f"count: {count} | local 'fc1.bias'",local.get('fc1.bias'))
            count = count + 1
        
        
        # for i in selected_idxs:
        #     net_glob.load_state_dict(global_model)
        #     #print("localupdates: ",localupdates)
        #     model_update = {k: localupdates[k] - global_model[k] for k in global_model.keys()}
        #     #model_update = {k: localupdates[t].get(k) for k in global_model.keys()}
        #     #model_update = localupdates
                    
        #     # compute local model norm
        #     delta_norm = torch.norm(
        #                 torch.cat([
        #                 torch.flatten(model_update[k])
        #                 for k in model_update.keys()
        #             ]))
        #     delta_norms.append(delta_norm)
        #     #local_updates.append(model_update)
        #     #loss_locals.append(loss)
        #     #print("local updates len",len(local_updates), "index",len(local_updates[0]))
        #     norm_med.append(torch.median(torch.stack(delta_norms)).cpu())
        
        print("local_updates: ",local_updates)
        model_update = {
            k: local_updates[k] * 0.0
            for k in local_updates.keys()
        }
        for i in range(num_selected_users):
            global_model = {
                k: global_model[k] + local_updates[i][k] / num_selected_users
                for k in global_model.keys()
            }
        
        
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        
        test_acc_, _ = test_img(net_glob, dataset_test, args)
       
        test_acc.append(test_acc_) 
        
        #train_local_loss.append(sum(loss_locals) / len(loss_locals))
        
        train_local_loss = [0,0,0]
        
        # print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
        #                 format(t, train_local_loss, norm_med, test_acc))
        # print("train_loss: ",sum(train_local_loss)/len(train_local_loss))
        # print("test_acc: ",sum(test_acc)/len(test_acc))
        # print("norm_med: ",sum(norm_med)/len(norm_med))
        
        
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                        format(t, sum(train_local_loss), sum(norm_med), sum(test_acc)))
        
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                        format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))

        t2 = time.time()
        hours, rem = divmod(t2 - t1, 3600)
        minutes, seconds = divmod(rem, 60)
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        


def aggregation_avg(global_model, local_updates):
    '''
    simple average
    '''
    
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}
    return global_model

if __name__ == '__main__':
    args = call_parser()

    #user_counter = int(args.num_users / 2)
    user_counter = 2
    print("user counter : ", user_counter)

    server_args = {
        0: {
            "user_index": user_counter, "dataset": "cifar", "gpu": -1, "round": 10
        },
        1: {
            "user_index": args.num_users, "dataset": "cifar", "gpu": -1, "round": 10
        }
    }
    args.num_users = user_counter
    serve(args)
