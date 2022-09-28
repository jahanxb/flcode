#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from asyncore import read
import copy
from fileinput import filename
import sys
import threading

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
from collections import OrderedDict
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
        
        net_glob.train()

        new_model = list()
        local_m = []
        train_local_loss = []
        test_acc = []
        norm_med = []
        ####################################### run experiment ##########################

        # initialize data loader
        data_loader_list = []
        print("len(dict_user): ", len(dict_users))
        index = args.num_users
        for i in range(args.num_users):
            dataset = DatasetSplit(dataset_train, dict_users[i])
            ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            data_loader_list.append(ldr_train)
        ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

        m = max(int(args.frac * args.num_users), 1)
        print("m = ",m)
        n = 0
        loss = [0]
        for n in range(2):
            for t in range(args.round):
                args.local_lr = args.local_lr * args.decay_weight
            selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
            print("In Round Loop: selected_idxs: ",selected_idxs)
            num_selected_users = len(selected_idxs)

            ###################### local training : SGD for selected users ######################
            loss_locals = []
            local_updates = []
            delta_norms = []
            
            for i in selected_idxs:
                print(i)
                l_solver = LocalUpdate(args=args)
                net_glob.load_state_dict(global_model)
                # # choose local solver
                # if args.local_solver == 'local_sgd':
                #     new_model, loss = l_solver.local_sgd(
                #         net=copy.deepcopy(net_glob).to(args.device),
                #         ldr_train=data_loader_list[i])
                # # compute local delta
                # print("global_model:",global_model)
                # print("net_glob: ",net_glob)
                
                # new_model = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
                if i==0 and n==0:
                    new_model = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
                else:
                    sm0 = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
                #sm1 = torch.load(f'/mydata/flcode/models/pickles/node{1}[{t}][0].pkl')
                l1 = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{t}][0].pkl')
                #l2 = torch.load(f'/mydata/flcode/models/pickles/node{1}-loss[{t}][0].pkl')
                
                new_model = new_model + sm0
                
                loss = loss + l1
                
                print("liss: ",l1, "len: ",len(l1))
            #     new_model = sm0 + sm1
            #     loss = [l1[0] + l2[0]]
            #     #loss = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{t}][0].pkl')
                
                for o in new_model:
                    newomodel = OrderedDict(o)
                new_model = newomodel
            #     # for l in loss:  
            #     #     newlosso = OrderedDict(l)
            #     # loss = newlosso
            #     # new_model = dict(OrderedDict(new_model))
            #     # loss = dict(OrderedDict(loss))
            #     #print("new_model: ",new_model)
                model_update = {k: new_model[k] - global_model[k] for k in global_model.keys()}
            #     #model_update = {k: new_model[0].get(k) - global_model[k] for k in global_model.keys()}

                # compute local model norm
                delta_norm = torch.norm(
                    torch.cat([
                        torch.flatten(model_update[k])
                        for k in model_update.keys()
                    ]))
                delta_norms.append(delta_norm)

                # clipping local model or not ? : no clip for cifar10
                # threshold = delta_norm / args.clip
                # if threshold > 1.0:
                #     for k in model_update.keys():
                #         model_update[k] = model_update[k] / threshold
                print("loss: ",loss[0])
                local_updates.append(model_update)
                print("local updates len",len(local_updates), "index",len(local_updates[0]))
                loss_locals.append(loss[0])
            norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

            '''
            ####################################
            Download model from node1
            ################################
            '''
            

            '''
            #####################################
            '''

            ##################### communication: avg for all groups #######################
            model_update = {
                k: local_updates[0][k] * 0.0
                for k in local_updates[0].keys()
            }

            
            for i in range(num_selected_users):
                global_model = {
                    k: global_model[k] + local_updates[i][k] / num_selected_users
                    for k in global_model.keys()
                }

            
            print('################## TrainingTest on node0 ######################')
            ##################### testing on global model #######################
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            
            print('t {:3d}: train_loss = , norm = {:.3f}, test_acc = {:.3f}'.
                  format(t, norm_med[-1], test_acc[-1]))

            
            #################################

            t2 = time.time()
            hours, rem = divmod(t2 - t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Local training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            #################################
            
            

        t2 = time.time()
        hours, rem = divmod(t2 - t1, 3600)
        minutes, seconds = divmod(rem, 60)
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        # ##############################
        # ## End of Fedml
        # ###############################
        
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
