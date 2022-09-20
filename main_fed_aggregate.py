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
        
        for n in range(nodes):
            for t in range(args.round):
                print(f'appending node{n}{t}')
                local_updates.append(torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl'))
                loss_locals.append(torch.load(f'/mydata/flcode/models/pickles/node{0}-loss[{t}][0].pkl'))
                
        
        sample_per_users = 5 #12500  # for two users , we take 25000 samples as per the loop
        num_selected_users = len(local_updates)

        t1 = time.time()
        data_loader_list = []
        print("len(dict_user): ", len(dict_users))
        index = args.num_users
        for i in range(args.num_users):
            dataset = DatasetSplit(dataset_train, dict_users[i])
            ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            data_loader_list.append(ldr_train)
        ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
        
        
        m = max(int(args.frac * args.num_users), 1)
        for t in range(args.round):
            args.local_lr = args.local_lr * args.decay_weight
            selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
            
        
        for i in selected_idxs:
            print(i)
            l_solver = LocalUpdate(args=args)
            net_glob.load_state_dict(global_model)
                # choose local solver
            if args.local_solver == 'local_sgd':
                new_model, loss = l_solver.local_sgd(
                        net=copy.deepcopy(net_glob).to(args.device),
                        ldr_train=data_loader_list[i])
                # compute local delta
            model_update = {k: new_model[k] - global_model[k] for k in global_model.keys()}

            # compute local model norm
            delta_norm = torch.norm(
                    torch.cat([
                        torch.flatten(model_update[k])
                        for k in model_update.keys()
                    ]))
            delta_norms.append(delta_norm)
        
        
        
        print("num_selected_users,",num_selected_users)
        print("global_model ",global_model.keys())
        print("local sucks: ",local_updates[0][0].get('fc3.bias'))
        
        for i in range(num_selected_users):
            print("i = ",i)
            #print("local update: ",local_updates[i])
            # global_model = {
                
            # }    
            global_model = {
                    # k: global_model[k] + local_updates[i][k] / num_selected_users
                    k: global_model[k] + local_updates[i][0].get(k) / num_selected_users
                    for k in global_model.keys()
                    }
            
            
        #global_model = aggregation_avg(global_model, local_updates)
        
        # m = max(int(args.frac * args.num_users), 1)
        # for t in range(args.round):
        #     args.local_lr = args.local_lr * args.decay_weight
        #     selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
            
        #     num_selected_users = len(selected_idxs)
        print("num_selected_users: ",num_selected_users)
        
        # print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
        # ##################### testing on global model #######################
        # net_glob.load_state_dict(global_model)
        # net_glob.eval()
        # test_acc_, _ = test_img(net_glob, dataset_test, args)
        # test_acc.append(test_acc_)
        
        # newll = list()
        # for i in range(len(loss_locals)):
        #     newll.append(sum(loss_locals[i]))
        
        # loss_locals = newll
        

        
        # train_local_loss.append(sum(loss_locals) / len(loss_locals))
        
        # print("t=",t)
        # print("train_loss: ",train_local_loss)
        
        # print("norm_med: ",norm_med) # local training of SGD 
        
        # print("test_acc: ",test_acc)
        
        # print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
        #           format(t, train_local_loss[-1], test_acc[-1]))

        # if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
        #     np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
        #                    test_acc,
        #                    delimiter=",")
        #     np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
        #                    train_local_loss,
        #                    delimiter=",")
        #     np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
        #     #break;

        # # print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
        # #           format(t, sum(train_local_loss) / len(train_local_loss), sum(norm_med) / len(norm_med), sum(test_acc) / len(test_acc)))
        
        
        # t2 = time.time()
        # hours, rem = divmod(t2 - t1, 3600)
        # minutes, seconds = divmod(rem, 60)
        # print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                

    
    
    
        print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
        ##################### testing on global model #######################
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        test_acc_, _ = test_img(net_glob, dataset_test, args)
        test_acc.append(test_acc_)
        
        newll = list()
        for i in range(len(loss_locals)):
            newll.append(sum(loss_locals[i]))
        
        loss_locals = newll
        
        train_local_loss.append(sum(loss_locals) / len(loss_locals))
        # print('t {:3d}: '.format(t, ))
        print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                  format(t, train_local_loss[-1], test_acc[-1]))

        if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
            np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
                           test_acc,
                           delimiter=",")
            np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
                           train_local_loss,
                           delimiter=",")
            np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
            #break;

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
