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

from kafka import KafkaProducer, KafkaConsumer
from multiprocessing import Pool, Process, ProcessError, Queue
import pika
from celery import Celery
import pickle,json

from queues_func_list import Node0RabbitQueues as rq0
from queues_func_list import Node1RabbitQueues as rq1

from pymongo import MongoClient


import asyncio
import os,paramiko

from declared_nodes import client_nodes_addr


async def waiting_exception_to_interupt():
    print("Waiting...")
    await asyncio.sleep(5)
    print('....Wait Completed..Raising Exception')
    raise KeyboardInterrupt


async def raise_me():
    task = asyncio.create_task(waiting_exception_to_interupt())
    await task

#asyncio.run(raise_me())


node0 = 0
node1 = 1

def arrange_round_train(args):
    pass



def global_model_aggregate():
    pass

def ack_agent():
    pass

def something_something():
    pass



nodes = 11 # 10 nodes (11 for loop)

local_updates = []
loss_locals = []



def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        time.sleep(body.count(b'.'))
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)


def send_global_round(node_addr,model_path):
    
    localpath = model_path
    remotepath = model_path
    print('Connecting via ssh...')
    ssh = paramiko.SSHClient() 
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(node_addr, username='jahanxb')
    sftp = ssh.open_sftp()
    print('sftp opened...')
    sftp.put(localpath, remotepath)
    sftp.close()
    print('sftp closed.. Model Sent!... ssh connection closing soon..')
    ssh.close()



def serve(args):
    
    
    torch.manual_seed(args.seed+args.repeat)
    torch.cuda.manual_seed(args.seed+args.repeat)
    np.random.seed(args.seed+args.repeat)
    
    args, dataset_train, dataset_test, dict_users = data_setup(args)
    print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of classes:{}'.format(args.num_classes))
    print('num. of users:{}'.format(len(dict_users)))
    
    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
    sample_per_users = 25000
    print('num. of samples per user:{}'.format(sample_per_users))
       
    if args.dataset == 'fmnist' or args.dataset == 'cifar':
        dataset_test, val_set = torch.utils.data.random_split(
            dataset_test, [9000, 1000])
        print(len(dataset_test), len(val_set))
    elif args.dataset == 'svhn':
        dataset_test, val_set = torch.utils.data.random_split(
            dataset_test, [len(dataset_test)-2000, 2000])
        print(len(dataset_test), len(val_set))

    print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
    log_path = set_log_path(args)
    print(log_path)

    args, net_glob = model_setup(args)
    print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])
    
    # ###################################### model initialization ###########################
    print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
    t1 = time.time()
    net_glob.train()
    # copy weights
    global_model = copy.deepcopy(net_glob.state_dict())
    local_m = []
    train_local_loss = []
    test_acc = []
    norm_med = []
    loss_locals = []
    local_updates = []
    delta_norms = []
    
    nodes = 11
    node_index = 1
    num_selected_users = 2
    
    mconn = MongoClient('mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority')
    mdb = mconn['iteration_status']
    
    try:
        mdb.create_collection('master_global')
    except Exception as e:
        print(e)
        pass
    try:
        mdb.create_collection('master_global')
    except Exception as e:
        print(e)
        pass
    
    
    
    
    for t in range(args.round):
        seconds_to_match = 0
        loss_locals = []
        local_updates = []
        delta_norms = []
        m = max(int(args.frac * args.num_users), 1)
        args.local_lr = args.local_lr * args.decay_weight
        selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
        print(selected_idxs)
        num_selected_users = len(selected_idxs)
        
        print("num_selected_users: ",num_selected_users)
        
        for nodeid in range(node_index,nodes):    
            if t==0:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                master_global_for_round = f'master_global_for_node[{nodeid}]_round[{t}]'
            
                msg = pickle.dumps(global_model)

                
                torch.save(msg,f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                send_global_round(client_nodes_addr.get(nodeid),model_path)
                
                
                mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False}
                mdb.master_global.insert_one(mdb_msg)
            
            else:
                pass
            
            
        print(" [x] Sent Round=",t)
        print(f'Round Process Started... Current Round on Master t={t}')
        
        for n in range(node_index,nodes):    
            
            '''LOCAL ROUND CHECK'''
            while True:
                task_id = f'node[{n}]_local_round[{t}]'
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    if status.get('state-ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                        
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            
            '''LOCAL LOSS ROUND CHECK '''
            while True:
                task_id = f'node[{n}]_local_loss_round[{t}]'
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    if status.get('state-ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                       
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            ############################################################################################
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            
            lp = torch.load(f'/mydata/flcode/models/nodes_sftp/nodes_local/node[{n}]_local_round[{t}].pkl')
            lp = list(pickle.loads(lp))
            local_updates.append(lp)
            
            lp_loss = torch.load(f'/mydata/flcode/models/nodes_sftp/nodes_local_loss/node[{n}]_local_loss_round[{t}].pkl')
            lp_loss = list(pickle.loads(lp_loss))
            loss_locals.append(lp_loss[0])
        

        for i in range(num_selected_users):
                global_model = {
                    k: global_model[k] + local_updates[i][0][k] / num_selected_users
                    for k in global_model.keys()
                }

                        
       
        
        print("global_model: ",global_model.get('fc3.bias'))
            
            
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        test_acc_, _ = test_img(net_glob, dataset_test, args)
        test_acc.append(test_acc_)
        train_local_loss.append(sum(loss_locals) / len(loss_locals))
        print('t {:3d}: '.format(t, ))
        print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
        print('Submitting new global model: .....')
    
        # send model to nodes from here
        for nn in range(node_index,nodes):
            master_global_for_round = f'master_global_for_node[{nn}]_round[{t+1}]'
            
            msg = pickle.dumps(global_model)
            
            torch.save(msg,f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl")
            
                
            model_path = f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl"

         
            send_global_round(client_nodes_addr.get(nn),model_path)    
            mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False}
            mdb.master_global.insert_one(mdb_msg)
            print(" [x] Node=", nn," Sent Round=",t+1)

            
            
        t2 = time.time()
        hours, rem = divmod(t2 - t1, 3600) - seconds_to_match
        minutes, seconds = divmod(rem, 60)
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()
            
 


def aggregation_avg(global_model, local_updates):
    '''
    simple average
    '''
    
    # model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    # for i in range(len(local_updates)):
    #     model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    # global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}
    # return global_model
    
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0]}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}
    return global_model



if __name__ == '__main__':
    args = call_parser()

    #user_counter = int(args.num_users / 2)
    # user_counter = 2
    # print("user counter : ", user_counter)

    # server_args = {
    #     0: {
    #         "user_index": user_counter, "dataset": "cifar", "gpu": -1, "round": 3
    #     },
    #     1: {
    #         "user_index": args.num_users, "dataset": "cifar", "gpu": -1, "round": 3
    #     }
    # }
    # args.num_users = user_counter
    serve(args)
