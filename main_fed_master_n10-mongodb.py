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


from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from cryptography.fernet import Fernet

import asyncio
import os,paramiko,datetime

from declared_nodes import client_nodes_addr

mongodb_url = 'mongodb://jahanxb:phdunr@130.127.133.239:27017/?authMechanism=DEFAULT&authSource=flmongo&tls=false'

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
    
    #mconn = MongoClient('mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority')
    mconn = MongoClient(mongodb_url)
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
        
        
        ###########
        ## global model keys check###
        #############
        # print('#########global model keys#############')
        # for k in global_model.keys():
        #     print(k)
        
        ########## Implementing Encryption #############
        #HE = Pyfhel(context_params={'scheme':'bfv', 'n':2**14, 't_bits':32})
        #HE = Pyfhel()
        
        # HE = Pyfhel(key_gen=True, context_params={
        #     'scheme': 'CKKS',
        #     'n': 2**14,         # For CKKS, n/2 values can be encoded in a single ciphertext.
        #     'scale': 2**30,     # Each multiplication grows the final scale
        #     'qi_sizes': [60]+ [30]*8 +[60] # Number of bits of each prime in the chain.
        #                 # Intermediate prime sizes should be close to log2(scale).
        #                 # One per multiplication! More/higher qi_sizes means bigger
        #                 #  ciphertexts and slower ops.
        #     })
        # HE.relinKeyGen()
        
        
        # HE.keyGen()
        # HE.rotateKeyGen()
        # HE.relinKeyGen()
        
        #################################################
        
        
        for nodeid in range(node_index,nodes):    
            if t==0:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                master_global_for_round = f'master_global_for_node[{nodeid}]_round[{t}]'
            
                msg = pickle.dumps(global_model)
                
                
                ######## Encryption Step 1 - encrypt data #####################
                # numpyArray = np.array(list(global_model.get('conv1.weight')))
                # c = HE.encrypt(numpyArray)
                # p = HE.encode(master_global_for_round)
                
                # print("1. Creating serializable objects")
                # print(f"  Pyfhel object HE: {HE}")
                # print(f"  PyCtxt c=HE.encrypt([42]): {c}")
                # print(f"  PyPtxt p=HE.encode([-1]): {p}")
                
                # con_size, con_size_zstd   = HE.sizeof_context(),    HE.sizeof_context(compr_mode="zstd")
                # pk_size,  pk_size_zstd    = HE.sizeof_public_key(), HE.sizeof_public_key(compr_mode="zstd")
                # sk_size,  sk_size_zstd    = HE.sizeof_secret_key(), HE.sizeof_secret_key(compr_mode="zstd")
                # rotk_size,rotk_size_zstd  = HE.sizeof_rotate_key(), HE.sizeof_rotate_key(compr_mode="zstd")
                # rlk_size, rlk_size_zstd   = HE.sizeof_relin_key(),  HE.sizeof_relin_key(compr_mode="zstd")
                # c_size,   c_size_zstd     = c.sizeof_ciphertext(),  c.sizeof_ciphertext(compr_mode="zstd")
                # # alternatively, for ciphertext sizes you can use sys.getsizeof(c)

                # print("2. Checking size of serializable objects (with and without compression)")
                # print(f"   - context:    [ \"zstd\"  --> {con_size_zstd} | No compression --> {con_size}]")
                # print(f"   - public_key: [ \"zstd\"  --> {pk_size_zstd} | No compression --> {pk_size}]")
                # print(f"   - secret_key: [ \"zstd\"  --> {sk_size_zstd} | No compression --> {sk_size}]")
                # print(f"   - relin_key:  [ \"zstd\"  --> {rotk_size_zstd} | No compression --> {rotk_size}]")
                # print(f"   - rotate_key: [ \"zstd\"  --> {rlk_size_zstd} | No compression --> {rlk_size}]")
                # print(f"   - c:          [ \"zstd\"  --> {c_size_zstd} | No compression --> {c_size}]")
                
                
                
                ####################################################################
                
                ######## Encryption Step 2 - Save encrypt data #####################
                
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'
                
                
                
                #os.system(f'cd /mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}')
                
                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
                #print('encmsg: ',encmsg)
                
                
                # #tmp_dir = tempfile.TemporaryDirectory()
                # tmp_dir_name = encrypt_key_path

                # # Now we save all objects into files
                # HE.save_context(tmp_dir_name + "/context")
                # HE.save_public_key(tmp_dir_name + "/pub.key")
                # HE.save_secret_key(tmp_dir_name + "/sec.key")
                # HE.save_relin_key(tmp_dir_name + "/relin.key")
                # HE.save_rotate_key(tmp_dir_name + "/rotate.key")
                # c.save(tmp_dir_name + "/c.ctxt")
                # p.save(tmp_dir_name + "/p.ptxt")

                # print("2a. Saving everything into files. Let's check the temporary dir:")
                # print("\n\t".join(os.listdir(tmp_dir_name)))
                
                
                ###################################################################
                
                torch.save(msg,f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                #send_global_round(client_nodes_addr.get(nodeid),model_path)
                
                
                mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                        "conv1.weight":"",
                        "conv1.bias":"",
                        "conv2.weight":"",
                        "conv2.bias":"",
                        "conv3.weight":"",
                        "conv3.bias":"",
                        "fc1.weight":"",
                        "fc1.bias":"",
                        "fc2.weight":"",
                        "fc2.bias":"",
                        "fc3.weight":"",
                        "fc3.bias":"",
                        "data":encmsg,
                        "key":key
                           }
                
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
                    t1 = t1 + 5
                    status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    if status.get('state-ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                        local_model_key = status.get('key')
                        local_model = status.get('data')
                        
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
                    t1 = t1 + 5
                    status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    if status.get('state-ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                        local_model_loss_key = status.get('key')
                        local_model_loss = status.get('data')
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            ############################################################################################
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            
            
            #lp = torch.load(f'/mydata/flcode/models/nodes_sftp/nodes_local/node[{n}]_local_round[{t}].pkl')
            fernet = Fernet(local_model_key)
            lp = fernet.decrypt(local_model)
            
            
            
            lp = list(pickle.loads(lp))
            local_updates.append(lp)
            
            #lp_loss = torch.load(f'/mydata/flcode/models/nodes_sftp/nodes_local_loss/node[{n}]_local_loss_round[{t}].pkl')
            fernet = Fernet(local_model_loss_key)
            lp_loss = fernet.decrypt(local_model_loss)
            
            lp_loss = list(pickle.loads(lp_loss))
            loss_locals.append(lp_loss[0])
        
        print("num_selected_users: ",num_selected_users)
        for i in range(num_selected_users):
                print("i=",i)
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

            
            key = Fernet.generate_key()
            fernet = Fernet(key=key)
                
            encmsg = fernet.encrypt(msg)
                
            print("key: ",key)
         
            #send_global_round(client_nodes_addr.get(nn),model_path)    
            mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                       
                       "conv1.weight":"",
                        "conv1.bias":"",
                        "conv2.weight":"",
                        "conv2.bias":"",
                        "conv3.weight":"",
                        "conv3.bias":"",
                        "fc1.weight":"",
                        "fc1.bias":"",
                        "fc2.weight":"",
                        "fc2.bias":"",
                        "fc3.weight":"",
                        "fc3.bias":"",
                        "data":encmsg,
                        "key":key
                       
                       }
            mdb.master_global.insert_one(mdb_msg)
            print(" [x] Node=", nn," Sent Round=",t+1)

            
        
        t2 = time.time()
        
        
        #dbs_time = datetime.timedelta(seconds=seconds_to_match)
        dbs_time =  t2 - t1
        #dbs_time = dbs_time - seconds_to_match
        hours, rem = divmod(dbs_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        # print("dbs_time: ",dbs_time)
        
        # dhours, drem = divmod(dbs_time, 3600)
        # dminutes, dseconds = divmod(rem, 60)
        # print("dbs time: {:0>2}:{:0>2}:{:05.2f}".format(int(dhours), int(dminutes), dseconds))  
        
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
