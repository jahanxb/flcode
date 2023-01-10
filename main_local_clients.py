#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
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

warnings.filterwarnings("ignore")
torch.cuda.is_available()

import fdnodes_pb2_grpc as pb2_grpc
import fdnodes_pb2 as pb2

import file_grpc_lib as lib

from kafka import KafkaProducer, KafkaConsumer

import pika

from celery import Celery

import pickle, json

from client import global_node_addr,mongodb_url, client_num_users, training_and_testing_size

from pymongo import MongoClient
import asyncio

import os,paramiko

from cryptography.fernet import Fernet

from kafka import KafkaProducer, KafkaConsumer

import pika

from celery import Celery

import pickle, json

from client import global_node_addr,mongodb_url, client_num_users, training_and_testing_size, cassandra_addr

from pymongo import MongoClient
import asyncio

import os,paramiko

from cryptography.fernet import Fernet

from cassandra.cluster import Cluster
from cassandra.query import named_tuple_factory, dict_factory

casandra_cluster = Cluster([cassandra_addr],port=9042)


import blosc
import zlib

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




def send_local_round(node_addr,model_path):
    
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


def client_node_mongodb():
    pid = os.getpid()
    try:
            args = call_parser()
            NODE_INDEX = 0
            NODE_ID = 1
            
            NODE_INDEX, NODE_ID = client_num_users(args.num_users)
            train_size, test_size, sample_per_users= training_and_testing_size(args.num_users)
            
            print(f"NODE_INDEX: {NODE_INDEX} | NODE_ID: {NODE_ID}")
            print("Active PID : %i" % pid)
            torch.manual_seed(args.seed + args.repeat)
            torch.cuda.manual_seed(args.seed + args.repeat)
            np.random.seed(args.seed + args.repeat)

            args, dataset_train, dataset_test, dict_users = data_setup(args)
                        
            print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
            print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
            print('num. of training data:{}'.format(len(dataset_train)))
            print('num. of testing data:{}'.format(len(dataset_test)))
            print('num. of classes:{}'.format(args.num_classes))
            print('num. of users:{}'.format(len(dict_users)))

            print('arg.num_users:{}'.format(args.num_users))
            
            if args.dataset == 'fmnist' or args.dataset == 'svhn' or args.dataset == 'cifar':
                sample_per_users = 25000
            else:
                pass
            #sample_per_users = 25000  # for two users , we take 25000 samples as per the loop

            
            #train_size, test_size, sample_per_users= training_and_testing_size(args.num_users)
            #print(f"train_size: {train_size} | test_size: {test_size}")
            
            train_size = 9000
            test_size = 1000
            
            
            print('num. of samples per user:{}'.format(sample_per_users))
            if args.dataset == 'fmnist' or args.dataset == 'cifar':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [train_size, test_size])
                # dataset_test, val_set = torch.utils.data.random_split(
                #     dataset_test, [9000, 1000])
                print(len(dataset_test), len(val_set))
            elif args.dataset == 'svhn':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [len(dataset_test) - 2000, 2000])
                print(len(dataset_test), len(val_set))

            print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
            log_path = set_log_path(args)
            print(log_path)

            args, net_glob = model_setup(args)
            print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])

            ###################################### model initialization ###########################
            print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
            t1 = time.time()
            print("Training starting....")
            net_glob.train()
            print("Training completed...")
            print(net_glob.cpu())

            # copy weights
            global_model = copy.deepcopy(net_glob.state_dict())
            local_m = []
            train_local_loss = []
            test_acc = []
            norm_med = []
            loss_locals = []
            local_updates = []
            delta_norms = []
            ###################################### run experiment ##########################


            # initialize data loader
            data_loader_list = []
            print(len(dict_users))
            index = args.num_users
            for i in range(NODE_INDEX,NODE_ID):
            # for i in range(response_node0.user_index,args.num_users):
                print("broke here ")
                dataset = DatasetSplit(dataset_train, dict_users[i])
                ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                data_loader_list.append(ldr_train)
            ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

            mconn = MongoClient(mongodb_url)
            mdb = mconn['iteration_status']
            
            try:
                mdb.create_collection('mongodb_client_cluster')
            except Exception as e:
                print(e)
                pass
            
            m = max(int(args.frac * 1), 1)
            print("m = ",m)
            
            mynode = NODE_ID
            for t in range(args.round):
                seconds_to_match = 0
                loss_locals = []
                local_updates = []
                delta_norms = []
                n = mynode
                
                new_global_model_queue_id = f'master_global_for_node[{n}]_round[{t}]'
                ####################### MongoDB Queue Check #######################
                while True:
                    try:
                        time.sleep(5)
                        seconds_to_match = seconds_to_match + 5
                        status = mdb.master_global.find_one({'task_id':new_global_model_queue_id})
                        if status.get('state-ready') == True:
                            print('status: ',200,' For :',status.get('task_id'))
                            global_model = status.get('data')
                            
                            global_model = zlib.decompress(global_model)
                            global_model_key = status.get('key')
                            
                            print('global_model_key: ',global_model_key)
                            
                            
                            break
                        else:
                            pass
                    except Exception as e:
                        print(f'@ [{new_global_model_queue_id}] | MongoDB Exception Thrown :',e)    
                    
                ###################################################################
                args.local_lr = args.local_lr * args.decay_weight
                selected_idxs = list(np.random.choice(range(1), m, replace=False))
                print("In Round Loop: selected_idxs: ",selected_idxs)
                num_selected_users = len(selected_idxs)
                gm = []
                
                
                fernet = Fernet(global_model_key)
                                
                global_model = fernet.decrypt(global_model)
                
                global_model = pickle.loads(global_model)
                
                print("num_selected_users: ",num_selected_users)
                   
                print("t=",t ," | global_model: ",global_model.keys())
                
                ###################### local training : SGD for selected users ######################
                    
                for i in selected_idxs:
                    print("selected_idx",i)
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

                    local_updates.append(model_update)
                    print("local updates len",len(local_updates), "index",len(local_updates[0]))
                    loss_locals.append(loss)
                norm_med.append(torch.median(torch.stack(delta_norms)).cpu())
                
                t2 = time.time()
                hours, rem = divmod(t2 - t1, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Local training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                #################################
                
                local_model_node = f'node[{n}]_local_round[{t}]'
                    
                key = Fernet.generate_key()
                fernet = Fernet(key)
                
                msg = pickle.dumps(local_updates)
                
                encmsg = fernet.encrypt(msg)
                
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl"
                
                # send local model to global node
                #send_local_round(global_node_addr,model_path=model_path)
                compressed_encmsg = zlib.compress(encmsg)
                size = sys.getsizeof(compressed_encmsg)
                print(f'Size of msg: {size} bytes')
                
                
                mdb_msg = {'task_id':local_model_node,'state-ready':True,'consumed':False,
                           "data":compressed_encmsg, "key":key
                           
                           }
                mdb.mongodb_client_cluster.insert_one(mdb_msg)
                
                ###### loss local
                
                local_loss_node = f'node[{n}]_local_loss_round[{t}]'
                
                key = Fernet.generate_key()
                fernet = Fernet(key)
                
                msg = pickle.dumps(loss_locals)
                
                encmsg = fernet.encrypt(msg)
                
                # send local loss to global Node
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl"
                
                #send_local_round(global_node_addr,model_path=model_path)

                print(" [x] local Loss sent Queue=",t)

                compressed_encmsg = zlib.compress(encmsg)
                size = sys.getsizeof(compressed_encmsg)
                print(f'Size of msg: {size} bytes')
                
                mdb_msg = {'task_id':local_loss_node,'state-ready':True,'consumed':False,
                           "data":compressed_encmsg, "key":key
                           }
                mdb.mongodb_client_cluster.insert_one(mdb_msg)
                
                
                t2 = time.time()
        
        
                dbs_time =  t2 - t1
                # dbs_time = dbs_time - seconds_to_match
                hours, rem = divmod(dbs_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
                time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
                
                result = '\n'+ time_taken+' \n '+'t {:3d}'.format(t) + '\n'
    
                print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                filename = f'/mydata/flcode/node_output/mongodb-node{NODE_ID}-log.txt'
                #tfname = f'touch + {filename}'
                #os.system(tfname)
                with open(filename, 'a') as the_file:
                    the_file.write(result)
                    the_file.close()

    except Exception as e:
            print(f"Exception Thrown: {e}")
            #channel.unsubscribe(close)
            os.system('rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*')
            exit(0)


def client_node_cassandra():
    pid = os.getpid()
    

    try:
            args = call_parser()
            NODE_INDEX = 0
            NODE_ID = 1
            
            NODE_INDEX, NODE_ID = client_num_users(args.num_users)
            train_size, test_size, sample_per_users= training_and_testing_size(args.num_users)
            
            print(f"NODE_INDEX: {NODE_INDEX} | NODE_ID: {NODE_ID}")
            print("Active PID : %i" % pid)
            torch.manual_seed(args.seed + args.repeat)
            torch.cuda.manual_seed(args.seed + args.repeat)
            np.random.seed(args.seed + args.repeat)

            
            
            args, dataset_train, dataset_test, dict_users = data_setup(args)
            
            
            
            print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
            print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
            print('num. of training data:{}'.format(len(dataset_train)))
            print('num. of testing data:{}'.format(len(dataset_test)))
            print('num. of classes:{}'.format(args.num_classes))
            print('num. of users:{}'.format(len(dict_users)))

            print('arg.num_users:{}'.format(args.num_users))
            

            sample_per_users = 25000  # for two users , we take 25000 samples as per the loop

            
            #train_size, test_size, sample_per_users= training_and_testing_size(args.num_users)
            #print(f"train_size: {train_size} | test_size: {test_size}")
            
            train_size = 9000
            test_size = 1000
            
            
            print('num. of samples per user:{}'.format(sample_per_users))
            if args.dataset == 'fmnist' or args.dataset == 'cifar':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [train_size, test_size])
                # dataset_test, val_set = torch.utils.data.random_split(
                #     dataset_test, [9000, 1000])
                print(len(dataset_test), len(val_set))
            elif args.dataset == 'svhn':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [len(dataset_test) - 2000, 2000])
                print(len(dataset_test), len(val_set))

            print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
            log_path = set_log_path(args)
            print(log_path)

            args, net_glob = model_setup(args)
            print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])

            ###################################### model initialization ###########################
            print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
            t1 = time.time()
            print("Training starting....")
            net_glob.train()
            print("Training completed...")
            print(net_glob.cpu())

            # copy weights
            global_model = copy.deepcopy(net_glob.state_dict())
            local_m = []
            train_local_loss = []
            test_acc = []
            norm_med = []
            loss_locals = []
            local_updates = []
            delta_norms = []
            seconds_to_match = 0
            ###################################### run experiment ##########################


            # initialize data loader
            data_loader_list = []
            print(len(dict_users))
            index = args.num_users
            for i in range(NODE_INDEX,NODE_ID):
            # for i in range(response_node0.user_index,args.num_users):
                print("broke here ")
                dataset = DatasetSplit(dataset_train, dict_users[i])
                ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                data_loader_list.append(ldr_train)
            ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

            mconn = MongoClient(mongodb_url)
            mdb = mconn['iteration_status']
            
            try:
                mdb.create_collection('mongodb_client_cluster')
            except Exception as e:
                print(e)
                pass
            
            ########### Check if Cassandra Keyspace exists #############
            while True:           
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    session = casandra_cluster.connect()
                    session.execute("USE iteration_status;")
                    break
                except Exception as e:
                    print(e)
            
            
            ###########################################################
            
            
            
            m = max(int(args.frac * 1), 1)
            print("m = ",m)
            
            mynode = NODE_ID
            for t in range(args.round):
                
                loss_locals = []
                local_updates = []
                delta_norms = []
                n = mynode
                
                new_global_model_queue_id = f'master_global_for_node[{n}]_round[{t}]'

                
                ####################### Cassandra Keyspace table check #######################
                
                
                while True:
                    try:
                        time.sleep(5)
                        seconds_to_match = seconds_to_match + 5
                        #status = mdb.master_global.find_one({'task_id':new_global_model_queue_id})
                        
                        session = casandra_cluster.connect('iteration_status')
                        session.row_factory = dict_factory
                        select_str = f"select task_id,consumed,state_ready,key,data from iteration_status.master_global where task_id = '{new_global_model_queue_id}'; "
                        print("select_Str: ",select_str)
                        stn = session.execute(select_str)
                        
                        
                        status = stn[0]
                        print("STATUS: ",type(status))
                        
                        if status.get('state_ready') == 0:
                            print('status: ',200,' For :',status.get('task_id'))
                            
                            print("key status:",status.get('key'))
                            print("key status type:",type(status.get('key')))
                            
                            
                            keystr = str(status.get('key')).replace('\"',"\'")
                            datastr = str(status.get('data')).replace('\"',"\'")
                            keystr = keystr.replace("b'","")
                            keystr = keystr.replace("'","")
                            
                            datastr = datastr.replace("b'","")
                            datastr = datastr.replace("'","")
                            
                            global_model = bytes(datastr, 'utf-8')
                            global_model_key = bytes(keystr, 'utf-8')
                            
                            print('global_model_key: ',global_model_key)
                            
                            
                            print("key status type:",type(global_model_key))
                            
                            
                            break
                        else:
                            pass
                    except Exception as e:
                        print(f'@ [{new_global_model_queue_id}] | Cassandra Exception Thrown :',e)    
                    
                ###################################################################

                
                args.local_lr = args.local_lr * args.decay_weight
                selected_idxs = list(np.random.choice(range(1), m, replace=False))
                print("In Round Loop: selected_idxs: ",selected_idxs)
                num_selected_users = len(selected_idxs)
                gm = []
                
                
                fernet = Fernet(global_model_key)
                
                #global_model = torch.load(f'/mydata/flcode/models/nodes_trained_model/global_models/{new_global_model_queue_id}.pkl')
                
                global_model = fernet.decrypt(global_model)
                
                global_model = pickle.loads(global_model)
                
                print("num_selected_users: ",num_selected_users)
                   
                print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                
                ###################### local training : SGD for selected users ######################
                    
                for i in selected_idxs:
                    print("selected_idx",i)
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

                    local_updates.append(model_update)
                    print("local updates len",len(local_updates), "index",len(local_updates[0]))
                    loss_locals.append(loss)
                norm_med.append(torch.median(torch.stack(delta_norms)).cpu())
                
                t2 = time.time()
                hours, rem = divmod(t2 - t1, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Local training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                #################################
                
                local_model_node = f'node[{n}]_local_round[{t}]'
                    
                key = Fernet.generate_key()
                fernet = Fernet(key)
                
                msg = pickle.dumps(local_updates)
                
                encmsg = fernet.encrypt(msg)
                
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl"
                
                
                
                ############### Insert local model data on Cassandra #######
                session = casandra_cluster.connect()
                
                print("key: ", key)
                print("key: ", type(key))
                keystr = str(key).replace('\'',"\"")
                datastr = str(encmsg).replace('\'',"\"")
                print("keystr: ",keystr)
                insert_cql = f"""INSERT INTO iteration_status.cass_client_cluster (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+local_model_node+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
                session.execute(insert_cql)
                
                ################################################
                                
                
                ###### loss local
                
                local_loss_node = f'node[{n}]_local_loss_round[{t}]'
                
                key = Fernet.generate_key()
                fernet = Fernet(key)
                
                msg = pickle.dumps(loss_locals)
                
                encmsg = fernet.encrypt(msg)
                
                # send local loss to global Node
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl"
                
          

                print(" [x] local Loss sent Queue=",t)

                
 
                ############### Insert local model loss data on Cassandra #######
                session = casandra_cluster.connect()
                
                print("key: ", key)
                print("key: ", type(key))
                keystr = str(key).replace('\'',"\"")
                datastr = str(encmsg).replace('\'',"\"")
                print("keystr: ",keystr)
                insert_cql = f"""INSERT INTO iteration_status.cass_client_cluster (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+local_loss_node+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
                session.execute(insert_cql)
                
                ################################################
                t2 = time.time()
                dbs_time =  t2 - t1
                # dbs_time = dbs_time - seconds_to_match
                hours, rem = divmod(dbs_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
                time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
                result = '\n'+ time_taken+' \n '+'t {:3d}'.format(t) + '\n'
                print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                filename = f'/mydata/flcode/node_output/cassandra-node{NODE_ID}-log.txt'
                with open(filename, 'a') as the_file:
                    the_file.write(result)
                    the_file.close()
                

    except Exception as e:
            print(f"Exception Thrown: {e}")
            #channel.unsubscribe(close)
            os.system('rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*')
            exit(0)


def client_node_scp():
    pid = os.getpid()
    try:
            args = call_parser()
            NODE_INDEX, NODE_ID = client_num_users(args.num_users)
            train_size, test_size, sample_per_users= training_and_testing_size(args.num_users)
            print("Active PID : %i" % pid)
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
            

            sample_per_users = 25000  # for two users , we take 25000 samples as per the loop

            print('num. of samples per user:{}'.format(sample_per_users))
            if args.dataset == 'fmnist' or args.dataset == 'cifar':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [9000, 1000])
                print(len(dataset_test), len(val_set))
            elif args.dataset == 'svhn':
                dataset_test, val_set = torch.utils.data.random_split(
                    dataset_test, [len(dataset_test) - 2000, 2000])
                print(len(dataset_test), len(val_set))

            print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
            log_path = set_log_path(args)
            print(log_path)

            args, net_glob = model_setup(args)
            print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])

            ###################################### model initialization ###########################
            print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
            t1 = time.time()
            print("Training starting....")
            net_glob.train()
            print("Training completed...")
            print(net_glob.cpu())

            # copy weights
            global_model = copy.deepcopy(net_glob.state_dict())
            local_m = []
            train_local_loss = []
            test_acc = []
            norm_med = []
            loss_locals = []
            local_updates = []
            delta_norms = []
            ###################################### run experiment ##########################

            # initialize data loader
            data_loader_list = []
            print(len(dict_users))
            index = args.num_users
            for i in range(NODE_INDEX,NODE_ID):
            # for i in range(response_node0.user_index,args.num_users):
                print("broke here ")
                dataset = DatasetSplit(dataset_train, dict_users[i])
                ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                data_loader_list.append(ldr_train)
            ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

            mconn = MongoClient(mongodb_url)
            mdb = mconn['iteration_status']
            
            try:
                mdb.create_collection('mongodb_client_cluster')
            except Exception as e:
                print(e)
                pass
            
            m = max(int(args.frac * 1), 1)
            print("m = ",m)
            
            mynode = NODE_ID
            for t in range(args.round):
                loss_locals = []
                local_updates = []
                delta_norms = []
                n = mynode
                
                new_global_model_queue_id = f'master_global_for_node[{n}]_round[{t}]'
                ####################### MongoDB Queue Check #######################
                while True:
                    try:
                        time.sleep(5)
                        status = mdb.master_global.find_one({'task_id':new_global_model_queue_id})
                        if status.get('state-ready') == True:
                            print('status: ',200,' For :',status.get('task_id'))
                            break
                        else:
                            pass
                    except Exception as e:
                        print(f'@ [{new_global_model_queue_id}] | MongoDB Exception Thrown :',e)    
                    
                ###################################################################
                args.local_lr = args.local_lr * args.decay_weight
                selected_idxs = list(np.random.choice(range(1), m, replace=False))
                print("In Round Loop: selected_idxs: ",selected_idxs)
                num_selected_users = len(selected_idxs)
                gm = []
                global_model = torch.load(f'/mydata/flcode/models/nodes_trained_model/global_models/{new_global_model_queue_id}.pkl')
                
                global_model = pickle.loads(global_model)
                
                print("num_selected_users: ",num_selected_users)
                   
                print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                
                ###################### local training : SGD for selected users ######################
                    
                for i in selected_idxs:
                    print("selected_idx",i)
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

                    local_updates.append(model_update)
                    print("local updates len",len(local_updates), "index",len(local_updates[0]))
                    loss_locals.append(loss)
                norm_med.append(torch.median(torch.stack(delta_norms)).cpu())
                
                t2 = time.time()
                hours, rem = divmod(t2 - t1, 3600)
                minutes, seconds = divmod(rem, 60)
                print("Local training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                #################################
                
                local_model_node = f'node[{n}]_local_round[{t}]'
                    
                    
                msg = pickle.dumps(local_updates)
                
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local/{local_model_node}.pkl"
                
                # send local model to global node
                send_local_round(global_node_addr,model_path=model_path)
                
                mdb_msg = {'task_id':local_model_node,'state-ready':True,'consumed':False}
                mdb.mongodb_client_cluster.insert_one(mdb_msg)
                
                ###### loss local
                
                local_loss_node = f'node[{n}]_local_loss_round[{t}]'
                msg = pickle.dumps(loss_locals)
                
                # send local loss to global Node
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/nodes_local_loss/{local_loss_node}.pkl"
                
                send_local_round(global_node_addr,model_path=model_path)

                print(" [x] local Loss sent Queue=",t)

                
                
                mdb_msg = {'task_id':local_loss_node,'state-ready':True,'consumed':False}
                mdb.mongodb_client_cluster.insert_one(mdb_msg)
                
                print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                
                ################################################
                t2 = time.time()
                dbs_time =  t2 - t1
                # dbs_time = dbs_time - seconds_to_match
                hours, rem = divmod(dbs_time, 3600)
                minutes, seconds = divmod(rem, 60)
                print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
                time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
                result = '\n'+ time_taken+' \n '+'t {:3d}'.format(t) + '\n'
                print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                filename = f'/mydata/flcode/node_output/scp-node{NODE_ID}-log.txt'
                with open(filename, 'a') as the_file:
                    the_file.write(result)
                    the_file.close()
                

    except Exception as e:
            print(f"Exception Thrown: {e}")
            #channel.unsubscribe(close)
            os.system('rm -rf /mydata/flcode/models/nodes_trained_model/global_models/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local/* && rm -rf /mydata/flcode/models/nodes_trained_model/nodes_local_loss/*')
            exit(0)


def client_node_rabbitmq():
    print('RabbitMQ on client Node is Pending...')
    exit(0)
    pass


if __name__ == '__main__':
    args = call_parser()
    if args.db == 'mongodb':
        print('Mongodb selected...!')
        client_node_mongodb()
    elif args.db == 'cassandra':
        print('Cassandra Selected...!')
        client_node_cassandra()
    elif args.db == 'scp':
        print('SCP Selected...!')
        client_node_scp()
    elif args.db == 'rabbitmq':
        print('RabbitMQ selected...!')
        client_node_rabbitmq()
    else:
        print('Database Not specified or incorrect entry...!')
        exit(0)