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
from cryptography.fernet import Fernet
from cassandra.cluster import Cluster
from cassandra.query import named_tuple_factory, dict_factory

import asyncio
import os,paramiko,datetime


from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from cryptography.fernet import Fernet

import asyncio
import os,paramiko,datetime, zlib

from declared_nodes import client_nodes_addr, mongodb_url, cassandra_addr
import psycopg2

from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable


casandra_cluster = Cluster([cassandra_addr],port=9042)


async def waiting_exception_to_interupt():
    print("Waiting...")
    await asyncio.sleep(5)
    print('....Wait Completed..Raising Exception')
    raise KeyboardInterrupt


async def raise_me():
    task = asyncio.create_task(waiting_exception_to_interupt())
    await task



async def waiting_exception_to_interupt():
    print("Waiting...")
    await asyncio.sleep(5)
    print('....Wait Completed..Raising Exception')
    raise KeyboardInterrupt


async def raise_me():
    task = asyncio.create_task(waiting_exception_to_interupt())
    await task


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


def serve_cassandra(args):
    
    
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
    #num_selected_users = 2
    
        
    ####### create or drop Cassandra KEYSPACE ################
    '''
    create keyspace iteration_status with replication {'class':'SimpleStrategy','replication_factor':10}; # replication factor depends on num of nodes 
    '''
    '''
    USE iteration_status;
    '''
    
    table_string = '''
    CREATE TABLE master_global (
        task_id text,
        state_ready boolean,
        consumed boolean,
        conv1_weight varchar,
        conv1_bias varchar,
        conv2_weight varchar,
        conv2_bias varchar,
        conv3_weight varchar,
        conv3_bias varchar,
        fc1_weight varchar,
        fc1_bias varchar,
        fc2_weight varchar,
        fc2_bias varchar,
        fc3_weight varchar,
        fc3_bias varchar,
        data varchar,
        key varchar,
        
        PRIMARY KEY (task_id)
    );
    
    '''
    
    
    table_string = '''
    CREATE TABLE master_global (
        task_id text,
        state_ready int,
        consumed int,
        key text,
        data text,
        PRIMARY KEY (task_id)
    );
    
    '''
    
    table_string_client = '''
    CREATE TABLE cass_client_cluster (
                task_id text,
                state_ready int,
                consumed int,
                key text,
                data text,
                PRIMARY KEY (task_id)
                );
    
                '''
    
    
    
    
    try:
        session = casandra_cluster.connect()
        session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':10};")
    except Exception as e:
        print(e)
        
    
    try:
        session = casandra_cluster.connect('iteration_status',wait_for_all_pools=True)
        
        try:
            session.execute('DROP TABLE iteration_status.master_global')
            session.execute(table_string)
        except:
            session.execute(table_string)
            pass
    
    except Exception as e:
        print(e)
        pass
    
    
    
    try:
        session = casandra_cluster.connect('iteration_status',wait_for_all_pools=True)
        
        try:
            session.execute('DROP TABLE iteration_status.cass_client_cluster')
            session.execute(table_string_client)
        except:
            session.execute(table_string_client)
            pass
    
    except Exception as e:
        print(e)
        pass
    
    
    
    
    #########################################################

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
                
                
                ####################################################################
                
                ######## Encryption Step 2 - Save encrypt data #####################
                
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'
                
                
                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
                
                ###################################################################
                
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

                
                ############### Insert data on Cassandra #######
                session = casandra_cluster.connect()
                
                print("key: ", key)
                print("key: ", type(key))
                keystr = str(key).replace('\'',"\"")
                datastr = str(encmsg).replace('\'',"\"")
                print("keystr: ",keystr)
                insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+master_global_for_round+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        
                # print('insert_cql: ',insert_cql)
                
                session.execute(insert_cql)
                
                ################################################
                
            
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
                    session = casandra_cluster.connect('iteration_status')
                    session.row_factory = dict_factory
                    select_str = f"select task_id,consumed,state_ready,key,data from iteration_status.cass_client_cluster where task_id = '{task_id}'; "
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
                            
                            
                            
                        local_model = bytes(datastr, 'utf-8')
                        local_model_key = bytes(keystr, 'utf-8')
                            
                            
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | Cassandra Exception Thrown :',e)
            
            
            
                                
            
            
            '''LOCAL LOSS ROUND CHECK '''
            
            while True:
                task_id = f'node[{n}]_local_loss_round[{t}]'
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    t1 = t1 + 5
                    session = casandra_cluster.connect('iteration_status')
                    session.row_factory = dict_factory
                    select_str = f"select task_id,consumed,state_ready,key,data from iteration_status.cass_client_cluster where task_id = '{task_id}'; "
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
                            
                            
                            
                        local_model_loss = bytes(datastr, 'utf-8')
                        local_model_loss_key = bytes(keystr, 'utf-8')
                            
                            
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | Cassandra Exception Thrown :',e)
            
            
            
            
            
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            
            
            fernet = Fernet(local_model_key)
            lp = fernet.decrypt(local_model)
            
            
            
            lp = list(pickle.loads(lp))
            local_updates.append(lp)
            
            
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
        
        print("global_model: ",global_model.keys())
            
            
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
            
            torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
            
                
            model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

            
            key = Fernet.generate_key()
            fernet = Fernet(key=key)
                
            encmsg = fernet.encrypt(msg)
                
            print("key: ",key)
            print("key: ", key)
            print("key: ", type(key))
            keystr = str(key).replace('\'',"\"")
            datastr = str(encmsg).replace('\'',"\"")
            print("keystr: ",keystr)
            insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+master_global_for_round+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        
                
            session.execute(insert_cql)
            
            
            
            
            print(" [x] Node=", nn," Sent Round=",t+1)

            
        
        t2 = time.time()
        
        
        #dbs_time = datetime.timedelta(seconds=seconds_to_match)
        dbs_time =  t2 - t1
        #dbs_time = dbs_time - seconds_to_match
        hours, rem = divmod(dbs_time, 3600)
        minutes, seconds = divmod(rem, 60)
                
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/output/cassandra-10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()
            
 

def serve_mongodb(args):
    
    
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
                
        for nodeid in range(node_index,nodes):    
            if t==0:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                master_global_for_round = f'master_global_for_node[{nodeid}]_round[{t}]'
            
                msg = pickle.dumps(global_model)
                

                
                ####################################################################
                
                ######## Encryption Step 2 - Save encrypt data #####################
                
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'

                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
 
                torch.save(msg,f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                #send_global_round(client_nodes_addr.get(nodeid),model_path)
                
                #compressed_pickle = blosc.compress(encmsg)
                
                compressed_encmsg = zlib.compress(encmsg)
                size = sys.getsizeof(compressed_encmsg)
                print(f'Size of msg: {size} bytes')
               
                mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                        
                        "data":compressed_encmsg,
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
                        local_model = zlib.decompress(local_model)
                        
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
                        local_model_loss = zlib.decompress(local_model_loss)
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

                        
       
        
        print("global_model: ",global_model.keys())
            
            
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

            compressed_encmsg = zlib.compress(encmsg)
            #send_global_round(client_nodes_addr.get(nn),model_path)    
            mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                       
                        "data":compressed_encmsg,
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
         
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/output/mongodb-10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()
            
 




def serve_scp(args):
    
    
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
    sample_per_users = 5
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
    
    nodes = 2
    node_index = 1
    #num_selected_users = 2
    
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
        loss_locals = []
        local_updates = []
        delta_norms = []
        m = max(int(args.frac * args.num_users), 1)
        args.local_lr = args.local_lr * args.decay_weight
        selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
        print(selected_idxs)
        num_selected_users = len(selected_idxs)
        
        for nodeid in range(1,11):    
            if t==0:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                master_global_for_round = f'master_global_for_node[{nodeid}]_round[{t}]'
            
                msg = pickle.dumps(global_model)
            
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                send_global_round(client_nodes_addr.get(nodeid),model_path)
                
                
                mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False}
                mdb.master_global.insert_one(mdb_msg)
            
            else:
                pass
            
            
        print(" [x] Sent Round=",t)
        print(f'Round Process Started... Current Round on Master t={t}')
        
        for n in range(1,11):    
            ######################### Check status of Queues through MongoDB ############################
            '''GLOBAL ROUND CHECK'''
            
            '''LOCAL ROUND CHECK'''
            while True:
                task_id = f'node[{n}]_local_round[{t}]'
                try:
                    time.sleep(5)
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
            
            lp = torch.load(f'/mydata/flcode/models/nodes_trained_model/nodes_local/node[{n}]_local_round[{t}].pkl')
            lp = list(pickle.loads(lp))
            local_updates.append(lp)
            
            lp_loss = torch.load(f'/mydata/flcode/models/nodes_trained_model/nodes_local_loss/node[{n}]_local_loss_round[{t}].pkl')
            lp_loss = list(pickle.loads(lp_loss))
            loss_locals.append(lp_loss[0])
    
            
        for i in range(num_selected_users):
            global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
        print("global_model: ",global_model.keys())
            
            
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
        for nn in range(1,11):
            master_global_for_round = f'master_global_for_node[{nn}]_round[{t+1}]'
            
            msg = pickle.dumps(global_model)
            
            torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
            
                
            model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"
         
            send_global_round(client_nodes_addr.get(nn),model_path)    
            mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False}
            mdb.master_global.insert_one(mdb_msg)
            print(" [x] Node=", nn," Sent Round=",t+1)
            
        t2 = time.time()
        
        
        #dbs_time = datetime.timedelta(seconds=seconds_to_match)
        dbs_time =  t2 - t1
        #dbs_time = dbs_time - seconds_to_match
        hours, rem = divmod(dbs_time, 3600)
        minutes, seconds = divmod(rem, 60)
                
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/output/scp-10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()


            
def server_rabbitmq(args):
    print('rabbitMQ pending...')
    exit(0)
    pass


def serve_postgres(args):
    
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
    #num_selected_users = 2
    
        
    ####### create or drop Cassandra KEYSPACE ################
    '''
    create keyspace iteration_status with replication {'class':'SimpleStrategy','replication_factor':10}; # replication factor depends on num of nodes 
    '''
    '''
    USE iteration_status;
    '''
    
    table_string = '''
    CREATE TABLE master_global (
        task_id text,
        state_ready boolean,
        consumed boolean,
        conv1_weight varchar,
        conv1_bias varchar,
        conv2_weight varchar,
        conv2_bias varchar,
        conv3_weight varchar,
        conv3_bias varchar,
        fc1_weight varchar,
        fc1_bias varchar,
        fc2_weight varchar,
        fc2_bias varchar,
        fc3_weight varchar,
        fc3_bias varchar,
        data varchar,
        key varchar,
        
        PRIMARY KEY (task_id)
    );
    
    '''
    
    
    table_string = '''
    CREATE TABLE iteration_status.master_global (
        task_id text,
        state_ready int,
        consumed int,
        key text,
        data text,
        PRIMARY KEY (task_id)
    );
    
    '''
    
    table_string_client = '''
    CREATE TABLE iteration_status.client_cluster (
                task_id text,
                state_ready int,
                consumed int,
                key text,
                data text,
                PRIMARY KEY (task_id)
                );
    
                '''
    
    
    
    
    try:
        conn = psycopg2.connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "35.224.200.63", port = "5432")
        print ("Opened database successfully")
        curr = conn.cursor()
        curr.execute('DROP TABLE IF EXISTS iteration_status.master_global;')
        curr.execute(table_string)
        conn.commit()
        conn.close()
        #session = casandra_cluster.connect()
        #session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':10};")
    except Exception as e:
        print(e)
        
    try:
        conn = psycopg2.connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "35.224.200.63", port = "5432")
        print ("Opened database successfully")
        curr = conn.cursor()
        curr.execute('DROP TABLE IF EXISTS iteration_status.client_cluster;')
        curr.execute(table_string_client)
        conn.commit()
        conn.close()
        #session = casandra_cluster.connect()
        #session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':10};")
    except Exception as e:
        print(e)


    session = psycopg2.connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "35.224.200.63", port = "5432")
    
    #########################################################

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
                
                
                ####################################################################
                
                ######## Encryption Step 2 - Save encrypt data #####################
                
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'
                
                
                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
                
                ###################################################################
                
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

                
                ############### Insert data on postgres #######
                #session = casandra_cluster.connect()
                cur = session.cursor()


                print("key: ", key)
                print("key: ", type(key))
                keystr = str(key).replace('\'',"\"")
                datastr = str(encmsg).replace('\'',"\"")
                print("keystr: ",keystr)
                insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+master_global_for_round+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        
                # print('insert_cql: ',insert_cql)
                
                cur.execute(insert_cql)
                session.commit()
                ################################################
                
            
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
                    cur = session.cursor()
                    #session.row_factory = dict_factory
                    select_str = f"select task_id,consumed,state_ready,key,data from iteration_status.client_cluster where task_id = '{task_id}'; "
                    cur.execute(select_str)
                    rows = cur.fetchall()
                    columns = [col[0] for col in cur.description]
                    statusl = list()
                    for row in rows:
                        statusl.append(dict(zip(columns, row)))
                    #print("select_Str: ",select_str)
                    #cur.execute(select_str)
                    #session.commit()
                        
                        
                    status = statusl[0]
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
                            
                            
                            
                        local_model = bytes(datastr, 'utf-8')
                        local_model_key = bytes(keystr, 'utf-8')
                            
                            
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | Postgres Exception Thrown :',e)
            
            
            
                                
            
            
            '''LOCAL LOSS ROUND CHECK '''
            
            while True:
                task_id = f'node[{n}]_local_loss_round[{t}]'
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    t1 = t1 + 5
                    #session = casandra_cluster.connect('iteration_status')
                    #session.row_factory = dict_factory
                    select_str = f"select task_id,consumed,state_ready,key,data from iteration_status.client_cluster where task_id = '{task_id}'; "
                    cur.execute(select_str)
                    rows = cur.fetchall()
                    columns = [col[0] for col in cur.description]
                    statusl = list()
                    for row in rows:
                        statusl.append(dict(zip(columns, row)))
                    #print("select_Str: ",select_str)
                    #stn = session.execute(select_str)
                    
                    print("status list",statusl)
                        
                    status = statusl[0]
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
                            
                            
                            
                        local_model_loss = bytes(datastr, 'utf-8')
                        local_model_loss_key = bytes(keystr, 'utf-8')
                            
                            
                        break
                    else:
                        pass
                except Exception as e:
                    print(f'@ [{task_id}] | Postgres Exception Thrown :',e)
            
            
            
            
            
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            
            
            fernet = Fernet(local_model_key)
            lp = fernet.decrypt(local_model)
            
            
            
            lp = list(pickle.loads(lp))
            local_updates.append(lp)
            
            
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
        
        print("global_model: ",global_model.keys())
            
            
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
            
            torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
            
                
            model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

            
            key = Fernet.generate_key()
            fernet = Fernet(key=key)
                
            encmsg = fernet.encrypt(msg)
                
            print("key: ",key)
            print("key: ", key)
            print("key: ", type(key))
            keystr = str(key).replace('\'',"\"")
            datastr = str(encmsg).replace('\'',"\"")
            print("keystr: ",keystr)
            #insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
            #                  VALUES ({"'"+master_global_for_round+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        
                
            #session.execute(insert_cql)
            
            
            cur = session.cursor()


            #print("key: ", key)
            #print("key: ", type(key))
            #keystr = str(key).replace('\'',"\"")
            #datastr = str(encmsg).replace('\'',"\"")
            print("keystr: ",keystr)
            insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+master_global_for_round+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        
                # print('insert_cql: ',insert_cql)
                
            cur.execute(insert_cql)
            session.commit()
            
            
            
            
            print(" [x] Node=", nn," Sent Round=",t+1)

            
        
        t2 = time.time()
        
        
        #dbs_time = datetime.timedelta(seconds=seconds_to_match)
        dbs_time =  t2 - t1
        #dbs_time = dbs_time - seconds_to_match
        hours, rem = divmod(dbs_time, 3600)
        minutes, seconds = divmod(rem, 60)
                
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/output/postgres-10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()


# def serve_postgres_1(args):
#     #conn = psycopg2.connect(database = "ddfl", user = "postgres", password = "postgres", host = "130.127.133.239", port = "5432")
#     # print ("Opened database successfully")

#     # cur = conn.cursor()
#     # cur.execute('''CREATE TABLE master_global.COMPANY
#     #   (ID INT PRIMARY KEY     NOT NULL,
#     #   NAME           TEXT    NOT NULL,
#     #   AGE            INT     NOT NULL,
#     #   ADDRESS        CHAR(50),
#     #   SALARY         REAL);''')
#     # print ("Table created successfully")

#     # conn.commit()
#     # conn.close()
#     table_string = '''
#     CREATE TABLE iteration_status.master_global (
#         task_id text,
#         state_ready int,
#         consumed int,
#         key text,
#         data text,
#         PRIMARY KEY (task_id)
#     );
    
#     '''
    
#     table_string_client = '''
#     CREATE TABLE iteration_status.client_cluster (
#                 task_id text,
#                 state_ready int,
#                 consumed int,
#                 key text,
#                 data text,
#                 PRIMARY KEY (task_id)
#                 );
    
#                 '''
    
    
    
    
#     try:
#         conn = psycopg2.connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "104.198.252.184", port = "5432")
#         print ("Opened database successfully")
#         curr = conn.cursor()
#         curr.execute('DROP TABLE IF EXISTS iteration_status.master_global;')
#         curr.execute(table_string)
#         conn.commit()
#         conn.close()
#         #session = casandra_cluster.connect()
#         #session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':10};")
#     except Exception as e:
#         print(e)
        
#     try:
#         conn = psycopg2.connect(database = "ddfl", user = "postgres", password = "ng.dB.Q'3s`^9HVx", host = "104.198.252.184", port = "5432")
#         print ("Opened database successfully")
#         curr = conn.cursor()
#         curr.execute('DROP TABLE IF EXISTS iteration_status.client_cluster;')
#         curr.execute(table_string_client)
#         conn.commit()
#         conn.close()
#         #session = casandra_cluster.connect()
#         #session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':10};")
#     except Exception as e:
#         print(e)


def test_neo4j(args):
    uri_neo4j = "neo4j://10.10.1.10:7687"
    user_neo4j = "neo4j"
    password_new4j = "oi2KksBMaHfsB355HdoHsI2Kzv4NoOUm7MnPNtnESIY"
    
    driver = GraphDatabase.driver(uri_neo4j, auth=(user_neo4j, password_new4j))
    
    
        
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
    
    
    '''
    Create Neo4J database graph and drop previous ones 
    '''
    
    table_string = '''
    CREATE TABLE iteration_status.master_global (
        task_id text,
        state_ready int,
        consumed int,
        key text,
        data text,
        PRIMARY KEY (task_id)
    );
    
    '''
    
    table_string_client = '''
    CREATE TABLE iteration_status.client_cluster (
                task_id text,
                state_ready int,
                consumed int,
                key text,
                data text,
                PRIMARY KEY (task_id)
                );
    
                '''
    try:
        pass
    except Exception as e:
        pass
    
    #driver.session("CREATE OR REPLACE DATABASE ddfl")
    
    
    def create_master_node(tx, task_id, state_ready, consumed, key, data):
        result = tx.run(
        "CREATE (g:global_model {task_id: $task_id, state_ready: $state_ready, consumed: $consumed, data: $data, key: $key})",
        task_id = task_id,state_ready = state_ready, consumed = consumed, data = data, key = key
        )
        summary = result.consume()        
        return summary
    
    def get_client_node(tx, task_id):
        result = tx.run("MATCH (c:client_cluster {task_id: $task_id}) RETURN c.task_id, c.state_ready,c.consumed,c.key,c.data",task_id=task_id)
        records = list(result)  # a list of Record objects
        summary = result.consume()
        return records, summary 
    
    
    
    
    
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
                

                
                ####################################################################
                
                ######## Encryption Step 2 - Save encrypt data #####################
                
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'

                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
 
                torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                #send_global_round(client_nodes_addr.get(nodeid),model_path)
                
                #compressed_pickle = blosc.compress(encmsg)
                
                #compressed_encmsg = zlib.compress(encmsg)
                compressed_encmsg = encmsg
                size = sys.getsizeof(compressed_encmsg)
                print(f'Size of msg: {size} bytes')
               
                # mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                        
                #         "data":compressed_encmsg,
                #         "key":key
                #            }
                
                task_id = master_global_for_round
                state_ready = True
                consumed = False
                data = compressed_encmsg
                key = key
                
                with driver.session(database="neo4j") as session:
                    summary = session.execute_write(create_master_node, task_id=task_id, state_ready=state_ready, consumed=consumed,key=key,data=data)
                    print("Created {nodes_created} nodes in {time} ms.".format(
                    nodes_created=summary.counters.nodes_created,
                    time=summary.result_available_after
                    ))
                
            
            else:
                pass
            
            
        print(" [x] Sent Round=",t)
        print(f'Round Process Started... Current Round on Master t={t}')
        
        for n in range(node_index,nodes):    
            
            '''LOCAL ROUND CHECK'''
            while True:
                task_id = f'node[{n}]_local_round[{t}]'
                
                
                status = dict()
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    t1 = t1 + 5
                    #status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    
                    with driver.session(database="neo4j") as session:
                        records, summary = session.execute_read(get_client_node,task_id=task_id)

                        # Summary information
                        print("The query `{query}` returned {records_count} records in {time} ms.".format(
                                query=summary.query, records_count=len(records),
                                time=summary.result_available_after))

                        #print("records: ",records)
                        
                        # Loop through results and do something with them
                        for task in records:
                            #print(task.data())  # obtain record as dict
                            status = task.data()
                            print("inside loop  ")
                            print("STATUS: ",type(status))
                            print("STATUS: ",status.keys())
                            print("STATUS: ",status.get('c.state_ready'))    
                            print("loop end")
                        
                        status = dict(task_id=status.get('c.task_id'),state_ready=status.get('c.state_ready'),
                                      consumed=status.get("c.consumed"),key=status.get("c.key"),data=status.get('c.data'))
                        
                        print("STATUS: ",type(status))
                        print("STATUS: ",status.keys())
                        print("STATUS: ",status.get('c.state_ready'))
                        print("task id: ",task_id)
                        
                        
                        
                    
                    
                    
                    
                    if status.get('state_ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                        local_model_key = status.get('key')
                        print('local_model_key: ',local_model_key)
                        #local_model_key = str(local_model_key).replace('\'',"\"")
                        local_model = status.get('data')
                        #local_model = zlib.decompress(local_model)
                        
                        break
                    else:
                        pass
                    
                except Exception as e:
                    print(f'@ [{task_id}] | Neo4j [local model] Exception Thrown :',e)
                    
            
            
            '''LOCAL LOSS ROUND CHECK '''
            while True:
                task_id = f'node[{n}]_local_loss_round[{t}]'
                status = dict()
                try:
                    time.sleep(5)
                    seconds_to_match = seconds_to_match + 5
                    t1 = t1 + 5
                    #status = mdb.mongodb_client_cluster.find_one({'task_id':task_id})
                    
                    with driver.session(database="neo4j") as session:
                        records, summary = session.execute_read(get_client_node,task_id=task_id)

                        # Summary information
                        print("The query `{query}` returned {records_count} records in {time} ms.".format(
                                query=summary.query, records_count=len(records),
                                time=summary.result_available_after))

                        
                        # Loop through results and do something with them
                        for task in records:
                            #print(task.data())  # obtain record as dict
                            status = task.data()
                            print("inside loop  ")
                            print("STATUS: ",type(status))
                            print("STATUS: ",status.keys())
                            print("STATUS: ",status.get('c.state_ready'))    
                            print("loop end")
                             
                        status = dict(task_id=status.get('c.task_id'),state_ready=status.get('c.state_ready'),
                                      consumed=status.get("c.consumed"),key=status.get("c.key"),data=status.get('c.data'))
                        
                        print("STATUS: ",type(status))
                        print("STATUS: ",status.keys())
                        print("STATUS: ",status.get('c.state_ready'))
                        
                        
                        
                    
                    
                    if status.get('state_ready') == True:
                        print('status: ',200,' For :',status.get('task_id'))
                        local_model_loss_key = status.get('key')
                        print('local_model_loss_key: ',local_model_loss_key)
                        #local_model_loss_key = str(local_model_loss_key).replace('\'',"\"")
                        local_model_loss = status.get('data')
                        #local_model_loss = zlib.decompress(local_model_loss)
                        break
                        
                    else:
                        pass
                        
                except Exception as e:
                    print(f'@ [{task_id}] | Neo4j [local-loss] Exception Thrown :',e)
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

                        
       
        
        print("global_model: ",global_model.keys())
            
            
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
            
            torch.save(msg,f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl")
            
                
            model_path = f"/mydata/flcode/models/nodes_trained_model/global_models/{master_global_for_round}.pkl"

            
            key = Fernet.generate_key()
            fernet = Fernet(key=key)
                
            encmsg = fernet.encrypt(msg)
                
            print("key: ",key)

            #compressed_encmsg = zlib.compress(encmsg)
            compressed_encmsg = encmsg
            #send_global_round(client_nodes_addr.get(nn),model_path)    
            # mdb_msg = {'task_id':master_global_for_round,'state-ready':True,'consumed':False,
                       
            #             "data":compressed_encmsg,
            #             "key":key
                       
            #            }
            # mdb.master_global.insert_one(mdb_msg)
            
            task_id = master_global_for_round
            state_ready = True
            consumed = False
            data = compressed_encmsg
            key = key
                
            with driver.session(database="neo4j") as session:
                summary = session.execute_write(create_master_node, task_id=task_id, state_ready=state_ready, consumed=consumed,key=key,data=data)
                print("Created {nodes_created} nodes in {time} ms.".format(
                    nodes_created=summary.counters.nodes_created,
                    time=summary.result_available_after
                    ))
            
            
            
            
            print(" [x] Node=", nn," Sent Round=",t+1)

            
        
        t2 = time.time()
        
        #dbs_time = datetime.timedelta(seconds=seconds_to_match)
        dbs_time =  t2 - t1
        #dbs_time = dbs_time - seconds_to_match
        hours, rem = divmod(dbs_time, 3600)
        minutes, seconds = divmod(rem, 60)
         
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))   
        time_taken = "training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        result = '\n'+ time_taken+' \n '+'t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.format(t, train_local_loss[-1], test_acc[-1]) + '\n'
    
        with open('/mydata/flcode/output/neo4j-10nodes-results-log.txt', 'a') as the_file:
            the_file.write(result)
            the_file.close()

    
    
    
    

    


if __name__ == '__main__':
    args = call_parser()
    if args.db == 'mongodb':
        print('Mongodb selected...!')
        serve_mongodb(args)
    elif args.db == 'cassandra':
        print('Cassandra Selected...!')
        serve_cassandra(args)
    elif args.db == 'scp':
        print('SCP Selected...!')
        serve_scp(args)
    elif args.db == 'rabbitmq':
        print('RabbitMQ selected...!')
        server_rabbitmq(args)
    elif args.db == 'postgres':
        print('Postgres selected...!')
        serve_postgres(args)
    elif args.db == 'neo4j':
        print('Neo4j Selected...!')
        test_neo4j(args)
    else:
        print('Database Not specified or incorrect entry...!')
        exit(0)
        