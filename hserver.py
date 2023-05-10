#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.8
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

#DataLake based Hoptions call_parser 
from hoptions import call_parser

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
from cassandra.policies import RoundRobinPolicy

import asyncio
import os,paramiko,datetime


from Pyfhel import Pyfhel, PyPtxt, PyCtxt

from cryptography.fernet import Fernet

import asyncio
import os,paramiko,datetime, zlib


import psycopg2

from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable

master_node_ip = '10.10.1.1'
client_nodes_addr = {
    1:'10.10.1.2',2:'10.10.1.3', 3:'10.10.1.4',4:'10.10.1.5',5:'10.10.1.6',
    6:'10.10.1.7',7:'10.10.1.8',8:'10.10.1.9',9:'10.10.1.10',10:'10.10.1.11'
    }
mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/?authMechanism=DEFAULT&authSource=admin&tls=false'
mongodb_connection = MongoClient(mongodb_url)

cassandra_addr = '10.10.1.2'
#casandra_cluster = Cluster([cassandra_addr],port=9042, connect_timeout=120.0)
socket_options = {'connect_timeout': 30.0, 'read_timeout': 60.0}
load_balancing_policy = RoundRobinPolicy()

casandra_cluster = Cluster([cassandra_addr], port=9042, 
                           load_balancing_policy=load_balancing_policy,connect_timeout=30.0
                           )


nodes = 11
node_index = 1
num_selected_users = 2


class MetaDataEtl:
    
    @staticmethod
    def fetch_metadata(**kwargs):
        pass
    
    @staticmethod
    def insert_metadata(sizeofmodel,call_parser_args,master_global_for_round,encryption_key):
        mdb = mongodb_connection['global_model']
    
        try:
            mdb.create_collection('metadata')
        except Exception as e:
            print(e)
            pass
        
        args_dict = vars(call_parser_args)
        md = {
        
            'global_model_ip':master_node_ip,
            'task_id':master_global_for_round,
            'model_size':sizeofmodel,
            #'key':encryption_key
        }
        metadata = {**md, **args_dict}
        print("metadata: \n",metadata)
        metadata['device'] = str(metadata['device']) 
        stat = mdb.metadata.insert_one(metadata)
        
        print(stat)
        
    
    @staticmethod
    def global_model_status(task_id,status,consumed,encryption_key):
        mdb = mongodb_connection['global_model']
    
        try:
            mdb.create_collection('global_model_training_status')
        except Exception as e:
            print(e)
            pass
        
        training_status = {
            'model_id':"",
            "metadata_id":"",
            "task_id":task_id,
            "status": status,
            "consumed":consumed,
            'key':encryption_key
        }
        
        mdb.global_model_training_status.insert_one(training_status)
        
        
        
    
    @staticmethod
    def etl_pipeline(**kwargs):
        pass
    
    @staticmethod
    def client_training_status_check(task_id):
        #check training completion status on client nodes
        mdb = mongodb_connection['local_model']
        try:
            status = mdb.local_model_training_status.find_one({'task_id':task_id})
            return status
        
        except Exception as e:
            print(e)
        
    
    
    @staticmethod
    def storage_service(**kwargs):
        #unstructed data Storage 
        pass
    
    
    @staticmethod
    def fetch_client_updates(**kwargs):
        #fetch client updates from different clients from Apache Cassandra
        pass
    
    def metadata_etl(args):
        # Metadata pipeline made on NoSQL (JSON) MongoDB
        # Centralized Pipeline to Keep track of process
        # Handles Storage of Global Model and Global Model Updates
        # Handles Storage of Local Model and Local Model Updates
        # Also, handles Storage of Unstructed Data 
        # Keeps the data Heterogenity (Keeping data Heterogenous and same among all platforms )  
        pass


class ReplicationEngine:
    
    @staticmethod
    def replication_engine(args):
        # Decentralized Data replication (Storage for Structured data / Semi Structured data)
        # Global Model replications and Serves as the main data source to all client Nodes ( Availability for all clients so that server don't choke)
        pass
    
    @staticmethod
    def data_replication_operation(**kwargs):
        pass
    
    @staticmethod
    def data_storage(mode,data,key,task_id):
        # Take Unstructure Storage from MongoDB and then distribute it to Cassandra for Clients 
        # This is a decentralized storage and also a structured Storage 
        ############### Insert data on Cassandra #######
        session = casandra_cluster.connect()
        print("key: ", key)
        print("key: ", type(key))
        keystr = str(key).replace('\'',"\"")
        datastr = str(data).replace('\'',"\"")
        print("keystr: ",keystr)
        insert_cql = f"""INSERT INTO iteration_status.master_global (task_id, state_ready, consumed , key, data) 
                              VALUES ({"'"+task_id+"'"}, {0}, {-1} , {"'"+keystr+"'"}, {"'"+datastr+"'"} ); """
        session.execute(insert_cql)
                
        ################################################
    
    
    @staticmethod
    def create_data_storage_schema(**kwargs):
        # create schema for storage for both clients and global server
        '''
        create keyspace iteration_status with replication {'class':'SimpleStrategy','replication_factor':4}; # replication factor depends on num of nodes 
        '''
        
        '''
        USE iteration_status;
        '''

        global_table_cassandra_string = '''
                CREATE TABLE master_global (
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
            session.execute("CREATE KEYSPACE IF NOT EXISTS iteration_status with replication = {'class':'SimpleStrategy','replication_factor':1};")
        except Exception as e:
            print(e)
        
    
        try:
            session = casandra_cluster.connect('iteration_status',wait_for_all_pools=True)
        
            try:
                session.execute('DROP TABLE iteration_status.master_global')
                session.execute(global_table_cassandra_string)
            except:
                session.execute(global_table_cassandra_string)
                pass
    
        except Exception as e:
            print(e)
            pass
    
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
            
            
            
    
    
    def schema_func(**kwargs):
        # Handles Schema 
        pass
    

class DataLakeStorageEngine:
    
    @staticmethod
    def datalake_storage_engine(**kwargs):
        # Centralized RDBMS (Postgres) keeping all data and binding relationship 
        # Keep score of accuracy and built relationship and act as Datalake Storage 
        pass
    
    @staticmethod
    def create_schema(**kwargs):
        # Create Schema for the training set 
        pass
    
    @staticmethod
    def persistent_storage_unit(**kwargs):
        # Writes data on SQL based Storage 
        pass
    
    @staticmethod
    def schema_relation(**kwargs):
        # Creates relationship of training between clients and global model 
        # we can also store model in byte-string form or just store the model ObjectID from MongoDB (This needs some work I will get back to it later)
        pass
    
    


def serve_global_model(args):
    print('... Intializing Apache Cassandra Schema for Clients(Local Models) and Master (Global Model) Node ')
    ReplicationEngine.create_data_storage_schema()
    print('........... Apache Cassandra Schema Created Successfully..... ')
    
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
            
                ######## Encryption- Save encrypt data #####################    
                encrypt_key_path = f'/mydata/flcode/models/node_encrypted/global_models/{master_global_for_round}'

                key = Fernet.generate_key()
                fernet = Fernet(key=key)
                
                encmsg = fernet.encrypt(msg)
                
                print("key: ",key)
 
                torch.save(msg,f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl")
                
                model_path = f"/mydata/flcode/models/nodes_sftp/global_models/{master_global_for_round}.pkl"

                # send model to nodes from here 
                print("mongodb_client_cluster.get() =",client_nodes_addr.get(nodeid))
                #print("show me data : ",encmsg)
                compressed_encmsg = zlib.compress(encmsg)
                #compressed_encmsg = encmsg
                size = sys.getsizeof(compressed_encmsg)
                print(f'Size of msg: {size} bytes')
                
                
                
                # Create Meta data and send it to MongoDB 
                MetaDataEtl.insert_metadata(sizeofmodel=size,call_parser_args = args,master_global_for_round=master_global_for_round,encryption_key=key)
                
                ### Send Model data and Key to Apache Cassandra 
                ReplicationEngine.data_storage(mode='global',data=compressed_encmsg,key=key,task_id=master_global_for_round)

                ### Update the model status once its being ready and sent 
                MetaDataEtl.global_model_status(task_id=master_global_for_round,status=True,consumed=False,encryption_key=key)
                
                ### Here you can add data related to global model Storage 
                
                #mdb.master_global.insert_one(mdb_msg) 
                           
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
                    ### Find mongodb client info 
                    status = MetaDataEtl.client_training_status_check(task_id=task_id)
                    
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
            
    
    


if __name__ == '__main__':
    args = call_parser()
    print('Heterogenous Data Lake check ')
    serve_global_model(args)
    
    # if args.db == 'mongodb':
    #     print('Mongodb selected...!')
    #     serve_mongodb(args)
    # elif args.db == 'cassandra':
    #     print('Cassandra Selected...!')
    #     serve_cassandra(args)
    # elif args.db == 'scp':
    #     print('SCP Selected...!')
    #     serve_scp(args)
    # elif args.db == 'rabbitmq':
    #     print('RabbitMQ selected...!')
    #     server_rabbitmq(args)
    # elif args.db == 'postgres':
    #     print('Postgres selected...!')
    #     serve_postgres(args)
    # elif args.db == 'neo4j':
    #     print('Neo4j Selected...!')
    #     test_neo4j(args)
    # else:
    #     print('Database Not specified or incorrect entry...!')
    #     exit(0)
        