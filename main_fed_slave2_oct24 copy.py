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

from pymongo import MongoClient

from celery import Celery

import pickle, json

from queues_func_list_oct24 import Node1RabbitQueues as rq1
node0 = 0
node1 = 1

list_callback_master_global = []
list_callback_local_global = []
list_callback_global = []
list_callback_local = []
list_callback_local_loss=[]

for i in range(3,10):
    if i==0:
        list_callback_master_global.append('callback_master_global')
        list_callback_local_global.append('callback_local_global')
        list_callback_global.append('callback_global')
        list_callback_local.append('callback_local')
        list_callback_local_loss.append('callback_local_loss')
    else:
        list_callback_master_global.append(f'callback_master_global_{i}')
        list_callback_local_global.append(f'callback_local_global_{i}')
        list_callback_global.append(f'callback_global_{i}')
        list_callback_local.append(f'callback_local_{i}')
        list_callback_local_loss.append(f'callback_local_loss_{i}')


# def pdf_process_function(msg):
#     print("processing")
#     #print(" [x] Received " + str(msg))

#     print('pickle loading started...')
#     gmdl = pickle.loads(msg)
#     #global_model = gmdl
#     print('pickle loading completed...')

#     time.sleep(5) # delays for 5 seconds
#     print("processing finished");
#     return gmdl


# # create a function which is called on incoming messages
# def callback(ch, method, properties, body):
#     gdm = pdf_process_function(body)
#     time.sleep(5)
#     print('[x] press ctrl+c to move to next step')            
#     global_model = gdm
#     torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{0}]_global.pkl")
#     # set up subscription on the queue


def client_node():
    pid = os.getpid()
    # with grpc.insecure_channel("10.10.1.3:9999") as channel:
    #     stub = pb2_grpc.NodeExchangeStub(channel)
    #     request = pb2.fdnode(nodeid=0)
    #     response_node0 = stub.get_args(request)
    #     print(response_node0)

    #     request = pb2.fdnode(nodeid=1)
    #     response_node1 = stub.get_args(request)
    #     print(response_node1)

    try:
            args = call_parser()
            # args.num_users = response_node1.user_index
            # args.gpu = response_node1.gpu
            # args.round = response_node0.round
            # args.dataset = response_node0.dataset
            # args.tau = 10
            # args.frac = 1

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
            
            # producer = KafkaProducer(bootstrap_servers='10.10.1.3:9092')
            # for i in range(1,100):
            #     producer.send('foobar',b'message bytes')
            
            
            # sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))

            # sample_per_users = 0
            # for i in range(response_node1.user_index, len(dict_users)):
            #     sample_per_users += int(sum([len(dict_users[i]) / len(dict_users)]))

            sample_per_users = 5  # for two users , we take 25000 samples as per the loop

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
          
            ###################################### run experiment ##########################

            # initialize data loader
            data_loader_list = []
            print(len(dict_users))
            index = args.num_users
            for i in range(1,2):
            # for i in range(response_node0.user_index,args.num_users):
                print("broke here ")
                dataset = DatasetSplit(dataset_train, dict_users[i])
                ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                data_loader_list.append(ldr_train)
            ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

            m = max(int(args.frac * 1), 1)
            print("m = ",m)
            
            mconn = MongoClient('mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority')
            mdb = mconn['iteration_status']
            try:
                mdb.create_collection('client_node1')
            except Exception as e:
                print(e)
                pass
            
            for t in range(args.round):
                loss_locals = []
                local_updates = []
                delta_norms = []
                if t==0:
                    # ####################### MongoDB Queue Check #######################
                    # try:
                    #     task_id = f'node[{node1}]_global_round[{t}]'
                    #     while True:
                    #         time.sleep(5)
                    #         status = mdb.master_global_for_node1.find_one({'task_id':task_id})
                    #         if status.get('state-ready') == True:
                    #             print('status: ',200,' For :',status.get('task_id'))
                    #         break
                    #     else:
                    #         pass
                    # except Exception as e:
                    #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                    
                    
                    # ###################################################################
                    
                    
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                        
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    
                    
                    
                    
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )

                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)
                    
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                    
                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)
                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')
                
                elif t==1:
                    
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_1,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    
                    mdb_msg = {'task_id':task_queue,'state-ready':True,'consumed':False}
                    mdb.client_node1.insert_one(mdb_msg)

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                    
                    

                    
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')
                
                elif t==2:
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_2,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')      
                elif t==3:
                    
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_3,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')            
                                              
                elif t==4:
                   # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_4,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')           
                                
                elif t==5:
                   # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_5,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')
                
                elif t==6:
                    
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_6,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')

                
                elif t==7:
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_7,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')
                
                elif t==8:
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_8,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')
                elif t==9:
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue
            
                    new_global_model_queue_id = f'master_global_for_node[{node1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_9,
                    auto_ack=True)

                    
                    
                    try:
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()
                    
                    print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)
                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    
                    print("num_selected_users: ",num_selected_users)
                    
                    ###################### local training : SGD for selected users ######################
                    loss_locals = []
                    local_updates = []
                    delta_norms = []
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

            
                    print("local_updates len(): ",len(local_updates))
                
                    ################### Local Model ###################
                
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                
                    task_queue = f'node[{node1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    
            
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    #### Global Model
                    
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{node1}]_global_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(global_model)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    
                    connection.close()
                    print(" [x] Sent Round=",t)

                    
                    # local Loss loss_locals
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    
                    
                    
                    task_queue = f'node[{node1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    
                    
                    channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=message,
                    properties=pika.BasicProperties(delivery_mode=2)
                    )
                    

                    print(" [x] local Loss sent Queue=",t)

                
                    connection.close()
                
                    print(f'Moving to next iteration round t+1:[{t}+1] = {t+1} ')
                    input('Press Any key to start: ')            



    except Exception as e:
            print(f"Exception Thrown: {e}")
            #channel.unsubscribe(close)
            exit(0)


def close(channel):
    channel.close()


if __name__ == '__main__':
    ################################### hyperparameter setup ########################################

    client_node()
