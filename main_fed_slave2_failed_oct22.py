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

from queues_func_list import Node1RabbitQueues as rq1

node1 = 1
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
#     torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{1}]_global.pkl")
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
            for t in range(args.round):
                
                if t==0:
                    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    

            
                    new_global_model_queue_id = f'master_global_for_node[{1}]_round[{t}]'
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
                    #gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
                    # gm.append(gm0)
                    # #gm.append(gm1)
            
                    # print("gm: ",gm[0].get('fc3.bias'))
                    # #print("gm: ",gm[1].get('fc3.bias'))

                    # for i in range(num_selected_users):
                    #     global_model = {
                    #         k: global_model[k] + gm[0][k]
                    #     for k in global_model.keys()
                    # }
                    
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

                
                    model_update = {
                        k: local_updates[0][k] * 0.0
                        for k in local_updates[0].keys()
                    }
                    for i in range(num_selected_users):
                        global_model = {
                            k: global_model[k] + local_updates[i][k] / num_selected_users
                            for k in global_model.keys()
                        }

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
                
                    task_queue = f'node[{1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    #print('task_queue',task_queue)
            
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
                    
                    task_queue = f'node[{1}]_global_round[{t}]'
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
                    
                    
                    
                    
                    task_queue = f'node[{1}]_local_loss_round[{0}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
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
                
                elif t==1:
                    #### Get aggregated Global Model #####
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue

                    new_global_model_queue_id = f'master_global_for_node[{1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_1,
                    auto_ack=True)
                    
                    
                    
                    # start consuming (blocks)
                    #channel.start_consuming()
                    
                    #connection.close()
                    
                    try:
                        
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()

                    
                    
                    
                    #####################################
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)

                    
                    gm = []
                    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    #gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
                    # gm.append(gm0)
                    # #gm.append(gm1)
            
                    # print("gm: ",gm[0].get('fc3.bias'))
                    # #print("gm: ",gm[1].get('fc3.bias'))

                    # for i in range(num_selected_users):
                    #     global_model = {
                    #         k: global_model[k] + gm[0][k]
                    #     for k in global_model.keys()
                    # }
                    
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

                    #### Global Model
                    print("dumb global model:",global_model)
                    print(type(global_model))
                    
                    
                
                    model_update = {
                        k: local_updates[0][k] * 0.0
                        for k in local_updates[0].keys()
                    }
                    ###################
                    ### This loop is creating problem
                    ####################
                    # for i in range(num_selected_users):
                    #     global_model = {
                    #         k: global_model[k] + local_updates[i][k] / num_selected_users
                    #         for k in global_model.keys()
                    #     }
                    ###########################
                    
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
                
                    task_queue = f'node[{1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    #print('task_queue',task_queue)
                
                
                
                
                    channel.basic_publish(
                        exchange='',
                        routing_key=task_queue,
                        body=message,
                        properties=pika.BasicProperties(delivery_mode=2)
                    )
                
                    connection.close()
                
                
                    
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    
                    task_queue = f'node[{1}]_global_round[{t}]'
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
                    
                    
                    
                    
                    task_queue = f'node[{1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
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
                
                elif t==2:
                    
                    #### Get aggregated Global Model #####
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue

                    new_global_model_queue_id = f'master_global_for_node[{1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global_2,
                    auto_ack=True)

                    
                    
                    # start consuming (blocks)
                    #channel.start_consuming()
                    
                    #connection.close()
                    
                    try:
                        
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()

                    
                    
                    #####################################
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)

                    
                    gm = []
                    gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{t}].pkl')
                    #gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
                    gm.append(gm0)
                    #gm.append(gm1)
            
                    print("gm: ",gm[0].get('fc3.bias'))
                    #print("gm: ",gm[1].get('fc3.bias'))

                    for i in range(num_selected_users):
                        global_model = {
                            k: global_model[k] + gm[0][k]
                        for k in global_model.keys()
                    }
                    
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

                
                    model_update = {
                        k: local_updates[0][k] * 0.0
                        for k in local_updates[0].keys()
                    }
                    
                    # for i in range(num_selected_users):
                    #     global_model = {
                    #         k: global_model[k] + local_updates[i][k] / num_selected_users
                    #         for k in global_model.keys()
                    #     }

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
                
                    task_queue = f'node[{1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    #print('task_queue',task_queue)
                
                
                
                
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
                    
                    task_queue = f'node[{1}]_global_round[{t}]'
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
                    
                    
                    
                    
                    task_queue = f'node[{1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
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
                
                else:
                    break
                    #### Get aggregated Global Model #####
                    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
                    params = pika.URLParameters(url)
                    connection = pika.BlockingConnection(params)
                    channel = connection.channel() # start a channel
                    #channel.queue_declare(queue='global_model_round_queue_[0][0]') # Declare a queue

                    new_global_model_queue_id = f'master_global_for_node[{1}]_round[{t}]'
                    channel.basic_consume(new_global_model_queue_id,
                    rq1.callback_master_global,
                    auto_ack=True)

                    
                    
                    # start consuming (blocks)
                    #channel.start_consuming()
                    
                    #connection.close()
                    
                    try:
                        
                        channel.start_consuming()
                    except KeyboardInterrupt:
                        channel.stop_consuming()
                        connection.close()

                    
                    
                    #####################################
                    
                    
                    args.local_lr = args.local_lr * args.decay_weight
                    selected_idxs = list(np.random.choice(range(1), m, replace=False))
                    print("In Round Loop: selected_idxs: ",selected_idxs)
                    num_selected_users = len(selected_idxs)

                    
                    gm = []
                    gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{1}]_round[{t}].pkl')
                    #gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
                    gm.append(gm0)
                    #gm.append(gm1)
            
                    print("gm: ",gm[0].get('fc3.bias'))
                    #print("gm: ",gm[1].get('fc3.bias'))

                    for i in range(num_selected_users):
                        global_model = {
                            k: global_model[k] + gm[0][k]
                        for k in global_model.keys()
                    }
                    
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

                
                    model_update = {
                        k: local_updates[0][k] * 0.0
                        for k in local_updates[0].keys()
                    }
                    for i in range(num_selected_users):
                        global_model = {
                            k: global_model[k] + local_updates[i][k] / num_selected_users
                            for k in global_model.keys()
                        }

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
                
                    task_queue = f'node[{1}]_local_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    
                
                
                    msg = pickle.dumps(local_updates)
                
                
                    message = msg
                
                    #print('task_queue',task_queue)
                
                
                
                
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
                    
                    task_queue = f'node[{1}]_global_round[{t}]'
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
                    
                    
                    
                    
                    task_queue = f'node[{1}]_local_loss_round[{t}]'
                    channel.queue_declare(queue=task_queue, durable=True)
                    msg = pickle.dumps(loss_locals)
                    
                    
                    message = msg
                    
                    #print('task_queue',task_queue)
                    
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
