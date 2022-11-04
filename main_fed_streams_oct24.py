#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from asyncore import read
import copy
from fileinput import filename
import sys
import threading
from typing import OrderedDict

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



nodes = 2
# global local_updates
# global loss_locals
local_updates = []
loss_locals = []

# print('t {:3d}: train_loss = {:.3f}, norm = Not Recording, test_acc = {:.3f}'.
#                   format(t, train_local_loss[0], test_acc[0]))

# print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
#                 format(t, train_local_loss[t], norm_med[t], test_acc[t]))



def callback(ch, method, properties, body):
        print(" [x] Received %r" % body)
        time.sleep(body.count(b'.'))
        print(" [x] Done")
        ch.basic_ack(delivery_tag=method.delivery_tag)



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
    sample_per_users = 5
    print('num. of samples per user:{}'.format(sample_per_users))
    
    # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    # parameters = pika.ConnectionParameters('130.127.134.6',
    #                                5672,
    #                                '/',
    #                                credentials)

    # connection = pika.BlockingConnection(parameters)
    # #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
    # channel = connection.channel()

    # channel.queue_declare(queue='task_queue', durable=True)

    # message = "Hello World! This might help me"

    # channel.basic_publish(
    # exchange='',
    # routing_key='task_queue',
    # body=message,
    # properties=pika.BasicProperties(delivery_mode=2)
    # )

    # print(" [x] Sent %r" % message)
    # connection.close()
    
    
    '''
    yaha se connection pakrna ha 
    
    '''
    # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    # parameters = pika.ConnectionParameters('130.127.134.6',
    #                                5672,
    #                                '/',
    #                                credentials)
    # # ye wala code waise master node p hone chaiyee ab , q k ye ab receive kare ga round 
    # connection = pika.BlockingConnection(parameters)
    # channel = connection.channel()
            
    # channel.queue_declare(queue='task_queue',durable=True)
    # print(' [*] Waiting for messages. To exit press')
            
    

    # channel.basic_qos(prefetch_count=1)
    # channel.basic_consume(queue='task_queue', on_message_callback=callback)

    # channel.start_consuming()
    '''
    chalo idr end kar do icay 
    
    '''
    
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
    
    
    
    credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

    connection = pika.BlockingConnection(parameters)
    #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
    channel = connection.channel()

    nodes = 2
    
    num_selected_users = 2
    
    mconn = MongoClient('mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority')
    mdb = mconn['iteration_status']
    #mdb.create_collection('master_node')
    
    for t in range(args.round):
        loss_locals = []
        local_updates = []
        delta_norms = []
        m = max(int(args.frac * args.num_users), 1)
        args.local_lr = args.local_lr * args.decay_weight
        selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
        print(selected_idxs)
        num_selected_users = len(selected_idxs)
        
        if t==0:
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
            
            # ######################### Check status of Queues through MongoDB ############################
            # '''GLOBAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
                
            # try:
            #     task_id = f'node[{node1}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # try:
            #     task_id = f'node[{node1}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL LOSS ROUND CHECK '''
            # try:
            #     task_id = f'node[{node0}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
            
            # try:
            #     task_id = f'node[{node1}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            
            
            # ############################################################################################
            
            
                    
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
            #lm = local_updates
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            gm.append(gm0)
            gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + gm[0][k] + gm[1][k]
            #         for k in global_model.keys()
            #     }
            
            #print("local updates 0: ",local_updates[0].get('fc3.bias'))
            #print("local updates 1: ",local_updates[1].get('fc3.bias'))  
                
            # for i in range(len(lm)):
            #     global_model = {
            #         k: local_updates[k] + lm[i][k]
            #         for k in local_updates.keys()
            #     }
    
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
    
           

            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) 
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",t+1)
            
            connection.close()
            
            
                
        elif t==1:
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            # parameters = pika.ConnectionParameters('130.127.134.6',
            #                        5672,
            #                        '/',
            #                        credentials)

            # connection = pika.BlockingConnection(parameters)
            # #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            # channel = connection.channel()
            
            # master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            # task_queue = master_global_for_node0_round
            
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
            # master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            # task_queue = master_global_for_node1_round
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            #input('Insert Any Key.. If global Model is delivered...: ')
            
            # ######################### Check status of Queues through MongoDB ############################
            # '''GLOBAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
                
            # try:
            #     task_id = f'node[{node1}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # try:
            #     task_id = f'node[{node1}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL LOSS ROUND CHECK '''
            # try:
            #     task_id = f'node[{node0}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
            
            # try:
            #     task_id = f'node[{node1}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            
            
            # ############################################################################################

            
            
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_1,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_1,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_1,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_1,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_1,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_1,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            gm.append(gm0)
            gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])

            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) 
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            
            
            print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",t+1)
            
            connection.close()
            
        elif t==2:
            
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            # parameters = pika.ConnectionParameters('130.127.134.6',
            #                        5672,
            #                        '/',
            #                        credentials)

            # connection = pika.BlockingConnection(parameters)
            # #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            # channel = connection.channel()
            
            # master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            # task_queue = master_global_for_node0_round
            
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
            # master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            # task_queue = master_global_for_node1_round
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            #input('Insert Any Key.. If global Model is delivered...: ')
            
            
            # ######################### Check status of Queues through MongoDB ############################
            # '''GLOBAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         print('status: ',status)
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
                
            # try:
            #     task_id = f'node[{node1}]_global_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL ROUND CHECK'''
            # try:
            #     task_id = f'node[{node0}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # try:
            #     task_id = f'node[{node1}]_local_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            # '''LOCAL LOSS ROUND CHECK '''
            # try:
            #     task_id = f'node[{node0}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node0.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
                
            
            # try:
            #     task_id = f'node[{node1}]_local_loss_round[{t}]'
            #     while True:
            #         time.sleep(5)
            #         status = mdb.client_node1.find_one({'task_id':task_id})
            #         if status.get('state-ready') == True:
            #             print('status: ',200,' For :',status.get('task_id'))
            #             break
            #         else:
            #             pass
            # except Exception as e:
            #     print(f'@ [{task_id}] | MongoDB Exception Thrown :',e)
            
            
            
            # ############################################################################################

                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_2,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_2,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_2,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_2,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_2,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_2,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            gm.append(gm0)
            gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) 
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",t+1)
            
            connection.close()
        
        
        elif t==3:
            
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            # parameters = pika.ConnectionParameters('130.127.134.6',
            #                        5672,
            #                        '/',
            #                        credentials)

            # connection = pika.BlockingConnection(parameters)
            # #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            # channel = connection.channel()
            
            # master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            # task_queue = master_global_for_node0_round
            
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
            # master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            # task_queue = master_global_for_node1_round
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(global_model)
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=msg,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_3,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_3,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_3,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_3,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_3,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_3,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
    
            #print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",t+1)
            
            connection.close()     
                
        elif t==4:
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_4,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_4,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_4,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_4,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_4,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_4,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            gm.append(gm0)
            gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            # loss_locals.append(lp0_loss[0])
            # loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()        
 
        elif t==5:
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_5,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_5,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_5,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_5,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_5,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_5,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            # loss_locals.append(lp0_loss[0])
            # loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()
        
        elif t==6:
            
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_6,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_6,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_6,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_6,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_6,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_6,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            # loss_locals.append(lp0_loss[0])
            # loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()
        
        elif t==7:
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_7,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_7,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_7,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_7,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_7,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_7,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            # loss_locals.append(lp0_loss[0])
            # loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()
            
        elif t==8:
            
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_8,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_8,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_8,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_8,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_8,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_8,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            # loss_locals.append(lp0_loss[0])
            # loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k]) / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            print("global_model: ",global_model.get('fc3.bias'))
            
            
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()
        
        elif t==9:
            
            print(f'Next Iteration round: {t}')
            print('Waiting for the Client/Slave Node to complete the Process...')
            input('enter something to exit:')
            
            print('Initial Global Model...')
            print('Queue Preparation for Global Model')
            
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            master_global_for_node0_round = f'master_global_for_node[{0}]_round[{t}]'
            
            task_queue = master_global_for_node0_round
            
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
            master_global_for_node1_round = f'master_global_for_node[{1}]_round[{t}]'
            
            task_queue = master_global_for_node1_round
            channel.queue_declare(queue=task_queue, durable=True)
            msg = pickle.dumps(global_model)
            channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
            print(" [x] Sent Round=",t)
            print(f'Round Process Started... Current Round on Master t={t}')
            input('Insert Any Key.. If global Model is delivered...: ')
                
                
            print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
            ##################### testing on global model #######################
    

            # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
            url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
            params = pika.URLParameters(url)
            connection = pika.BlockingConnection(params)
            channel = connection.channel() # start a channel
                
            # ################### callback global #################
        
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #         rq0.callback_global_1,
            #             auto_ack=True)
            
            # ##################### callback local #################
                
            # channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
            #             rq0.callback_local_1,
            #             auto_ack=True)
                    
            # ################### callback Local Loss #################
                
            # channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss_1,
            #                 auto_ack=True)
            # #######################################################
                
            # channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
            #                 rq0.callback_local_loss,
            #                 auto_ack=True)
                
           
            ################### callback global #################
                
            channel.basic_consume(f'node[{0}]_global_round[{t}]',
                            rq0.callback_global_9,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_global_round[{t}]',
                            rq1.callback_global_9,
                            auto_ack=True)
                        
                        
            ##################### callback local #################
                        
            channel.basic_consume(f'node[{0}]_local_round[{t}]',
                            rq0.callback_local_9,
                            auto_ack=True)
        
            channel.basic_consume(f'node[{1}]_local_round[{t}]',
                            rq1.callback_local_9,
                            auto_ack=True)
                        
        
            
            ################### callback Local Loss #################
                        
            channel.basic_consume(f'node[{0}]_local_loss_round[{t}]',
                            rq0.callback_local_loss_9,
                            auto_ack=True)
            
            channel.basic_consume(f'node[{1}]_local_loss_round[{t}]',
                            rq1.callback_local_loss_9,
                            auto_ack=True)
                    
            #######################################################
                    
            
            
            #print("t=",t ," | global_model: ",global_model.get('fc3.bias'))
            
            try:
                channel.start_consuming()
            except KeyboardInterrupt:
                channel.stop_consuming()
                connection.close()
            
            
            
            lp0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_round[{t}].pkl')
            lp1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_round[{t}].pkl')
            local_updates.append(lp0)
            local_updates.append(lp1)
    
    
            lp0_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_local_loss_round[{t}].pkl')
            lp1_loss = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_local_loss_round[{t}].pkl')
    
            gm = []
            gm0 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[0]_global_round[{t}].pkl')
            gm1 = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node[1]_global_round[{t}].pkl')
    
            # gm.append(gm0)
            # gm.append(gm1)
            
            # print("gm: ",gm[0].get('fc3.bias'))
            # print("gm: ",gm[1].get('fc3.bias'))

            # for i in range(len(gm)):
            #     global_model = {
            #         k: global_model[k] + (gm[0][k] + gm[1][k])
            #         for k in global_model.keys()
            #     }
    
            loss_locals.append(lp0_loss[0])
            loss_locals.append(lp1_loss[0])
    
            # print("global_model: ",global_model.get('fc3.bias'))

            
            # for i in range(num_selected_users):
            #     pass
            # global_model = {
            #         k: global_model[k] + (local_updates[0][0][k]+local_updates[1][0][k])
            #         for k in global_model.keys()
            #     }
            
            for i in range(num_selected_users):
                global_model = {
                k: global_model[k] + local_updates[i][0][k] / num_selected_users
                for k in global_model.keys()
            }
            
            global_counter = 1
    
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], test_acc[-1]))
            
            print('Submitting new global model: .....')
    
    
            credentials = pika.PlainCredentials('jahanxb', 'phdunr')
            parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

            connection = pika.BlockingConnection(parameters)
            #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
            channel = connection.channel()
            
            nodes = [0,1]
            for n in nodes:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'master_global_for_node[{n}]_round[{t+1}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                    exchange='',
                    routing_key=task_queue,
                    body=msg,
                    properties=pika.BasicProperties(delivery_mode=2)
                )
                print(" [x] Node=", n," Sent Round=",global_counter)
            
            connection.close()
        
    t2 = time.time()
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


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
            "user_index": user_counter, "dataset": "cifar", "gpu": -1, "round": 3
        },
        1: {
            "user_index": args.num_users, "dataset": "cifar", "gpu": -1, "round": 3
        }
    }
    args.num_users = user_counter
    serve(args)
