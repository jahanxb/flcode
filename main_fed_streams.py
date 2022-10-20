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


def pdf_process_function(msg):
        print("processing")
        print('pickle loading started...')
        gmdl = pickle.loads(msg)
        #global_model = gmdl
        print('pickle loading completed...')

                
                
        time.sleep(5) # delays for 5 seconds
        print("processing finished");
        return gmdl




def callback_local_loss(ch, method, properties, body):
        bodytag = 0
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}-loss[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals


def callback_global(ch, method, properties, body):
        bodytag = 0
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}-global[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        #return global_model


def callback_local(ch, method, properties, body):
        bodytag = 0
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        #return local_updates





def callback_local_loss_1(ch, method, properties, body):
        bodytag = 1
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}-loss[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals


def callback_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}-global[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        #return global_model


def callback_local_1(ch, method, properties, body):
        bodytag = 1
        gdm = pdf_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node{bodytag}[{bodytag}][{0}].pkl")
        # set up subscription on the queue
        #return local_updates






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
    
    
    
    for t in range(args.round):
        if t==0:
                print('Initial Global Model...')
                print('Queue Preparation for Global Model')
                task_queue = f'global_model_round_queue_[{0}][{t}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                task_queue = f'global_model_round_queue_[{1}][{t}]'
                channel.queue_declare(queue=task_queue, durable=True)
                msg = pickle.dumps(global_model)
                channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
                
                
                
                print(" [x] Sent Round=",t)

        else:
                
                print(f'Next Iteration Node_{0} round: {t}')
                print('Waiting for the Client/Slave Node to complete the Process...')
                
                
            
    input('Press any key:')
    
    # for n in range(nodes):
    #     for t in range(args.round):
    #         print(f'appending node{n}{t}')
    #         #localupdates = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
    #         #lossy = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{t}][0].pkl')
    #         lossy = [[1,2,3]]
    #         localupdates = [1,2,3]
    #         local_updates.append(localupdates)
    #         loss_locals.append(lossy[0])
    #         print('')
            
    #print("len: ",len(local_updates))
    #print("local update: ",local_updates[10][0].get('fc3.bias'))
    num_selected_users = 2

    
    
    for t in range(args.round):
        for i in range(num_selected_users):
            pass
            # global_model = {
            #         k: global_model[k] + local_updates[t][0].get(k) / num_selected_users
            #         #k: localupdates[0].get(k) - global_model[k] for k in global_model.keys()
            #         #k: global_model[k] + local_updates[i][k] / num_selected_users
            #         for k in global_model.keys()
            #     }
            
            # task_queue = f'master_round_queue_[{i}][{t}]'
            # channel.queue_declare(queue=task_queue, durable=True)
            # msg = pickle.dumps(local_updates)
                

            # message = msg
                
            # print('task_queue',task_queue)
                
            # channel.basic_publish(
            #     exchange='',
            #     routing_key=task_queue,
            #     body=message,
            #     properties=pika.BasicProperties(delivery_mode=2)
            #     )
                
                

            #print(" [x] Sent Round=",t)
        
        #connection.close()
                


    # # for i in range(num_selected_users):
    # #     global_model = {
    # #                 k: global_model[k] + local_updates[i][0][k] / num_selected_users
    # #                 for k in global_model.keys()
    # #             }
    # #print("global_modeL: ",global_model)



    print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
    ##################### testing on global model #######################
    

    # Access the CLODUAMQP_URL environment variable and parse it (fallback to localhost)
    url = 'amqp://jahanxb:phdunr@130.127.134.6:5672/'
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel() # start a channel
    
    
    gc = [0,1]
    
    bodytag = 0
    for glc in gc:
        
        if glc == 1:
            bodytag = 1
            ################### callback global #################
        
            channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
                callback_global_1,
                auto_ack=True)
            
            
            ##################### callback local #################
            
            channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
                callback_local_1,
                auto_ack=True)
            
            
            ################### callback Local Loss #################
            
            channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
                callback_local_loss_1,
                auto_ack=True)
            
            
            
            
            
            #######################################################
            
            channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
                callback_local_loss,
                auto_ack=True)
            
        else:
            bodytag = 0
            ################### callback global #################
            
            channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
                callback_global,
                auto_ack=True)
            
            
            ##################### callback local #################
            
            channel.basic_consume(f'node_local_round_queue_[{glc}][{0}]',
                callback_local,
                auto_ack=True)
            
            
            ################### callback Local Loss #################
            
            channel.basic_consume(f'node_local_loss_queue_[{glc}][{0}]',
                callback_local_loss,
                auto_ack=True)
            
            
            
            
            
            #######################################################
            
            channel.basic_consume(f'node_global_round_queue_[{glc}][{0}]',
                callback_local_loss,
                auto_ack=True)
            
    print("global_model: ",global_model)
    try:
                
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
        connection.close()
            
            
            #global_model = gmdl
            
        
      
      
          
    # credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    # parameters = pika.ConnectionParameters('130.127.134.6',
    #                                5672,
    #                                '/',
    #                                credentials)

    # connection = pika.BlockingConnection(parameters)
    # #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
    # channel = connection.channel()

    
    local_updates = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node{0}[{0}][0].pkl')
    loss_locals = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node{0}-loss[{0}][0].pkl')
    global_model = torch.load(f'/mydata/flcode/models/rabbitmq-queues/pickles/node{0}-global[{0}][0].pkl')
    
    global_counter = 1
    
    net_glob.load_state_dict(global_model)
    net_glob.eval()
    test_acc_, _ = test_img(net_glob, dataset_test, args)
    test_acc.append(test_acc_)
    
    
    # for n in range(nodes):
    #     localupdates = torch.load(f'/mydata/flcode/models/pickles/node{n}[{0}][0].pkl')
    #     lossy = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{0}][0].pkl')
    #     local_updates.append(localupdates)
    #     loss_locals.append(lossy[0])
    #     print("loss_locals: ",loss_locals)
    
    
    print('loss_locals:[outside Func] ',loss_locals)
    
    
    train_local_loss.append(sum(loss_locals) / len(loss_locals))
    # print('t {:3d}: '.format(t, ))
    print('t {:3d}: train_loss = {:.3f}, norm = Not Recording, test_acc = {:.3f}'.
                  format(t, train_local_loss[0], test_acc[0]))


    
    print('Submitting new global model: .....')
    
    
    credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)

    connection = pika.BlockingConnection(parameters)
    #connection = pika.BlockingConnection(pika.ConnectionParameters(host='amqp://jahanxb:phdunr@130.127.134.6:15672'))
    channel = connection.channel()
    
    
    print('Initial Global Model...')
    print('Queue Preparation for Global Model')
    task_queue = f'global_model_round_queue_[{0}][{global_counter}]'
    channel.queue_declare(queue=task_queue, durable=True)
    msg = pickle.dumps(global_model)
    channel.basic_publish(
                exchange='',
                routing_key=task_queue,
                body=msg,
                properties=pika.BasicProperties(delivery_mode=2)
                )
    print(" [x] Sent Round=",global_counter)
    
    

    # if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
    #     np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
    #                        test_acc,
    #                        delimiter=",")
    #     np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
    #                        train_local_loss,
    #                        delimiter=",")
    #     np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
    #         #break;
    # print(f't {t}: train_loss = {train_local_loss}, norm = {norm_med}, test_acc = {test_acc}')
    

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
