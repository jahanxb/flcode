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

def arrange_round_train(args):
    pass



def global_model_aggregate():
    pass

def ack_agent():
    pass

def something_something():
    pass


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
    
    
    
    credentials = pika.PlainCredentials('jahanxb', 'phdunr')
    parameters = pika.ConnectionParameters('130.127.134.6',
                                   5672,
                                   '/',
                                   credentials)
    # ye wala code waise master node p hone chaiyee ab , q k ye ab receive kare ga round 
    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
            
    channel.queue_declare(queue='task_queue',durable=True)
    print(' [*] Waiting for messages. To exit press')
            
    

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='task_queue', on_message_callback=callback)

    channel.start_consuming()
        
    
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
    
    nodes = 2
    local_updates = []
    loss_locals = []
    
    
    # consumer = KafkaConsumer('my_favorite_topic', group_id='my_favorite_group')
    # for msg in consumer:
    #     print("kafka consumer msg: ",msg)
    
    
    
    
    for n in range(nodes):
        for t in range(args.round):
            print(f'appending node{n}{t}')
            #localupdates = torch.load(f'/mydata/flcode/models/pickles/node{n}[{t}][0].pkl')
            #lossy = torch.load(f'/mydata/flcode/models/pickles/node{n}-loss[{t}][0].pkl')
            lossy = [[1,2,3]]
            localupdates = [1,2,3]
            local_updates.append(localupdates)
            loss_locals.append(lossy[0])
            print('')
            
    # #print("len: ",len(local_updates))
    # print("local update: ",local_updates[10][0].get('fc3.bias'))
    # num_selected_users = 2

    
    
    # for t in range(args.round):
    #     for i in range(num_selected_users):
    #         global_model = {
    #                 k: global_model[k] + local_updates[t][0].get(k) / num_selected_users
    #                 #k: localupdates[0].get(k) - global_model[k] for k in global_model.keys()
    #                 #k: global_model[k] + local_updates[i][k] / num_selected_users
    #                 for k in global_model.keys()
    #             }


    # # for i in range(num_selected_users):
    # #     global_model = {
    # #                 k: global_model[k] + local_updates[i][0][k] / num_selected_users
    # #                 for k in global_model.keys()
    # #             }
    # #print("global_modeL: ",global_model)



    print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
    ##################### testing on global model #######################
    net_glob.load_state_dict(global_model)
    net_glob.eval()
    # test_acc_, _ = test_img(net_glob, dataset_test, args)
    # test_acc.append(test_acc_)
    # train_local_loss.append(sum(loss_locals) / len(loss_locals))
    # print('t {:3d}: '.format(t, ))
    # print('t {:3d}: train_loss = {:.3f}, norm = Not Recording, test_acc = {:.3f}'.
    #               format(t, train_local_loss[0], test_acc[0]))

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
