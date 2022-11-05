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


node0 = 0
node1 = 1

def pickle_process_function(msg):
    print("processing")
    print('pickle loading started...')
    gmdl = pickle.loads(msg)
    #global_model = gmdl
    print('pickle loading completed...')

                
                
    time.sleep(5) # delays for 5 seconds
    print("processing finished");
    return gmdl


class Node0RabbitQueues:
    @staticmethod
    def callback_local_loss(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    @staticmethod
    def callback_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model
        
        
    @staticmethod
    def callback_master_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
        
    
    @staticmethod
    def callback_local_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    

    @staticmethod
    def callback_local(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    @staticmethod
    def callback_local_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates
        
        
    
    
    @staticmethod
    def callback_local_loss_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates
        

    @staticmethod
    def callback_local_loss_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals


    @staticmethod
    def callback_master_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model



    @staticmethod
    def callback_local_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    @staticmethod
    def callback_master_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node0}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
        
    @staticmethod
    def callback_local_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node0}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates



class Node1RabbitQueues:
    
    @staticmethod
    def callback_local_loss(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    @staticmethod
    def callback_master_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local(ch, method, properties, body):
        bodytag = 0
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    @staticmethod
    def callback_master_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_1(ch, method, properties, body):
        bodytag = 1
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates
        
        
    
    
    @staticmethod
    def callback_local_loss_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_global_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_2(ch, method, properties, body):
        bodytag = 2
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_3(ch, method, properties, body):
        bodytag = 3
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates

    @staticmethod
    def callback_local_loss_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    @staticmethod
    def callback_global_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_4(ch, method, properties, body):
        bodytag = 4
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    @staticmethod
    def callback_master_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_5(ch, method, properties, body):
        bodytag = 5
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates

    @staticmethod
    def callback_local_loss_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_6(ch, method, properties, body):
        bodytag = 6
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    @staticmethod
    def callback_local_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_7(ch, method, properties, body):
        bodytag = 7
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_8(ch, method, properties, body):
        bodytag = 8
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates


    @staticmethod
    def callback_local_loss_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global loss_locals
        loss_locals = gdm
        print('loss_locals:[inside Func] ',loss_locals)
        
        torch.save(loss_locals, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_loss_round[{bodytag}].pkl")
        # set up subscription on the queue
        #input('Press Any Key to Move to next step')
        #print('[x] press ctrl+c to move to next step')
        #return loss_locals

    
    @staticmethod
    def callback_master_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/master_global_for_node[{node1}]_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model
    
    
    
    @staticmethod
    def callback_local_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_global_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return global_model

    
    
    
    @staticmethod
    def callback_global_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        global_model = gdm
        torch.save(global_model, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_global_round[{bodytag}].pkl")
        raise KeyboardInterrupt
        # set up subscription on the queue
        #return global_model

    @staticmethod
    def callback_local_9(ch, method, properties, body):
        bodytag = 9
        gdm = pickle_process_function(body)
        time.sleep(5)
        print('[x] press ctrl+c to move to next step')
        print('bodytag value now: ',bodytag)
        #global local_updates
        local_updates = gdm
        torch.save(local_updates, f"/mydata/flcode/models/rabbitmq-queues/pickles/node[{node1}]_local_round[{bodytag}].pkl")
        # set up subscription on the queue
        #return local_updates
