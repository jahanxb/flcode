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

NODE_ID = 1
global_node_addr = '10.10.1.1'

cassandra_addr = '10.10.1.2' 
#mongodb_url = 'mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority'

#mongodb_url = 'mongodb://jahanxb:phdunr@130.127.133.239:27017/?authMechanism=DEFAULT&authSource=flmongo&tls=false'
#mongodb_url = 'mongodb+srv://jahanxb:phdunr@flmongo.7repipw.mongodb.net/?retryWrites=true&w=majority'
#mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/iteration_status?authMechanism=DEFAULT&authSource=admin&tls=false'
#mongodb_url = 'mongodb://jahanxb1:phdunr@10.10.1.1:27017/?authMechanism=DEFAULT&authSource=admin&tls=false'

mongodb_url = 'mongodb://jahanxb:phdunr@10.10.1.1:27017/?authMechanism=DEFAULT&authSource=admin&tls=false'

def client_num_users(num_users):
    if num_users == 1:
        return 0,1
    elif num_users == 2:
        return 1,2
    elif num_users == 3:
        return 2,3
    elif num_users == 4:
        return 3,4
    elif num_users == 5:
        return 4,5
    elif num_users == 6:
        return 5,6
    elif num_users == 7:
        return 6,7
    elif num_users == 8:
        return 7,8
    elif num_users == 9:
        return 8,9
    elif num_users == 10:
        return 9,10
    
    
def training_and_testing_size(num_users):
    if num_users == 1:
        return int(9000/100),int(1000/100),int(25000/100)
    elif num_users == 2:
        return int(9000/90),int(1000/90),int(25000/90)
    elif num_users == 3:
        return int(9000/80),int(1000/80),int(25000/80)
    elif num_users == 4:
        return int(9000/70),int(1000/70),int(25000/70)
    elif num_users == 5:
        return int(9000/60),int(1000/60),int(25000/60)
    elif num_users == 6:
        return int(9000/50),int(1000/50),int(25000/50)
    elif num_users == 7:
        return int(9000/40),int(1000/40),int(25000/40)
    elif num_users == 8:
        return int(9000/30),int(1000/30),int(25000/30)
    elif num_users == 9:
        return int(9000/20),int(1000/20),int(25000/20)
    elif num_users == 10:
        return int(9000/10),int(1000/10),int(25000/10)
    