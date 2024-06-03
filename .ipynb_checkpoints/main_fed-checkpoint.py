#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from asyncore import read
import copy
from fileinput import filename
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
import asyncio

import filetrans_pb2 as file_pb2
import filetrans_pb2_grpc as file_pb2_grpc



filename = ''

import random,string
# CHUNK_SIZE = 1024 * 1024  # 1MB
CHUNK_SIZE = 2154387


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield pb2.Chunk(buffer=piece)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)

class FileServer(file_pb2_grpc.FileServerServicer):
    def __init__(self):

        class Servicer(file_pb2_grpc.FileServerServicer):
            def __init__(self):
                self.tmp_file_name = filename
                # letters = string.ascii_lowercase
                # ''.join(random.choice(letters) for i in range(10))

            def upload(self, request_iterator, context):
                save_chunks_to_file(request_iterator, self.tmp_file_name)
                return pb2.Reply(length=os.path.getsize(self.tmp_file_name))

            def download(self, request, context):
                if request.name:
                    return get_file_chunks(self.tmp_file_name)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        file_pb2_grpc.add_FileServerServicer_to_server(Servicer(), self.server)

    def start(self):
        # self.server.add_insecure_port(f'[::]:{port}')
        self.server.add_insecure_port("10.10.1.3:9991")
        try:
            while True:
                time.sleep(10)
                # time.sleep(60*60*24)
        except KeyboardInterrupt:
            self.server.stop(0)
    
    def stop_me(self):
        self.server.stop(0)



# class FileServer(file_pb2_grpc.FileServerServicer):
#                             def __init__(self):

#                                 class Servicer(file_pb2_grpc.FileServerServicer):
#                                     def __init__(self):
#                                         pass
#                                         #self.tmp_file_name = filename
                
#                                     def upload(self, request_iterator, context):
#                                         save_chunks_to_file(request_iterator, filename=filename)
#                                         return file_pb2.Reply(length=os.path.getsize(filename=filename))

#                                     def download(self, request, context):
#                                         if request.name:
#                                             return get_file_chunks(filename=filename)

#                             self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
#                             file_pb2_grpc.add_FileServerServicer_to_server(Servicer(), self.server)

#                             def start(self):
                            
#                                 self.server.add_insecure_port("10.10.1.3:9991")
#                                 self.server.start()

#                                 try:
#                                     while True:
#                                         time.sleep(10)
                
#                                 except KeyboardInterrupt:
#                                     self.server.stop(0)




# server_args = {
#     "user_index":5,"dataset":"cifar","gpu":-1,"round":50
# }


class ArgsExchange(pb2_grpc.NodeExchangeServicer):
    def get_args(self, request, context):
        return pb2.args_data(**server_args.get(request.nodeid, {}))


def serve(args):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    pb2_grpc.add_NodeExchangeServicer_to_server(ArgsExchange(), server)
    server.add_insecure_port("[::]:9999")
    server.start()
    
    # fsl = lib.FileServer()
    # fsl.start()
    
    try:
        
        
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
        
        # fsl = lib.FileServer()
        # fsl.start()
        #fsl = lib.FileServer()
        #fsl.start() # try to run in async 
        
        print('Starting File Server...')
        fsl = FileServer()
        fsl.start()
        
        nodes = 1
        for n in range(nodes):
            for t in range(args.round):
                
                filename = f"/mydata/flcode/models/pickles/node{0}[{t}][0].pkl"
                
                
                
                if os.path.exists(filename):
                    print('File Already Exists...Exiting')
                else:
                    try:
                        print("filename: ",filename)
                        print(f"Local Rounds Processing from node{n} {t+1}/{args.round}")
                        # server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
                        # fsl = file_pb2_grpc.add_FileServerServicer_to_server(file_pb2_grpc.FileServerServicer(),server)
                        # server.add_insecure_port("10.10.1.3:9991")
                        # server.start()
                        # lib.get_file_chunks(filename=filename)
                                            
                        
                    
                        if os.path.exists(filename):
                            print('File Downloaded...Exiting')
                        else:
                            print(f'[wait time: 2 mins] Waiting for module: node{n}[{t}][{0}].pkl')
                            time.sleep(120)
                        if os.path.exists(filename):
                            pass
                        else:
                            print('File Not Received...waiting for 2 minutes...Another try.')
                            time.sleep(120)
                        
                        
                    except Exception as e:
                        print('Exception catched Perhaps Model not trained yet...trying again in 120 secs')
                        print(str(e))
                        time.sleep(120)
                        try:
                            if os.path.exists(filename):
                                print('File Downloaded...Exiting')
                            else:
                                print('File Not Received...waiting for 2 minutes.. Press [ctrl+c] to exit..')
                                time.sleep(120)
                            
                        
                        except Exception:
                            pass
                            #print('Closing File Server ...')
                                
                
                        
        
        
        # sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))

        
        # ##############################
        # ## End of Fedml
        # ###############################

        
        local_updates = [1,3,4]
        num_selected_users = len(local_updates)
            
                    
        # print("num_selected_users,",num_selected_users)
        # for i in range(num_selected_users):
        #         global_model = {
        #             k: global_model[k] + local_updates[i][k] / num_selected_users
        #             for k in global_model.keys()
        #         }

        print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
        print("-------To Be Continued .............")
        ##################### testing on global model #######################
        # net_glob.load_state_dict(global_model)
        # net_glob.eval()
        # test_acc_, _ = test_img(net_glob, dataset_test, args)
        # test_acc.append(test_acc_)
        # train_local_loss.append(sum(loss_locals) / len(loss_locals))
        # # print('t {:3d}: '.format(t, ))
        # print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
        #           format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))

        # if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
        #     np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
        #                    test_acc,
        #                    delimiter=",")
        #     np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
        #                    train_local_loss,
        #                    delimiter=",")
        #     np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
        #     #break;

        # t2 = time.time()
        # hours, rem = divmod(t2 - t1, 3600)
        # minutes, seconds = divmod(rem, 60)
        # print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        


    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        server.stop(0)

    # server_args = {
    #     0: {
    #         "user_index": 5, "dataset": "cifar", "gpu": -1, "round": 50
    #     },
    #     1: {
    #         "user_index": 10, "dataset": "cifar", "gpu": -1, "round": 1
    #     }
    # }


if __name__ == '__main__':
    args = call_parser()

    #user_counter = int(args.num_users / 2)
    user_counter = 1
    print("user counter : ", user_counter)

    server_args = {
        0: {
            "user_index": user_counter, "dataset": "cifar", "gpu": -1, "round": 10
        },
        1: {
            "user_index": args.num_users, "dataset": "cifar", "gpu": -1, "round": 10
        }
    }
    args.num_users = user_counter
    serve(args)
