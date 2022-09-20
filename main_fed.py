#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from asyncore import read
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
    
    fsl = lib.FileServer()
    fsl.start()
    
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
        # sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))

        sample_per_users = 0
        for i in range(len(dict_users)):
            sample_per_users += int(sum([len(dict_users[i]) / len(dict_users)]))

        sample_per_users = 5 #12500  # for two users , we take 25000 samples as per the loop

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

        net_glob.train()

        # copy weights
        global_model = copy.deepcopy(net_glob.state_dict())
        local_m = []
        train_local_loss = []
        test_acc = []
        norm_med = []
        ####################################### run experiment ##########################

        # initialize data loader
        data_loader_list = []
        print("len(dict_user): ", len(dict_users))
        index = args.num_users
        for i in range(1):
            dataset = DatasetSplit(dataset_train, dict_users[i])
            ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
            data_loader_list.append(ldr_train)
        ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

        m = max(int(args.frac * 1), 1)
        print("m = ",m)
        for t in range(args.round):
            args.local_lr = args.local_lr * args.decay_weight
            selected_idxs = list(np.random.choice(range(1), m, replace=False))
            print("In Round Loop: selected_idxs: ",selected_idxs)
            num_selected_users = len(selected_idxs)

            ###################### local training : SGD for selected users ######################
            loss_locals = []
            local_updates = []
            delta_norms = []
            for i in selected_idxs:
                print(i)
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

                # clipping local model or not ? : no clip for cifar10
                # threshold = delta_norm / args.clip
                # if threshold > 1.0:
                #     for k in model_update.keys():
                #         model_update[k] = model_update[k] / threshold

                local_updates.append(model_update)
                print("local updates len",len(local_updates), "index",len(local_updates[0]))
                loss_locals.append(loss)
            norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

            '''
            ####################################
            Download model from node1
            ################################
            '''
            # import file_grpc_lib as lib
            # fsl = lib.FileServer()
            # fsl.start()

            # out_file_name = "/mydata/flcode/models/node1.pth"
            # #os.remove(out_file_name)
            # if os.path.exists(out_file_name):
            #     print("File Already Exists... Will use the existing one, please delete the old one or rename")
                
            # else:
            #     fsl.download('whatever_name', out_file_name)
            # print('Download from Node1 for .pth completed')
            # fsl.stop_me()

            '''
            #####################################
            '''

            ##################### communication: avg for all groups #######################
            model_update = {
                k: local_updates[0][k] * 0.0
                for k in local_updates[0].keys()
            }

            # torch.save(local_updates, "/mydata/flcode/models/pickles/node0.pkl")
            # torch.save(loss_locals, "/mydata/flcode/models/pickles/node0-loss.pkl")
            # print("local_updates len(): ",len(local_updates))
            # node1 = "/mydata/flcode/models/pickles/node1.pkl"
            # sm1 = torch.load(node1)

            # node0 = "/mydata/flcode/models/pickles/node0.pkl"
            # sm0 = torch.load(node0)

            # ##node0_loss = loss_locals + node1[2]    
            # local_updates = sm0 + sm1

            # num_selected_users = len(local_updates)
            
            # node1_loss = torch.load("/mydata/flcode/models/pickles/node1-loss.pkl")

            # loss_locals = loss_locals + node1_loss
            
            # print("num_selected_users,",num_selected_users)
            for i in range(num_selected_users):
                global_model = {
                    k: global_model[k] + local_updates[i][k] / num_selected_users
                    for k in global_model.keys()
                }

            # print("model_update: ",model_update)
            # print(type(model_update))

            # torch.save(net_glob.state_dict(),"/home/jahanxb/PycharmProjects/FLcode/models/temp.pth")
            # torch.save(model_update, "/home/jahanxb/PycharmProjects/FLcode/models/node0.pth")
            #
            # import file_grpc_lib as lib
            # client = lib.FileClient('localhost:8888')
            # out_file_name = '/home/jahanxb/PycharmProjects/FLcode/models/received.pth'
            # if os.path.exists(out_file_name):
            #     os.remove(out_file_name)
            # client.download('whatever_name', out_file_name)

            '''
            Here we should Catch the aggregator for the client model 
            
            '''
            print('################## TrainingTest on node0 ######################')
            ##################### testing on global model #######################
            net_glob.load_state_dict(global_model)
            net_glob.eval()
            test_acc_, _ = test_img(net_glob, dataset_test, args)
            test_acc.append(test_acc_)
            train_local_loss.append(sum(loss_locals) / len(loss_locals))
            # print('t {:3d}: '.format(t, ))
            print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                  format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))

            if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
                np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
                           test_acc,
                           delimiter=",")
                np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
                           train_local_loss,
                           delimiter=",")
                np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
                break;
            #################################

            t2 = time.time()
            hours, rem = divmod(t2 - t1, 3600)
            minutes, seconds = divmod(rem, 60)
            print("Local training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            #################################

        t2 = time.time()
        hours, rem = divmod(t2 - t1, 3600)
        minutes, seconds = divmod(rem, 60)
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        # ##############################
        # ## End of Fedml
        # ###############################

        torch.save(local_updates, "/mydata/flcode/models/pickles/node0.pkl")
        torch.save(loss_locals, "/mydata/flcode/models/pickles/node0-loss.pkl")
        print("local_updates len(): ",len(local_updates))
        node1 = "/mydata/flcode/models/pickles/node1.pkl"
        sm1 = torch.load(node1)

        node0 = "/mydata/flcode/models/pickles/node0.pkl"
        sm0 = torch.load(node0)

        ##node0_loss = loss_locals + node1[2]    
        local_updates = sm0 + sm1

        num_selected_users = len(local_updates)
            
        node1_loss = torch.load("/mydata/flcode/models/pickles/node1-loss.pkl")

        loss_locals = loss_locals + node1_loss
        print("local updates:",local_updates)
        print("num_selected_users,",num_selected_users)
        for i in range(num_selected_users):
                global_model = {
                    k: global_model[k] + local_updates[i][k] / num_selected_users
                    for k in global_model.keys()
                }

        print('################## TrainingTest onum_selected_usersn aggregated Model ######################')
        ##################### testing on global model #######################
        net_glob.load_state_dict(global_model)
        net_glob.eval()
        test_acc_, _ = test_img(net_glob, dataset_test, args)
        test_acc.append(test_acc_)
        train_local_loss.append(sum(loss_locals) / len(loss_locals))
        # print('t {:3d}: '.format(t, ))
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                  format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))

        if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
            np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
                           test_acc,
                           delimiter=",")
            np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
                           train_local_loss,
                           delimiter=",")
            np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
            #break;

        t2 = time.time()
        hours, rem = divmod(t2 - t1, 3600)
        minutes, seconds = divmod(rem, 60)
        print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

        # torch.save(local_updates, "/mydata/flcode/models/node0.pkl")
        # sm = torch.jit.script(model_update)
        # sm.save('/mydata/flcode/models/node0.pth')
        

        #out_file_name = "/mydata/flcode/models/node1.pkl"




        # if os.path.exists(out_file_name):
        #     print("File Already Exists... Will use the existing one, please delete the old one or rename")
        #     #os.remove(out_file_name)
        # else:
        #     print('Download from Node1 for .pth started...')
        #     fsl.download('whatever_name', out_file_name)
        #     #fsl.stop_me()


        # torch.load_state_dict(torch.load(out_file_name))
        # torch.eval()
        

        
        
            



        # client = lib.FileClient('localhost:8888')
        # out_file_name = '/home/jahanxb/PycharmProjects/FLcode/models/received.pkl'
        # if os.path.exists(out_file_name):
        #     os.remove(out_file_name)
        # client.download('whatever_name', out_file_name)


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

    user_counter = int(args.num_users / 2)
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
