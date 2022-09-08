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

import fedml_pb2, fedml_pb2_grpc


def client_node():
    pid = os.getpid()
    with grpc.insecure_channel("localhost:9999") as channel:
        stub = fedml_pb2_grpc.FedMlManagementServiceStub(channel)

        try:
            while True:
                print("Active PID : %i" % pid)
                # time.sleep(10)
                start = time.time()
                args = call_parser()
                print("num user is: ", args.num_users)
                response = stub.UserIndex(fedml_pb2.FedmlRequest(
                    user_index=args.num_users,
                    dataset=args.dataset,
                    gpu=args.gpu,
                    round=args.round
                ))
                # response = stub.ping(pingpong_pb2.Ping(count=counter))
                # counter = response.count
                # if counter % 100 == 0:
                #     print(
                #         "%.4f : resp=%s : procid=%i"
                #         % (time.time() - start, response.count, pid)
                #     )
                #     # counter = 0
                time.sleep(0.001)
                # #########################################
                # # Fedml Code
                # #######################################
                #
                # torch.manual_seed(args.seed + args.repeat)
                # torch.cuda.manual_seed(args.seed + args.repeat)
                # np.random.seed(args.seed + args.repeat)
                #
                # args, dataset_train, dataset_test, dict_users = data_setup(args)
                # print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
                # print(
                #     'length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
                # print('num. of training data:{}'.format(len(dataset_train)))
                # print('num. of testing data:{}'.format(len(dataset_test)))
                # print('num. of classes:{}'.format(args.num_classes))
                # print('num. of users:{}'.format(len(dict_users)))
                # # sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
                #
                # sample_per_users = 0
                # for i in range(len(dict_users)):
                #     sample_per_users += int(sum([len(dict_users[i]) / len(dict_users)]))
                #
                # sample_per_users = 5  # for two users , we take 25000 samples as per the loop
                #
                # print('num. of samples per user:{}'.format(sample_per_users))
                # if args.dataset == 'fmnist' or args.dataset == 'cifar':
                #     dataset_test, val_set = torch.utils.data.random_split(
                #         dataset_test, [9000, 1000])
                #     print(len(dataset_test), len(val_set))
                # elif args.dataset == 'svhn':
                #     dataset_test, val_set = torch.utils.data.random_split(
                #         dataset_test, [len(dataset_test) - 2000, 2000])
                #     print(len(dataset_test), len(val_set))
                #
                # print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
                # log_path = set_log_path(args)
                # print(log_path)
                #
                # args, net_glob = model_setup(args)
                # print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])
                #
                # ###################################### model initialization ###########################
                # print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
                # t1 = time.time()
                # net_glob.train()
                # # copy weights
                # global_model = copy.deepcopy(net_glob.state_dict())
                # local_m = []
                # train_local_loss = []
                # test_acc = []
                # norm_med = []
                # ####################################### run experiment ##########################
                #
                # # initialize data loader
                # data_loader_list = []
                # print(len(dict_users))
                # for i in range(args.num_users):
                #     dataset = DatasetSplit(dataset_train, dict_users[i])
                #     ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
                #     data_loader_list.append(ldr_train)
                # ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
                #
                # m = max(int(args.frac * args.num_users), 1)
                # for t in range(args.round):
                #     args.local_lr = args.local_lr * args.decay_weight
                #     selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
                #     num_selected_users = len(selected_idxs)
                #
                #     ###################### local training : SGD for selected users ######################
                #     loss_locals = []
                #     local_updates = []
                #     delta_norms = []
                #     for i in selected_idxs:
                #         l_solver = LocalUpdate(args=args)
                #         net_glob.load_state_dict(global_model)
                #         # choose local solver
                #         if args.local_solver == 'local_sgd':
                #             new_model, loss = l_solver.local_sgd(
                #                 net=copy.deepcopy(net_glob).to(args.device),
                #                 ldr_train=data_loader_list[i])
                #         # compute local delta
                #         model_update = {k: new_model[k] - global_model[k] for k in global_model.keys()}
                #
                #         # compute local model norm
                #         delta_norm = torch.norm(
                #             torch.cat([
                #                 torch.flatten(model_update[k])
                #                 for k in model_update.keys()
                #             ]))
                #         delta_norms.append(delta_norm)
                #
                #         # clipping local model or not ? : no clip for cifar10
                #         # threshold = delta_norm / args.clip
                #         # if threshold > 1.0:
                #         #     for k in model_update.keys():
                #         #         model_update[k] = model_update[k] / threshold
                #
                #         local_updates.append(model_update)
                #         loss_locals.append(loss)
                #     norm_med.append(torch.median(torch.stack(delta_norms)).cpu())
                #
                #     ##################### communication: avg for all groups #######################
                #     model_update = {
                #         k: local_updates[0][k] * 0.0
                #         for k in local_updates[0].keys()
                #     }
                #     for i in range(num_selected_users):
                #         global_model = {
                #             k: global_model[k] + local_updates[i][k] / num_selected_users
                #             for k in global_model.keys()
                #         }
                #
                #     ##################### testing on global model #######################
                #     net_glob.load_state_dict(global_model)
                #     net_glob.eval()
                #     test_acc_, _ = test_img(net_glob, dataset_test, args)
                #     test_acc.append(test_acc_)
                #     train_local_loss.append(sum(loss_locals) / len(loss_locals))
                #     # print('t {:3d}: '.format(t, ))
                #     print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                #           format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))
                #
                #     if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
                #         np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
                #                    test_acc,
                #                    delimiter=",")
                #         np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
                #                    train_local_loss,
                #                    delimiter=",")
                #         np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
                #         break;
                #
                # t2 = time.time()
                # hours, rem = divmod(t2 - t1, 3600)
                # minutes, seconds = divmod(rem, 60)
                # print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
                #
                # ##############################
                # ## End of Fedml
                # ###############################

        except KeyboardInterrupt:
            channel.unsubscribe(close)
            exit()


def close(channel):
    channel.close()


if __name__ == '__main__':
    ################################### hyperparameter setup ########################################

    client_node()
