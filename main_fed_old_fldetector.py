  
    
    

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import copy
import numpy as np
import time, math
import torch

from utils.data_utils import data_setup, DatasetSplit
from utils.model_utils import *
from utils.aggregation import *
from option_old import call_parser
from models.Update import LocalUpdate
from models.test import test_img
from torch.utils.data import DataLoader


# from utils.rdp_accountant import compute_rdp, get_privacy_spent
import warnings
warnings.filterwarnings("ignore")
torch.cuda.is_available()




import mxnet as mx
import numpy as np
from copy import deepcopy
import time
from numpy import random
from mxnet import nd, autograd, gluon


import nd_aggregation
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import random
import argparse
import byzantine
import sys
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy
from collections import namedtuple

import byzantine






class VAE(gluon.HybridBlock):
    def __init__(self, ctx):
        super(VAE, self).__init__()
        self.ctx = ctx
        with self.name_scope():
            self.fc1 = gluon.nn.Dense(500)
            self.fc21 = gluon.nn.Dense(20)
            self.fc22 = gluon.nn.Dense(20)
            self.fc3 = gluon.nn.Dense(500)
            self.fc4 = gluon.nn.Dense(640)

    def encode(self, x):
        h1 = nd.Activation(self.fc1(x), 'relu')
        return self.fc21(h1), nd.Activation(self.fc22(h1), 'softrelu')

    def reparametrize(self, mu, logvar):
        '''
        mu is a number and logvar is a ndarray
        '''
        std = nd.exp(0.5 * logvar)
        eps = nd.random_normal(
            loc=0, scale=1, shape=std.shape).as_in_context(self.ctx)
        return mu + eps * std

    def decode(self, z):
        h3 = nd.Activation(self.fc3(z), 'relu')
        return nd.Activation(self.fc4(h3), 'sigmoid')

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def lbfgs(args, S_k_list, Y_k_list, v):
    curr_S_k = nd.concat(*S_k_list, dim=1)
    curr_Y_k = nd.concat(*Y_k_list, dim=1)
    S_k_time_Y_k = nd.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = nd.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k.asnumpy())
    # L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.gpu(args.gpu))
    L_k = S_k_time_Y_k - nd.array(R_k, ctx=mx.cpu())
    sigma_k = nd.dot(Y_k_list[-1].T, S_k_list[-1]) / (nd.dot(S_k_list[-1].T, S_k_list[-1]))
    D_k_diag = nd.diag(S_k_time_Y_k)
    upper_mat = nd.concat(*[sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = nd.concat(*[L_k.T, -nd.diag(D_k_diag)], dim=1)
    mat = nd.concat(*[upper_mat, lower_mat], dim=0)
    mat_inv = nd.linalg.inverse(mat)

    approx_prod = sigma_k * v
    p_mat = nd.concat(*[nd.dot(curr_S_k.T, sigma_k * v), nd.dot(curr_Y_k.T, v)], dim=0)
    approx_prod -= nd.dot(nd.dot(nd.concat(*[sigma_k * curr_S_k, curr_Y_k], dim=1), mat_inv), p_mat)

    return approx_prod


def params_convert(net):
    tmp = []
    for param in net.collect_params().values():
        if param.grad_req == 'null':
            continue
        tmp.append(param.data().copy())
    params = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)
    return params


def clip(a, b, c):
    tmp = nd.minimum(nd.maximum(a, b), c)
    return tmp


def detection(score, nobyz):
    estimator = KMeans(n_clusters=2)
    estimator.fit(score.reshape(-1, 1))
    label_pred = estimator.labels_
    if np.mean(score[label_pred==0])<np.mean(score[label_pred==1]):
        #0 is the label of malicious clients
        label_pred = 1 - label_pred
    real_label=np.ones(100)
    real_label[:nobyz]=0
    acc=len(label_pred[label_pred==real_label])/100
    recall=1-np.sum(label_pred[:nobyz])/nobyz
    fpr=1-np.sum(label_pred[nobyz:])/(100-nobyz)
    fnr=np.sum(label_pred[:nobyz])/nobyz
    print("acc %0.4f; recall %0.4f; fpr %0.4f; fnr %0.4f;" % (acc, recall, fpr, fnr))
    print(silhouette_score(score.reshape(-1, 1), label_pred))

def detection1(score, nobyz):
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min)/(max-min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum([np.square(score[m]-center[label_pred[m]]) for m in range(len(score))])
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum([np.square(rand[m]-center[label_pred[m]]) for m in range(len(rand))])
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    #print(gapDiff)
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i+1
            break
    if select_k == 1:
        print('No attack detected!')
        return 0
    else:
        print('Attack Detected!')
        return 1








if __name__ == '__main__':
    ################################### hyperparameter setup ########################################
    args = call_parser()
    
    ###### FLDetector (0)#######
    args.nbyz = 28
    
    
    ###################
    
    torch.manual_seed(args.seed+args.repeat)
    torch.cuda.manual_seed(args.seed+args.repeat)
    np.random.seed(args.seed+args.repeat)
    
    args, dataset_train, dataset_test, dict_users = data_setup(args)
    print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
    print(
        'length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of classes:{}'.format(args.num_classes))
    print('num. of users:{}'.format(len(dict_users)))
    
    sample_per_users = int(sum([ len(dict_users[i]) for i in range(len(dict_users))])/len(dict_users))
    
    sample_per_users = 25000
    
    print('num. of samples per user:{}'.format(sample_per_users))
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
    #############################FLDetector (1) #############
    
    ctx = mx.cpu()
    with ctx:
        batch_size = args.batch_size

        if args.dataset == 'cifar':
            num_inputs = 32 * 32 * 3
            num_outputs = 10
        else:
            raise NotImplementedError

        
        # CNN
        cnn = gluon.nn.Sequential()
        with cnn.name_scope():
            cnn.add(gluon.nn.Conv2D(channels=30, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            cnn.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
            cnn.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
            # The Flatten layer collapses all axis, except the first one, into one axis.
            cnn.add(gluon.nn.Flatten())
            cnn.add(gluon.nn.Dense(512, activation="relu"))
            cnn.add(gluon.nn.Dense(num_outputs))


        def evaluate_accuracy(data_iterator, net, trigger=False, target=None):
            acc = mx.metric.Accuracy()
            
            for i, (data, label) in enumerate(data_iterator):
                data = nd.array(data)
                data = data.as_in_context(ctx)
                label = nd.array(label)
                label = label.as_in_context(ctx)
                remaining_idx = list(range(data.shape[0]))
                if trigger:
                    for example_id in range(data.shape[0]):
                        data[example_id][0][1][28] = 1
                        data[example_id][1][1][28] = 1
                        data[example_id][2][1][28] = 1
                        data[example_id][0][1][29] = 1
                        data[example_id][1][1][29] = 1
                        data[example_id][2][1][29] = 1
                        data[example_id][0][1][30] = 1
                        data[example_id][1][1][30] = 1
                        data[example_id][2][1][30] = 1
                        data[example_id][0][2][29] = 1
                        data[example_id][1][2][29] = 1
                        data[example_id][2][2][29] = 1

                        data[example_id][0][3][28] = 1
                        data[example_id][1][3][28] = 1
                        data[example_id][2][3][28] = 1
                        data[example_id][0][4][29] = 1
                        data[example_id][1][4][29] = 1
                        data[example_id][2][4][29] = 1
                        data[example_id][0][5][28] = 1
                        data[example_id][1][5][28] = 1
                        data[example_id][2][5][28] = 1
                        data[example_id][0][5][29] = 1
                        data[example_id][1][5][29] = 1
                        data[example_id][2][5][29] = 1
                        data[example_id][0][5][30] = 1
                        data[example_id][1][5][30] = 1
                        data[example_id][2][5][30] = 1
                    for example_id in range(data.shape[0]):
                        if label[example_id] != target:
                            label[example_id] = target
                        else:
                            remaining_idx.remove(example_id)
                output = net(data)
                predictions = nd.argmax(output, axis=1)

                predictions = predictions[remaining_idx]
                label = label[remaining_idx]

                acc.update(preds=predictions, labels=label)
            return acc.get()[1]

        
                # decide attack type
        if args.byz_type == 'partial_trim':
            # partial knowledge trim attack
            byz = byzantine.partial_trim
        elif args.byz_type == 'full_trim':
            # full knowledge trim attack
            byz = byzantine.full_trim
        elif args.byz_type == 'no':
            byz = byzantine.no_byz
        elif args.byz_type == 'gaussian':
            byz = byzantine.gaussian_attack
        elif args.byz_type == 'mean_attack':
            byz = byzantine.mean_attack
        elif args.byz_type == 'full_mean_attack':
            byz = byzantine.full_mean_attack
        elif args.byz_type == 'dir_partial_krum_lambda':
            byz = byzantine.dir_partial_krum_lambda
        elif args.byz_type == 'dir_full_krum_lambda':
            byz = byzantine.dir_full_krum_lambda
        elif args.byz_type == 'backdoor' or 'dba' or 'scaling_attack':
            byz = byzantine.scaling_attack
        elif args.byz_type == 'label_flip':
            byz = byzantine.no_byz
        else:
            raise NotImplementedError
    
    
    net = cnn
    net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), force_reinit=True, ctx=ctx)
    
    # define loss
    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    # set upt parameters
    num_workers = args.num_users
    lr = 0.0001
    epochs = args.round
    grad_list = []
    old_grad_list = []
    weight_record = []
    grad_record = []
    train_acc_list = []
    
    
    # set up seed
    seed = args.seed
    mx.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
    # biased assignment
    bias_weight = 0.1
    other_group_size = (1 - bias_weight) / 9.
    worker_per_group = num_workers / 10
    
    
    
    #assign non-IID training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]
        
    
    #################end of (1) FLDetector ###########
    
    ####################################### run experiment ##########################
    
    
    # initialize data loader
    data_loader_list = []
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)
    ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    
    ############ FLDetector (2) ##################
    
    def transform(data, label):
        return nd.transpose(data.astype(np.float32), (2, 0, 1)) / 255., label.astype(np.float32)
    
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=True, transform=transform), 50000,
                                                    shuffle=True, last_batch='rollover')
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.CIFAR10(train=False, transform=transform), 10000,
                                                    shuffle=False, last_batch='rollover')
    
    
    for _, (data, label) in enumerate(train_data):
        for (x, y) in zip(data, label):
            if args.dataset == 'cifar':
                x = x.as_in_context(ctx).reshape(1, 3, 32, 32)
                
            y = y.as_in_context(ctx)

            # assign a data point to a group
            upper_bound = (y.asnumpy()) * (1 - bias_weight) / 9. + bias_weight
            lower_bound = (y.asnumpy()) * (1 - bias_weight) / 9.
            rd = np.random.random_sample()

            if rd > upper_bound:
                worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y.asnumpy() + 1)
            elif rd < lower_bound:
                worker_group = int(np.floor(rd / other_group_size))
            else:
                worker_group = y.asnumpy()

            # assign a data point to a worker
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)

    # concatenate the data for each worker
    each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data]
    each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]

    # random shuffle the workers
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]
    
    
    malicious_score = np.zeros((1, args.num_users))

    
    
    
    ############ End of FLDetector (2)############
    
    
    
    
    
    m = max(int(args.frac * args.num_users), 1)
    for t in range(args.round):
        args.local_lr = args.local_lr * args.decay_weight
        selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
        num_selected_users = len(selected_idxs)

        ###################### local training : SGD for selected users ######################
        loss_locals = []
        local_updates = []
        delta_norms = []
        for i in selected_idxs:
            
            ######FLDetector (3)###############
            batch_x = each_worker_data[i][:]
            batch_y = each_worker_label[i][:]
            
            
            backdoor_target = 0
            with autograd.record():
                output = net(batch_x)
                loss = softmax_cross_entropy(output, batch_y)
            # backward
            loss.backward()
            grad_list.append([param.grad().copy() for param in net.collect_params().values() if param.grad_req != 'null'])

            param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in grad_list]

            tmp = []
            for param in net.collect_params().values():
                if param.grad_req != 'null':
                    tmp.append(param.data().copy())
            weight = nd.concat(*[x.reshape((-1, 1)) for x in tmp], dim=0)

            # use lbfgs to calculate hessian vector productgaussian
            # if t > 20:
            #     hvp = lbfgs(args, weight_record, grad_record, weight - last_weight)
            # else:
            #     hvp = None
            
            hvp = None

            # perform attack
            if t > 0:
                param_list = byz(param_list, args.nbyz)
            
            
            if args.aggregation == 'trim':
                grad, distance = nd_aggregation.trim(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'simple_mean':
                grad, distance = nd_aggregation.simple_mean(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'median':
                grad, distance = nd_aggregation.median(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            elif args.aggregation == 'krum':
                grad, distance = nd_aggregation.krum(old_grad_list, param_list, net, lr, args.nbyz, hvp)
            else:
                raise NotImplementedError
            # Update malicious distance score
            if distance is not None and t > 20:
                malicious_score = np.row_stack((malicious_score, distance))

            if malicious_score.shape[0] >= 11:
                if detection1(np.sum(malicious_score[-10:], axis=0), args.nbyz):
                    print('Stop at iteration:', t)
                    detection(np.sum(malicious_score[-10:], axis=0), args.nbyz)
                    break
            
            
            
            # update weight record and gradient record
            if t > 0:
                weight_record.append(weight - last_weight)
                grad_record.append(grad - last_grad)

            # free memory & reset the list
            if len(weight_record) > 10:
                del weight_record[0]
                del grad_record[0]

            last_weight = weight
            last_grad = grad
            old_grad_list = param_list
            del grad_list
            grad_list = []
            
            
            ######## End of FLDetector (3)#####
            
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
            loss_locals.append(loss)
        norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

        ##################### communication: avg for all groups #######################
        model_update = {
            k: local_updates[0][k] * 0.0
            for k in local_updates[0].keys()
        }
        for i in range(num_selected_users):
            global_model = {
                k: global_model[k] + local_updates[i][k] / num_selected_users
                for k in global_model.keys()
            }
        
        
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

    #     ########FLDetector #############
    #     if args.byz_type == 'scaling_attack' or 'no':
            
    #         test_accuracy = evaluate_accuracy(test_data, net)
    #         backdoor_acc = evaluate_accuracy(test_data, net, trigger=True, target=backdoor_target)
    #         train_acc_list.append((test_accuracy, backdoor_acc))
    #         print("args.byz_type == 'scaling_attack' or 'no'")
    #         print("Epoch %02d. Test_acc %0.4f. Backdoor_acc %0.4f." % (t, test_accuracy, backdoor_acc))

    #     else:
    #         print("args.byz_type != 'scaling_attack' or 'no'")
    #         test_accuracy = evaluate_accuracy(test_data, net)
    #         train_acc_list.append(test_accuracy)
    #         print("Epoch %02d. Test_acc %0.4f" % (t, test_accuracy))
        
        
    
    
    # test_accuracy = evaluate_accuracy(test_data, net)
    # print("Again ... ! ... Epoch %02d. Test_acc %0.4f" % (t, test_accuracy))
    # detection(np.sum(malicious_score[-10:], axis=0), args.nbyz)

    
    # ##########end of FLDetector#######
        
        
    t2 = time.time()
    hours, rem = divmod(t2-t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    
    
    
    
    
    