# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# import copy
# import numpy as np
# import time, math
# import torch

# from utils.data_utils_noniid import data_setup, DatasetSplit
# from utils.model_utils import *
# from utils.aggregation import *
# from options import call_parser
# from models.Update import LocalUpdate
# from models.test import test_img
# from torch.utils.data import DataLoader


# import warnings
# warnings.filterwarnings("ignore")
# torch.cuda.is_available()


# import os
# import numpy as np
# import torch,os, dill

# from torchvision import datasets, transforms
# from torch.utils.data import Dataset

# class DatasetSplit(Dataset):
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = list(idxs)

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image, label

# def fmnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from FashionMNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()

#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users


# def verify_noniid_distribution(dict_users, labels, num_users):
#     for i in range(num_users):
#         user_labels = labels[dict_users[i]]
#         unique_labels, counts = np.unique(user_labels, return_counts=True)
#         print(f"User {i}: Label distribution {dict(zip(unique_labels, counts))}")

# def calculate_label_distribution(dict_users, dataset):
#     """
#     Calculate and print the label distribution for each user.
#     :param dict_users: Dictionary of user data indices.
#     :param dataset: Dataset object containing the data.
#     """
#     labels = np.array(dataset.targets)
#     for user, indices in dict_users.items():
#         user_labels = labels[indices]
#         unique_labels, counts = np.unique(user_labels, return_counts=True)
#         label_distribution = dict(zip(unique_labels, counts))
#         print(f"User {user}: Label distribution {label_distribution}")


# if __name__ == '__main__':
#     ################################### hyperparameter setup ########################################
#     args = call_parser()
    
#     torch.manual_seed(args.seed + args.repeat)
#     torch.cuda.manual_seed(args.seed + args.repeat)
#     np.random.seed(args.seed + args.repeat)
    
#     args, dataset_train, dataset_test, dict_users = data_setup(args)
#     print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
#     print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
#     print('num. of training data:{}'.format(len(dataset_train)))
#     print('num. of testing data:{}'.format(len(dataset_test)))
#     print('num. of classes:{}'.format(args.num_classes))
#     print('num. of users:{}'.format(len(dict_users)))
    
#     sample_per_users = int(sum([len(dict_users[i]) for i in range(len(dict_users))]) / len(dict_users))
    
#     sample_per_users = 25000
    
#     print('num. of samples per user:{}'.format(sample_per_users))
#     if args.dataset == 'fmnist' or args.dataset == 'cifar':
#         dataset_test, val_set = torch.utils.data.random_split(dataset_test, [9000, 1000])
#         print(len(dataset_test), len(val_set))
#     elif args.dataset == 'svhn':
#         dataset_test, val_set = torch.utils.data.random_split(dataset_test, [len(dataset_test) - 2000, 2000])
#         print(len(dataset_test), len(val_set))

#     print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
#     log_path = set_log_path(args)
#     print(log_path)

#     args, net_glob = model_setup(args)
#     print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])
    
#     # Verify the non-IID distribution
#     verify_noniid_distribution(dict_users, np.array(dataset_train.targets), args.num_users)
    
#     # Verify the non-IID distribution
#     print("{:<50}".format("=" * 15 + " verifying non-IID distribution " + "=" * 50)[0:60])
#     calculate_label_distribution(dict_users, dataset_train)
    
#     ###################################### model initialization ###########################
#     print("{:<50}".format("=" * 15 + " training... " + "=" * 50)[0:60])
#     t1 = time.time()
#     net_glob.train()
#     # copy weights
#     global_model = copy.deepcopy(net_glob.state_dict())
#     local_m = []
#     train_local_loss = []
#     test_acc = []
#     norm_med = []
#     ####################################### run experiment ##########################
    
#     # initialize data loader
#     data_loader_list = []
#     for i in range(args.num_users):
#         dataset = DatasetSplit(dataset_train, dict_users[i])
#         ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
#         data_loader_list.append(ldr_train)
#     ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    
#     m = max(int(args.frac * args.num_users), 1)
#     for t in range(args.round):
#         args.local_lr = args.local_lr * args.decay_weight
#         selected_idxs = list(np.random.choice(range(args.num_users), m, replace=False))
#         num_selected_users = len(selected_idxs)

#         ###################### local training : SGD for selected users ######################
#         loss_locals = []
#         local_updates = []
#         delta_norms = []
#         for i in selected_idxs:
#             l_solver = LocalUpdate(args=args)
#             net_glob.load_state_dict(global_model)
#             # choose local solver
#             if args.local_solver == 'local_sgd':
#                 new_model, loss = l_solver.local_sgd(
#                     net=copy.deepcopy(net_glob).to(args.device),
#                     ldr_train=data_loader_list[i])
#             # compute local delta
#             model_update = {k: new_model[k] - global_model[k] for k in global_model.keys()}

#             # compute local model norm
#             delta_norm = torch.norm(
#                 torch.cat([
#                     torch.flatten(model_update[k])
#                     for k in model_update.keys()
#                 ]))
#             delta_norms.append(delta_norm)
            
#             local_updates.append(model_update)
#             loss_locals.append(loss)
#         norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

#         ##################### communication: avg for all groups #######################
#         model_update = {
#             k: local_updates[0][k] * 0.0
#             for k in local_updates[0].keys()
#         }
#         for i in range(num_selected_users):
#             global_model = {
#                 k: global_model[k] + local_updates[i][k] / num_selected_users
#                 for k in global_model.keys()
#             }
        
#         ##################### testing on global model #######################
#         net_glob.load_state_dict(global_model)
#         net_glob.eval()
#         test_acc_, _ = test_img(net_glob, dataset_test, args)
#         test_acc.append(test_acc_)
#         train_local_loss.append(sum(loss_locals) / len(loss_locals))
#         print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
#                 format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))

#         if math.isnan(train_local_loss[-1]) or train_local_loss[-1] > 1e8 or t == args.round - 1:
#             np.savetxt(log_path + "_test_acc_repeat_" + str(args.repeat) + ".csv",
#                         test_acc,
#                         delimiter=",")
#             np.savetxt(log_path + "_train_loss_repeat_" + str(args.repeat) + ".csv",
#                         train_local_loss,
#                         delimiter=",")
#             np.savetxt(log_path + "_norm__repeat_" + str(args.repeat) + ".csv", norm_med, delimiter=",")
#             break;

#     t2 = time.time()
#     hours, rem = divmod(t2 - t1, 3600)
#     minutes, seconds = divmod(rem, 60)
#     print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
import time, math
import torch

from utils.data_utils_noniid import DatasetSplit
from utils.model_utils import *
from utils.aggregation import *
from options import call_parser
from models.Update import LocalUpdate
from models.test import test_img
from torch.utils.data import DataLoader


import warnings
warnings.filterwarnings("ignore")
torch.cuda.is_available()


import os
import numpy as np
import torch,os, dill

from torchvision import datasets, transforms
from torch.utils.data import Dataset


def flip_labels(labels, flip_mapping):
    """
    Flip labels according to the provided mapping
    :param labels: Original labels
    :param flip_mapping: Dictionary mapping original labels to new labels
    :return: Flipped labels
    """
    flipped_labels = labels.copy()
    for original_label, new_label in flip_mapping.items():
        flipped_labels[labels == original_label] = new_label
    return flipped_labels


def fmnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from FashionMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_shards, num_imgs = 600, 100 #200, 300 #600, 100  # Increase the number of shards
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        num_user_shards = np.random.randint(1, 6)  # Randomly assign 1 to 5 shards to each user
        rand_set = set(np.random.choice(idx_shard, num_user_shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users







def verify_noniid_distribution(dict_users, labels, num_users):
    for i in range(num_users):
        user_labels = labels[dict_users[i]]
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        #print(f"User {i}: Label distribution {dict(zip(unique_labels, counts))}")
        print(f' User {i}: Label Distribution: {unique_labels} | counts: {counts}')

def calculate_label_distribution(dict_users, dataset):
    """
    Calculate and print the label distribution for each user.
    :param dict_users: Dictionary of user data indices.
    :param dataset: Dataset object containing the data.
    """
    labels = np.array(dataset.targets)
    for user, indices in dict_users.items():
        user_labels = labels[indices]
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        label_distribution = dict(zip(unique_labels, counts))
        #print(f"User {user}: Label distribution {label_distribution}")
        print(f' User {user}: Label Distribution: {label_distribution} | counts: {counts}')

if __name__ == '__main__':
    ################################### hyperparameter setup ########################################
    args = call_parser()
    
    torch.manual_seed(args.seed + args.repeat)
    torch.cuda.manual_seed(args.seed + args.repeat)
    np.random.seed(args.seed + args.repeat)
    
    #args, dataset_train, dataset_test, dict_users = data_setup(args)
    
    # args, dataset_train, dataset_test, dict_users = data_setup(args)
    args.num_classes = 10
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define the label flip mapping for odd values only, swapping them in descending order
    #flip_mapping = {1: 9, 3: 7, 5: 5, 7: 3, 9: 1}
    #flip_fraction = 0.1  # Example: 10% of users will have their labels flipped
    path = './data/fmnist'
    
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
    dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
    
    
    dict_users = fmnist_noniid(dataset_train, num_users=args.num_users)
    
    
    print("{:<50}".format("=" * 15 + " data setup " + "=" * 50)[0:60])
    print('length of dataset:{}'.format(len(dataset_train) + len(dataset_test)))
    print('num. of training data:{}'.format(len(dataset_train)))
    print('num. of testing data:{}'.format(len(dataset_test)))
    print('num. of classes:{}'.format(args.num_classes))
    print('num. of users:{}'.format(len(dict_users)))
    
    sample_per_users = int(sum([len(dict_users[i]) for i in range(len(dict_users))]) / len(dict_users))
    
    sample_per_users = 25000
    
    print('num. of samples per user:{}'.format(sample_per_users))
    if args.dataset == 'fmnist' or args.dataset == 'cifar':
        dataset_test, val_set = torch.utils.data.random_split(dataset_test, [9000, 1000])
        print(len(dataset_test), len(val_set))
    elif args.dataset == 'svhn':
        dataset_test, val_set = torch.utils.data.random_split(dataset_test, [len(dataset_test) - 2000, 2000])
        print(len(dataset_test), len(val_set))

    print("{:<50}".format("=" * 15 + " log path " + "=" * 50)[0:60])
    log_path = set_log_path(args)
    print(log_path)

    args, net_glob = model_setup(args)
    print("{:<50}".format("=" * 15 + " model setup " + "=" * 50)[0:60])
    
    # Verify the non-IID distribution
    #verify_noniid_distribution(dict_users, np.array(dataset_train.targets), args.num_users)
    
    # Verify the non-IID distribution
    print("{:<50}".format("=" * 15 + " verifying non-IID distribution " + "=" * 50)[0:60])
    calculate_label_distribution(dict_users, dataset_train)
    
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
    for i in range(args.num_users):
        dataset = DatasetSplit(dataset_train, dict_users[i])
        ldr_train = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        data_loader_list.append(ldr_train)
    ldr_train_public = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    
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

    t2 = time.time()
    hours, rem = divmod(t2 - t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))



########### baseline ###################
######## 100 Users/ Clients #########

'''
(venv) jahanxb@alsharif-Lambda-Vector:/jahanxbProject/flcode$ python main_fed_noniid_baseline.py --dataset fmnist --round 10 --tau 10 --gpu 0 --num_users 100 --frac 1 --iid 0
=============== data setup =================================
length of dataset:70000
num. of training data:60000
num. of testing data:10000
num. of classes:10
num. of users:100
num. of samples per user:25000
9000 1000
=============== log path ===================================
./log/fmnist/fedavg/cnn/round_10_tau_10_frac_1.0
=============== model setup ================================
=============== verifying non-IID distribution =============
 User 0: Label Distribution: {4: 100} | counts: [100]
 User 1: Label Distribution: {0: 100} | counts: [100]
 User 2: Label Distribution: {0: 100, 4: 100, 5: 100} | counts: [100 100 100]
 User 3: Label Distribution: {8: 100, 9: 100} | counts: [100 100]
 User 4: Label Distribution: {2: 100, 6: 100, 7: 100, 8: 100} | counts: [100 100 100 100]
 User 5: Label Distribution: {6: 200} | counts: [200]
 User 6: Label Distribution: {4: 200} | counts: [200]
 User 7: Label Distribution: {2: 100} | counts: [100]
 User 8: Label Distribution: {0: 100, 1: 100, 7: 200, 9: 100} | counts: [100 100 200 100]
 User 9: Label Distribution: {5: 100, 6: 100, 9: 100} | counts: [100 100 100]
 User 10: Label Distribution: {0: 100, 4: 100, 6: 100, 7: 200} | counts: [100 100 100 200]
 User 11: Label Distribution: {0: 100} | counts: [100]
 User 12: Label Distribution: {9: 100} | counts: [100]
 User 13: Label Distribution: {6: 100} | counts: [100]
 User 14: Label Distribution: {2: 100, 5: 100, 6: 100} | counts: [100 100 100]
 User 15: Label Distribution: {9: 100} | counts: [100]
 User 16: Label Distribution: {2: 100, 3: 100, 4: 100, 7: 100, 8: 100} | counts: [100 100 100 100 100]
 User 17: Label Distribution: {3: 200, 5: 100} | counts: [200 100]
 User 18: Label Distribution: {0: 100, 9: 100} | counts: [100 100]
 User 19: Label Distribution: {2: 200, 4: 100, 6: 100, 8: 100} | counts: [200 100 100 100]
 User 20: Label Distribution: {2: 100, 8: 100} | counts: [100 100]
 User 21: Label Distribution: {0: 200, 3: 100, 5: 100} | counts: [200 100 100]
 User 22: Label Distribution: {1: 200, 2: 100, 5: 100, 8: 100} | counts: [200 100 100 100]
 User 23: Label Distribution: {5: 100, 7: 100} | counts: [100 100]
 User 24: Label Distribution: {2: 100, 3: 100} | counts: [100 100]
 User 25: Label Distribution: {2: 100, 5: 100, 6: 200, 8: 100} | counts: [100 100 200 100]
 User 26: Label Distribution: {1: 100, 3: 200} | counts: [100 200]
 User 27: Label Distribution: {1: 100, 4: 100, 6: 100, 7: 200} | counts: [100 100 100 200]
 User 28: Label Distribution: {3: 100, 9: 100} | counts: [100 100]
 User 29: Label Distribution: {1: 100, 3: 100, 4: 100, 7: 100, 9: 100} | counts: [100 100 100 100 100]
 User 30: Label Distribution: {6: 200} | counts: [200]
 User 31: Label Distribution: {0: 100, 4: 100, 9: 100} | counts: [100 100 100]
 User 32: Label Distribution: {0: 100, 8: 100} | counts: [100 100]
 User 33: Label Distribution: {2: 100, 5: 100, 8: 100, 9: 100} | counts: [100 100 100 100]
 User 34: Label Distribution: {2: 200, 5: 100, 9: 100} | counts: [200 100 100]
 User 35: Label Distribution: {0: 100, 1: 100, 6: 100} | counts: [100 100 100]
 User 36: Label Distribution: {0: 100, 2: 100, 9: 100} | counts: [100 100 100]
 User 37: Label Distribution: {6: 100, 8: 200, 9: 100} | counts: [100 200 100]
 User 38: Label Distribution: {1: 200, 2: 100, 3: 100, 6: 100} | counts: [200 100 100 100]
 User 39: Label Distribution: {1: 100, 2: 100, 3: 100, 9: 200} | counts: [100 100 100 200]
 User 40: Label Distribution: {2: 100, 5: 200, 6: 100, 9: 100} | counts: [100 200 100 100]
 User 41: Label Distribution: {4: 100} | counts: [100]
 User 42: Label Distribution: {4: 100, 7: 100, 8: 300} | counts: [100 100 300]
 User 43: Label Distribution: {4: 100} | counts: [100]
 User 44: Label Distribution: {1: 100, 3: 100, 4: 100, 8: 100} | counts: [100 100 100 100]
 User 45: Label Distribution: {6: 100} | counts: [100]
 User 46: Label Distribution: {2: 100} | counts: [100]
 User 47: Label Distribution: {9: 100} | counts: [100]
 User 48: Label Distribution: {0: 100, 2: 200, 6: 100, 7: 100} | counts: [100 200 100 100]
 User 49: Label Distribution: {2: 100, 3: 100, 4: 100, 7: 200} | counts: [100 100 100 200]
 User 50: Label Distribution: {2: 100, 5: 100} | counts: [100 100]
 User 51: Label Distribution: {4: 100, 8: 100} | counts: [100 100]
 User 52: Label Distribution: {1: 100, 2: 100, 6: 100} | counts: [100 100 100]
 User 53: Label Distribution: {3: 100, 8: 100} | counts: [100 100]
 User 54: Label Distribution: {2: 200, 5: 100, 6: 100, 8: 100} | counts: [200 100 100 100]
 User 55: Label Distribution: {2: 100, 6: 200, 8: 100, 9: 100} | counts: [100 200 100 100]
 User 56: Label Distribution: {1: 200, 3: 100, 4: 100} | counts: [200 100 100]
 User 57: Label Distribution: {1: 100, 3: 200, 4: 100, 8: 100} | counts: [100 200 100 100]
 User 58: Label Distribution: {9: 100} | counts: [100]
 User 59: Label Distribution: {0: 100, 4: 100, 6: 100} | counts: [100 100 100]
 User 60: Label Distribution: {0: 100, 1: 100, 2: 100, 3: 100, 9: 100} | counts: [100 100 100 100 100]
 User 61: Label Distribution: {1: 200, 5: 100} | counts: [200 100]
 User 62: Label Distribution: {3: 200, 9: 100} | counts: [200 100]
 User 63: Label Distribution: {0: 100} | counts: [100]
 User 64: Label Distribution: {5: 100} | counts: [100]
 User 65: Label Distribution: {6: 100} | counts: [100]
 User 66: Label Distribution: {3: 200, 5: 100} | counts: [200 100]
 User 67: Label Distribution: {1: 100, 4: 100, 7: 200, 8: 100} | counts: [100 100 200 100]
 User 68: Label Distribution: {5: 100, 6: 100, 7: 100, 9: 200} | counts: [100 100 100 200]
 User 69: Label Distribution: {0: 100, 1: 100, 4: 100, 9: 100} | counts: [100 100 100 100]
 User 70: Label Distribution: {2: 100, 4: 100, 8: 200} | counts: [100 100 200]
 User 71: Label Distribution: {2: 200, 3: 100, 9: 200} | counts: [200 100 200]
 User 72: Label Distribution: {3: 100, 5: 300, 8: 100} | counts: [100 300 100]
 User 73: Label Distribution: {8: 100} | counts: [100]
 User 74: Label Distribution: {3: 100} | counts: [100]
 User 75: Label Distribution: {0: 100, 4: 100, 9: 100} | counts: [100 100 100]
 User 76: Label Distribution: {1: 100, 3: 100, 7: 200, 8: 100} | counts: [100 100 200 100]
 User 77: Label Distribution: {0: 200, 1: 200, 2: 100} | counts: [200 200 100]
 User 78: Label Distribution: {0: 100, 3: 100, 4: 100, 5: 100} | counts: [100 100 100 100]
 User 79: Label Distribution: {1: 100, 4: 100, 5: 100} | counts: [100 100 100]
 User 80: Label Distribution: {2: 100} | counts: [100]
 User 81: Label Distribution: {1: 100, 7: 100} | counts: [100 100]
 User 82: Label Distribution: {3: 100, 4: 100, 5: 100, 6: 100, 7: 100} | counts: [100 100 100 100 100]
 User 83: Label Distribution: {0: 100, 1: 100, 6: 100, 9: 100} | counts: [100 100 100 100]
 User 84: Label Distribution: {0: 200, 1: 300} | counts: [200 300]
 User 85: Label Distribution: {1: 100, 3: 100, 4: 100, 8: 200} | counts: [100 100 100 200]
 User 86: Label Distribution: {2: 100, 8: 100} | counts: [100 100]
 User 87: Label Distribution: {1: 100, 4: 100} | counts: [100 100]
 User 88: Label Distribution: {0: 100, 1: 100, 8: 100} | counts: [100 100 100]
 User 89: Label Distribution: {1: 100, 6: 100, 7: 100, 9: 100} | counts: [100 100 100 100]
 User 90: Label Distribution: {0: 100, 4: 100} | counts: [100 100]
 User 91: Label Distribution: {7: 100} | counts: [100]
 User 92: Label Distribution: {3: 100, 6: 100, 8: 100, 9: 100} | counts: [100 100 100 100]
 User 93: Label Distribution: {4: 100, 5: 100, 6: 100, 7: 100} | counts: [100 100 100 100]
 User 94: Label Distribution: {6: 100, 7: 300} | counts: [100 300]
 User 95: Label Distribution: {5: 100} | counts: [100]
 User 96: Label Distribution: {2: 100} | counts: [100]
 User 97: Label Distribution: {0: 100, 7: 100, 9: 100} | counts: [100 100 100]
 User 98: Label Distribution: {3: 100, 4: 100, 6: 100, 7: 100, 9: 100} | counts: [100 100 100 100 100]
 User 99: Label Distribution: {2: 100, 5: 100, 7: 200, 9: 100} | counts: [100 100 200 100]
=============== training... ================================
t   0: train_loss = 0.715, norm = 2.129, test_acc = 55.178
t   1: train_loss = 0.411, norm = 1.478, test_acc = 66.356
t   2: train_loss = 0.234, norm = 1.256, test_acc = 74.311
t   3: train_loss = 0.200, norm = 1.119, test_acc = 72.122
t   4: train_loss = 0.184, norm = 1.031, test_acc = 71.533
t   5: train_loss = 0.178, norm = 0.979, test_acc = 70.089
t   6: train_loss = 0.183, norm = 0.977, test_acc = 70.244
t   7: train_loss = 0.182, norm = 0.941, test_acc = 72.900
t   8: train_loss = 0.161, norm = 0.901, test_acc = 74.244
t   9: train_loss = 0.152, norm = 0.865, test_acc = 74.856
training time: 00:05:48.16

'''

############## baseline ################
######## 10 Users/ Clients #########
'''
(venv) jahanxb@alsharif-Lambda-Vector:/jahanxbProject/flcode$ python main_fed_noniid_baseline.py --dataset fmnist --round 10 --tau 10 --gpu 0 --num_users 10 --frac 1 --iid 0
=============== data setup =================================
length of dataset:70000
num. of training data:60000
num. of testing data:10000
num. of classes:10
num. of users:10
num. of samples per user:25000
9000 1000
=============== log path ===================================
./log/fmnist/fedavg/cnn/round_10_tau_10_frac_1.0
=============== model setup ================================
=============== verifying non-IID distribution =============
 User 0: Label Distribution: {4: 100} | counts: [100]
 User 1: Label Distribution: {0: 100} | counts: [100]
 User 2: Label Distribution: {0: 100, 4: 100, 5: 100} | counts: [100 100 100]
 User 3: Label Distribution: {8: 100, 9: 100} | counts: [100 100]
 User 4: Label Distribution: {2: 100, 6: 100, 7: 100, 8: 100} | counts: [100 100 100 100]
 User 5: Label Distribution: {6: 200} | counts: [200]
 User 6: Label Distribution: {4: 200} | counts: [200]
 User 7: Label Distribution: {2: 100} | counts: [100]
 User 8: Label Distribution: {0: 100, 1: 100, 7: 200, 9: 100} | counts: [100 100 200 100]
 User 9: Label Distribution: {5: 100, 6: 100, 9: 100} | counts: [100 100 100]
=============== training... ================================
t   0: train_loss = 0.640, norm = 1.562, test_acc = 40.433
t   1: train_loss = 0.302, norm = 1.392, test_acc = 32.600
t   2: train_loss = 0.182, norm = 1.612, test_acc = 37.567
t   3: train_loss = 0.631, norm = 1.501, test_acc = 36.833
t   4: train_loss = 0.334, norm = 1.297, test_acc = 41.878
t   5: train_loss = 0.139, norm = 1.087, test_acc = 48.778
t   6: train_loss = 0.206, norm = 1.169, test_acc = 40.278
t   7: train_loss = 0.187, norm = 1.139, test_acc = 50.156
t   8: train_loss = 0.206, norm = 1.035, test_acc = 47.778
t   9: train_loss = 0.123, norm = 1.022, test_acc = 55.722
training time: 00:00:35.54
'''