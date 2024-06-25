# import copy
# import numpy as np
# import time, math
# import torch

# from utils.data_utils_noniid import DatasetSplit
# from utils.model_utils import *
# from utils.aggregation import *
# from options import call_parser
# from models.Update import LocalUpdate
# from models.test import test_img
# from torch.utils.data import DataLoader

# import os
# import torch, dill
# from torchvision import datasets, transforms
# from torch.utils.data import Dataset

# import warnings
# warnings.filterwarnings("ignore")
# torch.cuda.is_available()

# # def flip_labels(labels, flip_mapping):
# #     """
# #     Flip labels according to the provided mapping.
# #     :param labels: Original labels.
# #     :param flip_mapping: Dictionary mapping original labels to new labels.
# #     :return: Flipped labels.
# #     """
# #     flipped_labels = labels.copy()
# #     for original_label, new_label in flip_mapping.items():
# #         flipped_labels[labels == original_label] = new_label
# #     return flipped_labels

# # def fmnist_noniid(dataset, num_users, flip_mapping, flip_fraction):
# #     """
# #     Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack.
# #     :param dataset: Dataset object containing the data.
# #     :param num_users: Number of users (clients).
# #     :param flip_mapping: Dictionary mapping original labels to flipped labels.
# #     :param flip_fraction: Fraction of users whose labels will be flipped.
# #     :return: dict of image index.
# #     """
# #     num_shards, num_imgs = 200, 300
# #     idx_shard = [i for i in range(num_shards)]
# #     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
# #     idxs = np.arange(num_shards * num_imgs)
# #     labels = np.array(dataset.targets)

# #     # Sort labels
# #     idxs_labels = np.vstack((idxs, labels))
# #     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
# #     idxs = idxs_labels[0, :]

# #     # Select random clients for label flipping
# #     num_flip_users = int(flip_fraction * num_users)
# #     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
# #     print(f"Flipping labels for users: {flip_users}")

# #     # Divide and assign
# #     for i in range(num_users):
# #         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
# #         idx_shard = list(set(idx_shard) - rand_set)
# #         user_data = []
# #         for rand in rand_set:
# #             user_data.extend(idxs[rand * num_imgs: (rand + 1) * num_imgs])
# #         user_data = np.array(user_data)
# #         #print('flip users: ',flip_users)
        
# #         if i in flip_users:
# #             print('i in flip_users: ',i)
            
# #             print('labels: ,', labels)
            
# #             print(
# #                 'user:data  ,',user_data
# #             )
# #             user_labels = labels[user_data]
            
# #             print('user_labels: ',user_labels)
            
# #             exit()
# #             print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
            
# #             '''I think this code written down is the M>F problem...'''
# #             flipped_labels = flip_labels(user_labels, flip_mapping)
# #             print('flipped labels: ',flipped_labels)
            
# #             labels[user_data] = flipped_labels
# #             print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data])}")
# #         dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)
    
# #     # Update the dataset labels with the flipped labels
# #     dataset.targets = torch.tensor(labels)
# #     return dict_users


# def flip_labels(labels, flip_mapping):
#     """
#     Flip labels according to the provided mapping.
#     :param labels: Original labels.
#     :param flip_mapping: Dictionary mapping original labels to new labels.
#     :return: Flipped labels.
#     """
#     flipped_labels = labels.copy()
    
#     print('flipped labels: ,', flipped_labels)
#     for original_label, new_label in flip_mapping.items():
#         print('+'*10)
#         print('original label: ',original_label)
#         print('new_label: ',new_label)
#         flipped_labels[labels == original_label] = new_label
#         print('flipped_labels: ',flipped_labels[11])
        
#     return flipped_labels



# def fmnist_noniid(dataset, num_users, flip_mapping, flip_fraction):
#     """
#     Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack.
#     :param dataset: Dataset object containing the data.
#     :param num_users: Number of users (clients).
#     :param flip_mapping: Dictionary mapping original labels to flipped labels.
#     :param flip_fraction: Fraction of users whose labels will be flipped.
#     :return: dict of image index.
#     """
#     labels = np.array(dataset.targets)
#     idxs = np.arange(len(labels))
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

#     # Shuffle the indices
#     np.random.shuffle(idxs)

#     # Split the indices among users
#     split_idxs = np.array_split(idxs, num_users)

#     # Select random clients for label flipping
#     num_flip_users = int(flip_fraction * num_users)
#     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
#     print(f"Flipping labels for users: {flip_users}")

#     # Assign data to users and flip labels if necessary
#     for i in range(num_users):
#         user_data = split_idxs[i]
#         if i in flip_users:
#             user_labels = labels[user_data]
#             print(f"Before flipping: User {i}, Labels: {user_labels}")
#             flipped_labels = flip_labels(user_labels, flip_mapping)
#             labels[user_data] = flipped_labels
#             print(f"After flipping: User {i}, Labels: {flipped_labels}")
#         dict_users[i] = user_data
    
#     # Update the dataset labels with the flipped labels
#     dataset.targets = torch.tensor(labels)
#     return dict_users




# if __name__ == '__main__':
#     ################################### hyperparameter setup ########################################
#     args = call_parser()
    
#     torch.manual_seed(args.seed + args.repeat)
#     torch.cuda.manual_seed(args.seed + args.repeat)
#     np.random.seed(args.seed + args.repeat)
    
#     # args, dataset_train, dataset_test, dict_users = data_setup(args)
#     args.num_classes = 10
#     args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Define the label flip mapping for odd values only, swapping them in descending order
#     flip_mapping = {1: 9, 3: 7, 5: 5, 7: 3, 9: 1}
#     flip_fraction = 0.1  # Example: 10% of users will have their labels flipped
#     path = './data/fmnist'
    
#     trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#     dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
#     dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
    
#     dict_users = fmnist_noniid(dataset_train, num_users=args.num_users, flip_mapping=flip_mapping, flip_fraction=flip_fraction)
    
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
            
#             # clipping local model or not ? : no clip for cifar10
#             # threshold = delta_norm / args.clip
#             # if threshold > 1.0:
#             #     for k in model_update.keys():
#             #         model_update[k] = model_update[k] / threshold
            
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


########################################
'''
---------------------------------------
'''
#########$$$$$$$$$$$$$$$$$$$$$$$$$$$$######

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

import os
import torch, dill
from torchvision import datasets, transforms
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore")
torch.cuda.is_available()

def flip_labels(labels, flip_mapping):
    """
    Flip labels according to the provided mapping.
    :param labels: Original labels.
    :param flip_mapping: Dictionary mapping original labels to new labels.
    :return: Flipped labels.
    """
    flipped_labels = labels.copy()
    for original_label, new_label in flip_mapping.items():
        flipped_labels[labels == original_label] = new_label
    return flipped_labels

def fmnist_noniid(dataset, num_users, flip_mapping, flip_fraction):
    """
    Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack.
    :param dataset: Dataset object containing the data.
    :param num_users: Number of users (clients).
    :param flip_mapping: Dictionary mapping original labels to flipped labels.
    :param flip_fraction: Fraction of users whose labels will be flipped.
    :return: dict of image index.
    """
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Shuffle the indices
    np.random.shuffle(idxs)

    # Sort labels to create non-IID distribution
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Split the indices among users in a non-IID way
    num_shards, num_imgs = num_users * 2, len(dataset) // (num_users * 2)
    shard_idxs = [i for i in range(num_shards)]
    for i in range(num_users):
        rand_set = set(np.random.choice(shard_idxs, 2, replace=False))
        shard_idxs = list(set(shard_idxs) - rand_set)
        for rand in rand_set:
            user_data = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)

    # Select random clients for label flipping
    num_flip_users = int(flip_fraction * num_users)
    flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
    print(f"Flipping labels for users: {flip_users}")

    # Flip labels for selected users
    for i in flip_users:
        user_data = dict_users[i]
        user_labels = labels[user_data]
        print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
        flipped_labels = flip_labels(user_labels, flip_mapping)
        labels[user_data] = flipped_labels
        print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data])}")
    
    # Update the dataset labels with the flipped labels
    dataset.targets = torch.tensor(labels)
    return dict_users







def fmnist_noniid_strong(dataset, num_users, flip_mapping, flip_fraction):
    """
    Strongly non-I.I.D client data from FashionMNIST dataset with optional label flipping attack.
    :param dataset: Dataset object containing the data.
    :param num_users: Number of users (clients).
    :param flip_mapping: Dictionary mapping original labels to flipped labels.
    :param flip_fraction: Fraction of users whose labels will be flipped.
    :return: dict of image index.
    """
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    # Shuffle the indices
    np.random.shuffle(idxs)

    # Sort labels to create a strongly non-IID distribution
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Number of shards should be a multiple of num_users for strong non-IID
    num_shards = num_users * 2
    shard_size = len(dataset) // num_shards

    # Assign each user a fixed number of shards (e.g., 2)
    for i in range(num_users):
        shard_idxs = np.random.choice(num_shards, 2, replace=False)
        for shard in shard_idxs:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[shard * shard_size:(shard + 1) * shard_size]), axis=0
            )
        num_shards -= 2

    # Select random clients for label flipping
    num_flip_users = int(flip_fraction * num_users)
    flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
    print(f"Flipping labels for users: {flip_users}")

    # Flip labels for selected users
    for i in flip_users:
        user_data = dict_users[i]
        user_labels = labels[user_data]
        print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
        flipped_labels = flip_labels(user_labels, flip_mapping)
        labels[user_data] = flipped_labels
        print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data])}")

    # Update the dataset labels with the flipped labels
    dataset.targets = torch.tensor(labels)
    return dict_users













def verify_noniid_distribution(dict_users, labels, num_users):
    for i in range(num_users):
        user_labels = labels[dict_users[i]]
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        print(f"User {i}: Label distribution {dict(zip(unique_labels, counts))}")

if __name__ == '__main__':
    ################################### hyperparameter setup ########################################
    args = call_parser()
    
    torch.manual_seed(args.seed + args.repeat)
    torch.cuda.manual_seed(args.seed + args.repeat)
    np.random.seed(args.seed + args.repeat)
    
    # args, dataset_train, dataset_test, dict_users = data_setup(args)
    args.num_classes = 10
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define the label flip mapping for odd values only, swapping them in descending order
    flip_mapping = {1: 9, 3: 7, 5: 5, 7: 3, 9: 1}
    flip_fraction = 0.1  # Example: 10% of users will have their labels flipped
    path = './data/fmnist'
    
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
    dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
    
    dict_users = fmnist_noniid_strong(dataset_train, num_users=args.num_users, flip_mapping=flip_mapping, flip_fraction=flip_fraction)
    
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
    verify_noniid_distribution(dict_users, np.array(dataset_train.targets), args.num_users)
    
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
            
            # clipping local model or not ? : no clip for cifar10
            # threshold = delta_norm / args.clip
            # if threshold > 1.0:
            #     for k in model_update.keys():
            #         model_update[k] = model_update[k] / threshold
            
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
