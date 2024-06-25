#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import numpy as np
import torch,os, dill

# from models.Nets import MLP, CNNMnist, CNNCifar
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



class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        #print(image)
        return image, label



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


'''
Non IID for Fmnist with random clients label attack
'''

def fmnist_noniid_old(dataset, num_users, flip_mapping=None, flip_fraction=0.1):
    """
    Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack
    :param dataset: Dataset object containing the data
    :param num_users: Number of users (clients)
    :param flip_mapping: Dictionary mapping original labels to flipped labels
    :param flip_fraction: Fraction of users whose labels will be flipped
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.train_labels.numpy()

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Select random clients for label flipping
    if flip_mapping is not None:
        num_flip_users = int(flip_fraction * num_users)
        flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
    else:
        flip_users = []

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_data = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            if i in flip_users:
                user_labels = labels[rand * num_imgs: (rand + 1) * num_imgs]
                user_labels = flip_labels(user_labels, flip_mapping)
                idxs_labels[1, rand * num_imgs: (rand + 1) * num_imgs] = user_labels
            dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)
    return dict_users



# def fmnist_noniid(dataset, num_users, flip_mapping, flip_fraction):
#     """
#     Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack
#     :param dataset: Dataset object containing the data
#     :param num_users: Number of users (clients)
#     :param flip_mapping: Dictionary mapping original labels to flipped labels
#     :param flip_fraction: Fraction of users whose labels will be flipped
#     :return: dict of image index
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.train_labels.numpy()

#     # Sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Select random clients for label flipping
#     # if flip_mapping is not None:
#     #     num_flip_users = int(flip_fraction * num_users)
#     #     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
#     #     print(f"Flipping labels for users: {flip_users}")
#     # else:
#     #     flip_users = []

#     num_flip_users = int(flip_fraction * num_users)
#     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
#     print(f"Flipping labels for users: {flip_users}")

#    # Divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             user_data = idxs[rand * num_imgs: (rand + 1) * num_imgs]
#             if i in flip_users:
#                 user_labels = labels[rand * num_imgs: (rand + 1) * num_imgs]
#                 print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
#                 flipped_labels = flip_labels(user_labels, flip_mapping)
#                 labels[rand * num_imgs: (rand + 1) * num_imgs] = flipped_labels
#                 print(f"After flipping: User {i}, Labels: {np.unique(labels[rand * num_imgs: (rand + 1) * num_imgs])}")
#             dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)

#     return dict_users


def fmnist_noniid(dataset, num_users, flip_mapping, flip_fraction):
    """
    Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack
    :param dataset: Dataset object containing the data
    :param num_users: Number of users (clients)
    :param flip_mapping: Dictionary mapping original labels to flipped labels
    :param flip_fraction: Fraction of users whose labels will be flipped
    :return: dict of image index
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # Sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Select random clients for label flipping
    num_flip_users = int(flip_fraction * num_users)
    flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
    print(f"Flipping labels for users: {flip_users}")

    # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_data = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            if i in flip_users:
                user_labels = labels[rand * num_imgs: (rand + 1) * num_imgs]
                print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
                flipped_labels = flip_labels(user_labels, flip_mapping)
                labels[rand * num_imgs: (rand + 1) * num_imgs] = flipped_labels
                print(f"After flipping: User {i}, Labels: {np.unique(labels[rand * num_imgs: (rand + 1) * num_imgs])}")
            dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)

    return dict_users




################################### data setup ########################################
# def data_setup(args):
    
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#     if args.dataset == 'mnist':
#         path = './data/mnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
#         dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#                 dict_users = mnist_iid(dataset_train, args.num_users)
#                 # print(dict_users)
#                 # print(type(dict_users))
                
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             dict_users = mnist_noniid(dataset_train, args.num_users)
#     elif args.dataset == 'svhn':
#         path = './data/svhn'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
#         dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
#         dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
#         dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
#         #dataset_train = torch.utils.data.ConcatDataset([dataset_train])
#         dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#             # if 1:
#                 dict_users = svhn_iid(dataset_train, args.num_users)
#                     # print(dict_users)
#                     # print(type(dict_users))
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             exit('Error: only consider IID setting in SVHN')
#     elif args.dataset == 'emnist':
#         path = './data/emnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         # train_loader = datasets.CelebA('../data/', split='train', target_type='identity', download=True)
        
#         trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
#         dataset_train = datasets.EMNIST(path, split='balanced', train=True, download=True, transform=trans_emnist)
#         dataset_test = datasets.EMNIST(path, split='balanced', train=False, download=True, transform=trans_emnist)
#         args.num_classes = 10
#         if args.iid:
#             dict_users = emnist_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')
#     elif args.dataset == 'fmnist':
#         path = './data/fmnist'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
#         dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
#         dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f: 
#                     dict_users = dill.load(f) 
#                     # print(dict_users)
#             else:
#                 dict_users = fmnist_iid(dataset_train, args.num_users)
#                 # print(dict_users)
#                 # print(type(dict_users))
                
#                 with open(path  + "/dict_users.pik", 'wb') as f: 
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f) 
#         else:
#             exit('Error: only consider IID setting in fmnist')
#     elif args.dataset == 'cifar':
#         path = './data/cifar'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, 
#                                         transform=transforms.Compose([transforms.RandomHorizontalFlip(), 
#                                                                         transforms.RandomCrop(32, 4),
#                                                                         transforms.ToTensor(),
#                                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                         std=[0.229, 0.224, 0.225])]))
#         dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, 
#                                         transform=transforms.Compose([transforms.ToTensor(),
#                                                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                                                             std=[0.229, 0.224, 0.225])]))
#         args.num_classes = 10
#         if args.iid:
#             dict_users = cifar_iid(dataset_train, args.num_users)
#         else:
#             exit('Error: only consider IID setting in CIFAR10')

#     ###############
#     ## adding support for vanilla svhn ###
#     # ####################
#     elif args.dataset == 'vanillasvhn':
#         path = './data/vanillasvhn'
#         if not os.path.exists(path):
#             os.makedirs(path)
#         trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])

#         dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
#         dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
#         dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
#         dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
#         args.num_classes = 10
#         if args.iid:
#             # check if exist
#             if os.path.isfile(path  + "/dict_users.pik"):
#                 with open(path  + "/dict_users.pik", 'rb') as f:
#                     dict_users = dill.load(f)
#                     # print(dict_users)
#             else:
#             # if 1:
#                 dict_users = svhn_iid(dataset_train, args.num_users)
#                     # print(dict_users)
#                     # print(type(dict_users))
#                 with open(path  + "/dict_users.pik", 'wb') as f:
#                     dill.dump(dict_users, f)
#                     # pickle.dump(dict_users, f)
#         else:
#             exit('Error: only consider IID setting in SVHN')
#     ###############################  VANILLA SVHN  END ########################################

#     else:
#         exit('Error: unrecognized dataset')
    
#     return args, dataset_train, dataset_test, dict_users



def data_setup_old(args):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.dataset == 'mnist':
        path = './data/mnist'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.MNIST(path, train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST(path, train=False, download=True, transform=trans_mnist)
        args.num_classes = 10
        if args.iid:
            if os.path.isfile(path + "/dict_users.pik"):
                with open(path + "/dict_users.pik", 'rb') as f:
                    dict_users = dill.load(f)
            else:
                dict_users = mnist_iid(dataset_train, args.num_users)
                with open(path + "/dict_users.pik", 'wb') as f:
                    dill.dump(dict_users, f)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'svhn':
        path = './data/svhn'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
        dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
        dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
        dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
        args.num_classes = 10
        if args.iid:
            if os.path.isfile(path + "/dict_users.pik"):
                with open(path + "/dict_users.pik", 'rb') as f:
                    dict_users = dill.load(f)
            else:
                dict_users = svhn_iid(dataset_train, args.num_users)
                with open(path + "/dict_users.pik", 'wb') as f:
                    dill.dump(dict_users, f)
        else:
            exit('Error: only consider IID setting in SVHN')
    elif args.dataset == 'emnist':
        path = './data/emnist'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_emnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1751,), (0.3332,))])
        dataset_train = datasets.EMNIST(path, split='balanced', train=True, download=True, transform=trans_emnist)
        dataset_test = datasets.EMNIST(path, split='balanced', train=False, download=True, transform=trans_emnist)
        args.num_classes = 10
        if args.iid:
            dict_users = emnist_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in EMNIST')
    elif args.dataset == 'fmnist':
        path = './data/fmnist'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
        args.num_classes = 10
        if args.iid:
            if os.path.isfile(path + "/dict_users.pik"):
                with open(path + "/dict_users.pik", 'rb') as f:
                    dict_users = dill.load(f)
            else:
                dict_users = fmnist_iid(dataset_train, args.num_users)
                with open(path + "/dict_users.pik", 'wb') as f:
                    dill.dump(dict_users, f)
        else:
            dict_users = fmnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        path = './data/cifar'
        if not os.path.exists(path):
            os.makedirs(path)
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True,
                                         transform=transforms.Compose([transforms.RandomHorizontalFlip(),
                                                                        transforms.RandomCrop(32, 4),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                             std=[0.229, 0.224, 0.225])]))
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                           std=[0.229, 0.224, 0.225])]))
        args.num_classes = 10
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    elif args.dataset == 'vanillasvhn':
        path = './data/vanillasvhn'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.43090966, 0.4302428, 0.44634357), (0.19759192, 0.20029082, 0.19811132))])
        dataset_train = datasets.SVHN(path, split='train', download=True, transform=trans_svhn)
        dataset_extra = datasets.SVHN(path, split='extra', download=True, transform=trans_svhn)
        dataset_train = torch.utils.data.ConcatDataset([dataset_train, dataset_extra])
        dataset_test = datasets.SVHN(path, split='test', download=True, transform=trans_svhn)
        args.num_classes = 10
        if args.iid:
            if os.path.isfile(path + "/dict_users.pik"):
                with open(path + "/dict_users.pik", 'rb') as f:
                    dict_users = dill.load(f)
            else:
                dict_users = svhn_iid(dataset_train, args.num_users)
                with open(path + "/dict_users.pik", 'wb') as f:
                    dill.dump(dict_users, f)
        else:
            exit('Error: only consider IID setting in vanillasvhn')
    else:
        exit('Error: unrecognized dataset')

    return args, dataset_train, dataset_test, dict_users





def data_setup(args):
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    if args.dataset == 'fmnist':
        path = './data/fmnist'
        if not os.path.exists(path):
            os.makedirs(path)
        trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
        dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
        args.num_classes = 10
        if args.iid:
            if os.path.isfile(path + "/dict_users.pik"):
                with open(path + "/dict_users.pik", 'rb') as f:
                    dict_users = dill.load(f)
            else:
                dict_users = fmnist_iid(dataset_train, args.num_users)
                with open(path + "/dict_users.pik", 'wb') as f:
                    dill.dump(dict_users, f)
        else:
            # Define the label flip mapping for odd values only
            flip_mapping = {1: 9, 3: 7, 5: 5, 7: 3, 9: 1}
            flip_fraction = 0.1  # Example: 10% of users will have their labels flipped
            dict_users = fmnist_noniid(dataset_train, args.num_users, flip_mapping, flip_fraction)
    else:
        exit('Error: unrecognized dataset')
    return args, dataset_train, dataset_test, dict_users

###################### utils #################################################
def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def svhn_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        # all_idxs=random.shuffle(all_idxs)
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    
    return dict_users

def fmnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from fashion MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    # print(dataset[0][0].size)
    return dict_users

def emnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from eMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
  
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
 
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def emnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
   
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

