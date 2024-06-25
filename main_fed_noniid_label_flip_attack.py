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

# def flip_labels(labels, flip_mapping):
#     """
#     Flip labels according to the provided mapping.
#     :param labels: Original labels.
#     :param flip_mapping: Dictionary mapping original labels to new labels.
#     :return: Flipped labels.
#     """
#     flipped_labels = labels.copy()
#     for original_label, new_label in flip_mapping.items():
#         flipped_labels[labels == original_label] = new_label
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
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = np.array(dataset.targets)

#     # Sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Select random clients for label flipping
#     num_flip_users = int(flip_fraction * num_users)
#     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
#     print(f"Flipping labels for users: {flip_users}")

#     # Divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         user_data = []
#         for rand in rand_set:
#             user_data.extend(idxs[rand * num_imgs: (rand + 1) * num_imgs])
#         user_data = np.array(user_data)
#         #print('flip users: ',flip_users)
        
#         if i in flip_users:
#             print('i in flip_users: ',i)
            
#             print('labels: ,', labels)
            
#             print(
#                 'user:data  ,',user_data
#             )
#             user_labels = labels[user_data]
            
#             print('user_labels: ',user_labels)
            
#             exit()
#             print(f"Before flipping: User {i}, Labels: {np.unique(user_labels)}")
            
#             '''I think this code written down is the M>F problem...'''
#             flipped_labels = flip_labels(user_labels, flip_mapping)
#             print('flipped labels: ',flipped_labels)
            
#             labels[user_data] = flipped_labels
#             print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data])}")
#         dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)
    
#     # Update the dataset labels with the flipped labels
#     dataset.targets = torch.tensor(labels)
#     return dict_users

'''
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

'''

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



def fmnist_noniid_1(dataset, num_users, flip_mapping, flip_fraction):
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
        print(f"Before flipping: User {i}, Labels: {np.unique(user_labels, return_counts=True)}")
        flipped_labels = flip_labels(user_labels, flip_mapping)
        labels[user_data] = flipped_labels
        print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data], return_counts=True)}")
    
    # Update the dataset labels with the flipped labels
    dataset.targets = torch.tensor(labels)
    return dict_users


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

    '''
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
    '''
    
       # Divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            user_data = idxs[rand * num_imgs: (rand + 1) * num_imgs]
            if i in flip_users:
                user_labels = labels[rand * num_imgs: (rand + 1) * num_imgs]
                print(f"Before flipping: User {i}, Labels: {np.unique(user_labels, return_counts=True)}")
                flipped_labels = flip_labels(user_labels, flip_mapping)
                labels[rand * num_imgs: (rand + 1) * num_imgs] = flipped_labels
                print(f"After flipping: User {i}, Labels: {np.unique(labels[rand * num_imgs: (rand + 1) * num_imgs], return_counts=True)}")
                # Verify that labels were flipped correctly
                for original_label, new_label in flip_mapping.items():
                    if original_label in user_labels:
                        original_count = np.sum(user_labels == original_label)
                        new_count = np.sum(flipped_labels == new_label)
                        print(f"Label {original_label} flipped to {new_label}: {original_count == new_count}")
            dict_users[i] = np.concatenate((dict_users[i], user_data), axis=0)

    
    
    
    return dict_users


def verify_noniid_distribution(dict_users, labels, num_users):
    for i in range(num_users):
        user_labels = labels[dict_users[i]]
        unique_labels, counts = np.unique(user_labels, return_counts=True)
        print(f"User {i}: Label distribution {dict(zip(unique_labels, counts))}")



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
        print(f"User {user}: Label distribution {label_distribution}")



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
    
    dict_users = fmnist_noniid(dataset_train, num_users=args.num_users, flip_mapping=flip_mapping, flip_fraction=flip_fraction)
    
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



'''
The output provided shows that each user has only two unique labels, which is indicative of a non-IID (non-Independent and Identically Distributed) setting. In an IID setting, each user would typically have a more uniform distribution of all possible labels.

Here's a summary of why this indicates non-IID:

Limited Labels per User: Each user only has 2-3 different labels out of the total 10 possible labels. For instance, User 0 has labels {2, 8}, User 1 has labels {0, 5}, etc.
Label Distribution: The label distribution among the users is not balanced across all labels, meaning not every user has a uniform representation of all classes. Some labels are completely missing from certain users.
This is characteristic of a non-IID distribution where data is grouped by certain characteristics (labels in this case), and each user's data is skewed towards a subset of labels. If the distribution were IID, we would expect each user to have an approximately equal number of samples from each class.

Thus, based on the label distributions you provided, your data indeed seems to be non-IID. If the label flipping is not working as expected, you may need to ensure the logic for flipping labels is correctly applied.


(venv) jahanxb@alsharif-Lambda-Vector:/jahanxbProject/flcode$ python main_fed_noniid_label_flip_attack.py --dataset fmnist --round 10 --gpu 0 --tau 10 --num_users 10 --frac 1 --iid 0
Flipping labels for users: [4]
Before flipping: User 4, Labels: [0 1 2 3 4 5 6 7 8 9]
After flipping: User 4, Labels: [0 1 2 3 4 5 6 7 8 9]
Before flipping: User 4, Labels: [0 1 2 3 4 5 6 7 8 9]
After flipping: User 4, Labels: [0 1 2 3 4 5 6 7 8 9]
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
User 0: Label distribution {2: 300, 8: 300}
User 1: Label distribution {0: 300, 5: 300}
User 2: Label distribution {2: 300, 5: 300}
User 3: Label distribution {6: 300, 8: 300}
User 4: Label distribution {6: 300, 9: 300}
User 5: Label distribution {0: 300, 1: 300}
User 6: Label distribution {0: 300, 8: 300}
User 7: Label distribution {1: 300, 4: 300}
User 8: Label distribution {0: 300, 3: 1, 7: 299}
User 9: Label distribution {3: 295, 6: 300, 7: 5}
=============== verifying non-IID distribution =============
User 0: Label distribution {2: 300, 8: 300}
User 1: Label distribution {0: 300, 5: 300}
User 2: Label distribution {2: 300, 5: 300}
User 3: Label distribution {6: 300, 8: 300}
User 4: Label distribution {6: 300, 9: 300}
User 5: Label distribution {0: 300, 1: 300}
User 6: Label distribution {0: 300, 8: 300}
User 7: Label distribution {1: 300, 4: 300}
User 8: Label distribution {0: 300, 3: 1, 7: 299}
User 9: Label distribution {3: 295, 6: 300, 7: 5}
=============== training... ================================
t   0: train_loss = 0.445, norm = 2.391, test_acc = 19.267
t   1: train_loss = 0.186, norm = 1.545, test_acc = 23.644
t   2: train_loss = 0.125, norm = 1.361, test_acc = 37.167
t   3: train_loss = 0.121, norm = 1.334, test_acc = 31.322
t   4: train_loss = 0.126, norm = 1.591, test_acc = 38.244
t   5: train_loss = 0.110, norm = 1.243, test_acc = 47.700
t   6: train_loss = 0.092, norm = 1.046, test_acc = 46.378
t   7: train_loss = 0.079, norm = 1.088, test_acc = 53.900
t   8: train_loss = 0.082, norm = 0.755, test_acc = 61.422
t   9: train_loss = 0.076, norm = 0.961, test_acc = 57.133
training time: 00:01:18.02

'''