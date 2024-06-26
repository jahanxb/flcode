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



def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    :param vec1: First vector.
    :param vec2: Second vector.
    :return: Cosine similarity value.
    """
    dot_product = torch.dot(vec1, vec2)
    norm_vec1 = torch.norm(vec1)
    norm_vec2 = torch.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity.item()

# def identify_malicious_users(updates, z_score_threshold=2.0):
#     """
#     Identify potentially malicious users based on cosine similarity of their updates.
#     :param updates: List of user updates (each update is a dictionary of model parameter tensors).
#     :param z_score_threshold: Z-score threshold for detecting outliers.
#     :return: List of indices of potentially malicious users.
#     """
#     num_users = len(updates)
#     update_vectors = []

#     # Flatten the model updates into vectors
#     for update in updates:
#         vector = torch.cat([torch.flatten(param) for param in update.values()])
#         update_vectors.append(vector)

#     # Aggregate the global update
#     global_update = torch.mean(torch.stack(update_vectors), dim=0)

#     # Compute cosine similarity to the global update
#     similarities = []
#     for i in range(num_users):
#         similarity = cosine_similarity(update_vectors[i], global_update)
#         similarities.append(similarity)

#     # Compute z-scores of the similarities
#     mean_similarity = np.mean(similarities)
#     std_similarity = np.std(similarities)
#     z_scores = [(similarity - mean_similarity) / std_similarity for similarity in similarities]

#     # Identify malicious users based on z-score threshold
#     malicious_users = [i for i in range(num_users) if z_scores[i] < -z_score_threshold]
    
#     return malicious_users



def old_identify_malicious_users(updates, update_indices, z_score_threshold=2.0):
    """
    Identify potentially malicious users based on cosine similarity of their updates.
    :param updates: List of user updates (each update is a dictionary of model parameter tensors).
    :param update_indices: The indices of the users providing the updates.
    :param z_score_threshold: Z-score threshold for detecting outliers.
    :return: List of actual user IDs identified as potentially malicious.
    """
    num_users = len(updates)
    update_vectors = []

    
    # Flatten the model updates into vectors
    for update in updates:
        vector = torch.cat([torch.flatten(param) for param in update.values()])
        update_vectors.append(vector)

    # Aggregate the global update
    global_update = torch.mean(torch.stack(update_vectors), dim=0)

    # Compute cosine similarity to the global update
    similarities = []
    for vector in update_vectors:
        similarity = cosine_similarity(vector, global_update)
        similarities.append(similarity)
    
    # Compute z-scores of the similarities
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    z_scores = [(similarity - mean_similarity) / std_similarity for similarity in similarities]

    # Identify malicious users based on z-score threshold
    malicious_indices = [i for i, z_score in enumerate(z_scores) if z_score < -z_score_threshold]
    malicious_user_ids = [update_indices[i] for i in malicious_indices]  # Map indices to global user IDs

    return malicious_user_ids


def identify_malicious_users(updates, update_indices, z_score_threshold=2.0):
    """
    Identify potentially malicious users based on cosine similarity of their updates.
    :param updates: List of user updates (each update is a dictionary of model parameter tensors).
    :param update_indices: The indices of the users providing the updates.
    :param z_score_threshold: Z-score threshold for detecting outliers.
    :return: List of actual user IDs identified as potentially malicious.
    """
    num_users = len(updates)
    update_vectors = []

    # Flatten the model updates into vectors
    for update in updates:
        vector = torch.cat([torch.flatten(param) for param in update.values()])
        update_vectors.append(vector)

    # Aggregate the global update
    global_update = torch.mean(torch.stack(update_vectors), dim=0)

    # Compute cosine similarity to the global update
    similarities = []
    for vector in update_vectors:
        similarity = cosine_similarity(vector, global_update)
        similarities.append(similarity)

    # Compute z-scores of the similarities
    mean_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    z_scores = [(sim - mean_similarity) / std_similarity for sim in similarities]

    # Identify malicious users based on z-score threshold
    malicious_indices = [i for i, z in enumerate(z_scores) if z < -z_score_threshold]
    malicious_user_ids = [update_indices[i] for i in malicious_indices]

    print("Updates: ", update_indices)
    print("[Range 1-10]: Malicious indices: ", malicious_indices)
    print("Malicious user IDs: ", malicious_user_ids)

    return malicious_user_ids


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





def fmnist_noniid_label_flip_attack(dataset, num_users, flip_mapping, flip_fraction):
    """
    Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack
    :param dataset: Dataset object containing the data
    :param num_users: Number of users (clients)
    :param flip_mapping: Dictionary mapping original labels to flipped labels
    :param flip_fraction: Fraction of users whose labels will be flipped
    :return: dict of image index
    """
    num_shards, num_imgs = 600, 100  # Increase the number of shards
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
        num_user_shards = np.random.randint(1, 6)  # Randomly assign 1 to 5 shards to each user
        rand_set = set(np.random.choice(idx_shard, num_user_shards, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        user_data = []
        for rand in rand_set:
            user_data.extend(idxs[rand * num_imgs:(rand + 1) * num_imgs])

        user_data = np.array(user_data)
        if i in flip_users:
            user_labels = labels[user_data]
            print(f"Before flipping: User {i}, Labels: {np.unique(user_labels, ' | Count: ', return_counts=True)}")
            flipped_labels = flip_labels(user_labels, flip_mapping)
            #print('flipped labels: ',flipped_labels)
            
            labels[user_data] = flipped_labels
            print(f"After flipping: User {i}, Labels: {np.unique(labels[user_data],' | Count: ', return_counts=True)}")

            # Verify that labels were flipped correctly
            for original_label, new_label in flip_mapping.items():
                original_count = np.sum(user_labels == original_label)
                new_count = np.sum(flipped_labels == new_label)
                print(f"Label {original_label} flipped to {new_label}: {original_count} -> {new_count}")
                if original_count > 0 and new_count == original_count:
                    print(f"Label {original_label} was flipped correctly to {new_label}")
                else:
                    print(f"Label {original_label} flipping to {new_label} failed")

        dict_users[i] = user_data

    # Update the dataset labels with the flipped labels
    dataset.targets = torch.tensor(labels)
    return dict_users


'''
The code given below called equal_all_label_in_all_case_dist_fmnist_noniid_label_flip_attack()
does equal distribution of all labels and may not be suitable for non-iid setting because it favors more of IID setting
'''
# def equal_all_label_in_all_case_dist_fmnist_noniid_label_flip_attack(dataset, num_users, flip_mapping, flip_fraction):
#     """
#     Sample non-I.I.D client data from FashionMNIST dataset with optional label flipping attack
#     :param dataset: Dataset object containing the data
#     :param num_users: Number of users (clients)
#     :param flip_mapping: Dictionary mapping original labels to flipped labels
#     :param flip_fraction: Fraction of users whose labels will be flipped
#     :return: dict of image index
#     """
#     num_shards, num_imgs = 600, 100  # Increase the number of shards
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards * num_imgs)
#     labels = dataset.targets.numpy()

#     # Sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]

#     # Select random clients for label flipping
#     num_flip_users = int(flip_fraction * num_users)
#     flip_users = np.random.choice(range(num_users), num_flip_users, replace=False)
#     print(f"Flipping labels for users: {flip_users}")

#     # Assign shards to users in a stratified manner
#     shards_per_user = num_shards // num_users
#     leftover_shards = num_shards % num_users

#     for user in range(num_users):
#         num_user_shards = shards_per_user + (1 if leftover_shards > 0 else 0)
#         leftover_shards -= 1

#         user_shards = np.random.choice(idx_shard, num_user_shards, replace=False)
#         idx_shard = list(set(idx_shard) - set(user_shards))

#         user_data = np.concatenate([idxs[shard * num_imgs:(shard + 1) * num_imgs] for shard in user_shards])
#         user_data = np.array(user_data)

#         if user in flip_users:
#             user_labels = labels[user_data]
#             print(f"Before flipping: User {user}, Labels: {np.unique(user_labels, return_counts=True)}")
#             flipped_labels = flip_labels(user_labels, flip_mapping)
#             labels[user_data] = flipped_labels
#             print(f"After flipping: User {user}, Labels: {np.unique(labels[user_data], return_counts=True)}")

#             # Verify that labels were flipped correctly
#             for original_label, new_label in flip_mapping.items():
#                 original_count = np.sum(user_labels == original_label)
#                 new_count = np.sum(flipped_labels == new_label)
#                 print(f"Label {original_label} flipped to {new_label}: {original_count} -> {new_count}")
#                 if original_count > 0 and new_count == original_count:
#                     print(f"Label {original_label} was flipped correctly to {new_label}")
#                 else:
#                     print(f"Label {original_label} flipping to {new_label} failed")

#         dict_users[user] = user_data

#     # Update the dataset labels with the flipped labels
#     dataset.targets = torch.tensor(labels)
#     return dict_users



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
        #label_distribution = dict(zip(unique_labels, counts))
        #print(f"User {user}: Label distribution {label_distribution}")
        print(f' User {user}: Label Distribution: {unique_labels} | counts: {counts}')


if __name__ == '__main__':
    ################################### hyperparameter setup ########################################
    args = call_parser()
    
    torch.manual_seed(args.seed + args.repeat)
    torch.cuda.manual_seed(args.seed + args.repeat)
    np.random.seed(args.seed + args.repeat)
    
    # args, dataset_train, dataset_test, dict_users = data_setup(args)
    args.num_classes = 10
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define the label flip mapping for all values only, swapping them in descending order , just 5 is swapped with 0
    #flip_mapping = {1: 9, 2: 8, 3: 7, 4: 6, 6: 4, 7: 3, 8: 2, 9: 1, 5:0, 0:5}
    
    flip_mapping = {0:5, 5:0}
    
    flip_fraction = 0.1  # Example: 10% of users will have their labels flipped
    path = './data/fmnist'
    
    trans_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.FashionMNIST(path, train=True, download=True, transform=trans_fmnist)
    dataset_test = datasets.FashionMNIST(path, train=False, download=True, transform=trans_fmnist)
    
    dict_users = fmnist_noniid_label_flip_attack(dataset_train, num_users=args.num_users, flip_mapping=flip_mapping, flip_fraction=flip_fraction)
    
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
            
            # clipping local model or not ? : no clip for cifar10
            # threshold = delta_norm / args.clip
            # if threshold > 1.0:
            #     for k in model_update.keys():
            #         model_update[k] = model_update[k] / threshold
            
            local_updates.append(model_update)
            loss_locals.append(loss)
        norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

        ##################### communication: avg for all groups #######################
        
        
        #threshold = 2.0
        #malicious_users = identify_malicious_users(local_updates, threshold)
        #print(f"Identified malicious users: {malicious_users} | total: {len(malicious_users)}")
        
        
        malicious_user_ids = identify_malicious_users(local_updates, selected_idxs, z_score_threshold=2.0)
        print(f"Identified malicious users: {malicious_user_ids} | total: {len(malicious_user_ids)}")
        
        
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




############## Label Flip Attack (40%) ################
######## 100 Users/ Clients ###########################

'''

(venv) jahanxb@alsharif-Lambda-Vector:/jahanxbProject/flcode$ python main_fed_noniid_label_flip_attack.py --dataset fmnist --round 10 --tau 10 --gpu 0 --num_users 100 --frac 1 --iid 0
Flipping labels for users: [83 30 56 24 16 23  2 27 28 13 99 92 76 14  0 21  3 29 61 79 35 11 84 44
 73  5 25 77 74 62 65  1 18 48 36 78  6 89 91 10]
Before flipping: User 0, Labels: (array([2, 4, 7]), array([  0, 200, 100]), array([100, 100, 100]))
After flipping: User 0, Labels: (array([3, 6, 8]), array([100, 200,   0]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 1, Labels: (array([8, 9]), array([100,   0]), array([100, 200]))
After flipping: User 1, Labels: (array([1, 2]), array([  0, 100]), array([200, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 200 -> 200
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 2, Labels: (array([0, 3, 4]), array([  0, 200, 100]), array([100, 100, 100]))
After flipping: User 2, Labels: (array([5, 6, 7]), array([  0, 100, 200]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 3, Labels: (array([2, 7]), array([  0, 200]), array([200, 100]))
After flipping: User 3, Labels: (array([3, 8]), array([200,   0]), array([100, 200]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 200 -> 200
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 5, Labels: (array([2, 5, 9]), array([  0, 200, 300]), array([300, 100, 100]))
After flipping: User 5, Labels: (array([0, 1, 8]), array([200, 300,   0]), array([100, 100, 300]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 300 -> 300
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 6, Labels: (array([5, 9]), array([  0, 100]), array([100, 100]))
After flipping: User 6, Labels: (array([0, 1]), array([  0, 100]), array([100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 10, Labels: (array([3, 6, 7]), array([  0, 200, 100]), array([100, 100, 100]))
After flipping: User 10, Labels: (array([3, 4, 7]), array([100, 200,   0]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 11, Labels: (array([0, 7]), array([  0, 100]), array([100, 100]))
After flipping: User 11, Labels: (array([3, 5]), array([100,   0]), array([100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 13, Labels: (array([3, 5, 9]), array([100,   0, 200]), array([100, 100, 100]))
After flipping: User 13, Labels: (array([0, 1, 7]), array([  0, 200, 100]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 14, Labels: (array([0, 5]), array([100,   0]), array([100, 100]))
After flipping: User 14, Labels: (array([0, 5]), array([  0, 100]), array([100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 16, Labels: (array([2, 3, 5]), array([  0, 200, 100]), array([100, 100, 200]))
After flipping: User 16, Labels: (array([0, 7, 8]), array([100, 200,   0]), array([200, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 200 -> 200
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 18, Labels: (array([4, 7, 8, 9]), array([  0, 200, 100, 300]), array([100, 100, 100, 100]))
After flipping: User 18, Labels: (array([1, 2, 3, 6]), array([300, 100, 200,   0]), array([100, 100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 21, Labels: (array([1, 2, 9]), array([200,   0, 100]), array([100, 100, 100]))
After flipping: User 21, Labels: (array([1, 8, 9]), array([100,   0, 200]), array([100, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 23, Labels: (array([3]), array([0]), array([100]))
After flipping: User 23, Labels: (array([7]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 24, Labels: (array([1, 5, 6]), array([  0, 100, 300]), array([100, 200, 100]))
After flipping: User 24, Labels: (array([0, 4, 9]), array([100, 300,   0]), array([200, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 200 -> 200
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 25, Labels: (array([4, 5, 9]), array([  0, 200, 100]), array([100, 100, 100]))
After flipping: User 25, Labels: (array([0, 1, 6]), array([200, 100,   0]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 27, Labels: (array([0, 5, 6]), array([100, 200,   0]), array([100, 100, 100]))
After flipping: User 27, Labels: (array([0, 4, 5]), array([200,   0, 100]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 28, Labels: (array([0, 9]), array([100,   0]), array([200, 100]))
After flipping: User 28, Labels: (array([1, 5]), array([  0, 100]), array([100, 200]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 200 -> 200
Label 0 was flipped correctly to 5
Before flipping: User 29, Labels: (array([6]), array([0]), array([100]))
After flipping: User 29, Labels: (array([4]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 30, Labels: (array([1]), array([0]), array([200]))
After flipping: User 30, Labels: (array([9]), array([0]), array([200]))
Label 1 flipped to 9: 200 -> 200
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 35, Labels: (array([0, 4, 7]), array([  0, 200, 100]), array([200, 100, 100]))
After flipping: User 35, Labels: (array([3, 5, 6]), array([100,   0, 200]), array([100, 200, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 200 -> 200
Label 0 was flipped correctly to 5
Before flipping: User 36, Labels: (array([0, 1, 3, 5, 6]), array([400, 200, 300, 100,   0]), array([100, 100, 100, 100, 100]))
After flipping: User 36, Labels: (array([0, 4, 5, 7, 9]), array([100,   0, 400, 300, 200]), array([100, 100, 100, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 44, Labels: (array([1, 4, 7, 9]), array([200, 100, 300,   0]), array([100, 100, 200, 100]))
After flipping: User 44, Labels: (array([1, 3, 6, 9]), array([  0, 300, 100, 200]), array([100, 200, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 200 -> 200
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 48, Labels: (array([6]), array([0]), array([300]))
After flipping: User 48, Labels: (array([4]), array([0]), array([300]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 300 -> 300
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 56, Labels: (array([6]), array([0]), array([100]))
After flipping: User 56, Labels: (array([4]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 61, Labels: (array([5]), array([0]), array([100]))
After flipping: User 61, Labels: (array([0]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 62, Labels: (array([3]), array([0]), array([100]))
After flipping: User 62, Labels: (array([7]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 65, Labels: (array([4, 7, 8, 9]), array([200, 100,   0, 300]), array([100, 100, 200, 100]))
After flipping: User 65, Labels: (array([1, 2, 3, 6]), array([300,   0, 100, 200]), array([100, 200, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 200 -> 200
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 73, Labels: (array([2, 6, 8]), array([  0, 100, 200]), array([100, 100, 100]))
After flipping: User 73, Labels: (array([2, 4, 8]), array([200, 100,   0]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 74, Labels: (array([0, 5, 6, 8]), array([300, 100, 400,   0]), array([100, 200, 100, 100]))
After flipping: User 74, Labels: (array([0, 2, 4, 5]), array([100,   0, 400, 300]), array([200, 100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 200 -> 200
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 76, Labels: (array([0, 3, 7]), array([100,   0, 200]), array([100, 200, 100]))
After flipping: User 76, Labels: (array([3, 5, 7]), array([200, 100,   0]), array([100, 100, 200]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 200 -> 200
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 77, Labels: (array([2, 9]), array([  0, 100]), array([100, 100]))
After flipping: User 77, Labels: (array([1, 8]), array([100,   0]), array([100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 78, Labels: (array([3, 4, 6, 9]), array([  0, 200, 300, 400]), array([200, 100, 100, 100]))
After flipping: User 78, Labels: (array([1, 4, 6, 7]), array([400, 300, 200,   0]), array([100, 100, 100, 200]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 200 -> 200
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 79, Labels: (array([0]), array([0]), array([100]))
After flipping: User 79, Labels: (array([5]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 83, Labels: (array([3, 5, 7]), array([200,   0, 100]), array([100, 100, 100]))
After flipping: User 83, Labels: (array([0, 3, 7]), array([  0, 100, 200]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 84, Labels: (array([0, 1, 7, 8]), array([200, 100,   0, 300]), array([100, 100, 200, 100]))
After flipping: User 84, Labels: (array([2, 3, 5, 9]), array([300,   0, 200, 100]), array([100, 200, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 200 -> 200
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 100 -> 100
Label 0 was flipped correctly to 5
Before flipping: User 89, Labels: (array([3, 4, 6]), array([100,   0, 200]), array([100, 100, 100]))
After flipping: User 89, Labels: (array([4, 6, 7]), array([200,   0, 100]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 100 -> 100
Label 3 was flipped correctly to 7
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 91, Labels: (array([1, 4, 6, 9]), array([  0, 400, 100, 200]), array([200, 100, 100, 100]))
After flipping: User 91, Labels: (array([1, 4, 6, 9]), array([200, 100, 400,   0]), array([100, 100, 100, 200]))
Label 1 flipped to 9: 200 -> 200
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 100 -> 100
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 100 -> 100
Label 9 was flipped correctly to 1
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 92, Labels: (array([6, 7, 8]), array([200, 100,   0]), array([100, 100, 100]))
After flipping: User 92, Labels: (array([2, 3, 4]), array([  0, 100, 200]), array([100, 100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 99, Labels: (array([6]), array([0]), array([100]))
After flipping: User 99, Labels: (array([4]), array([0]), array([100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
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
 User 0: Label Distribution: [3 6 8] | counts: [100 100 100]
 User 1: Label Distribution: [1 2] | counts: [200 100]
 User 2: Label Distribution: [5 6 7] | counts: [100 100 100]
 User 3: Label Distribution: [3 8] | counts: [100 200]
 User 4: Label Distribution: [2 3 4 6 9] | counts: [100 100 100 100 100]
 User 5: Label Distribution: [0 1 8] | counts: [100 100 300]
 User 6: Label Distribution: [0 1] | counts: [100 100]
 User 7: Label Distribution: [5] | counts: [100]
 User 8: Label Distribution: [0 1 3 4 7] | counts: [100 100 100 100 100]
 User 9: Label Distribution: [5 8] | counts: [100 100]
 User 10: Label Distribution: [3 4 7] | counts: [100 100 100]
 User 11: Label Distribution: [3 5] | counts: [100 100]
 User 12: Label Distribution: [1 6 8 9] | counts: [200 100 100 100]
 User 13: Label Distribution: [0 1 7] | counts: [100 100 100]
 User 14: Label Distribution: [0 5] | counts: [100 100]
 User 15: Label Distribution: [0 3 5 9] | counts: [100 100 200 100]
 User 16: Label Distribution: [0 7 8] | counts: [200 100 100]
 User 17: Label Distribution: [5 7] | counts: [100 200]
 User 18: Label Distribution: [1 2 3 6] | counts: [100 100 100 100]
 User 19: Label Distribution: [3 4 8] | counts: [100 100 100]
 User 20: Label Distribution: [8] | counts: [100]
 User 21: Label Distribution: [1 8 9] | counts: [100 100 100]
 User 22: Label Distribution: [1 4 6 9] | counts: [100 100 100 200]
 User 23: Label Distribution: [7] | counts: [100]
 User 24: Label Distribution: [0 4 9] | counts: [200 100 100]
 User 25: Label Distribution: [0 1 6] | counts: [100 100 100]
 User 26: Label Distribution: [3 5 9] | counts: [200 100 200]
 User 27: Label Distribution: [0 4 5] | counts: [100 100 100]
 User 28: Label Distribution: [1 5] | counts: [100 200]
 User 29: Label Distribution: [4] | counts: [100]
 User 30: Label Distribution: [9] | counts: [200]
 User 31: Label Distribution: [5 7] | counts: [100 100]
 User 32: Label Distribution: [1 8] | counts: [100 100]
 User 33: Label Distribution: [1 3 9] | counts: [100 200 100]
 User 34: Label Distribution: [6 7] | counts: [100 100]
 User 35: Label Distribution: [3 5 6] | counts: [100 200 100]
 User 36: Label Distribution: [0 4 5 7 9] | counts: [100 100 100 100 100]
 User 37: Label Distribution: [1 3 7 8] | counts: [200 100 100 100]
 User 38: Label Distribution: [9] | counts: [100]
 User 39: Label Distribution: [0 3 4 9] | counts: [200 100 100 100]
 User 40: Label Distribution: [5] | counts: [100]
 User 41: Label Distribution: [6 8] | counts: [100 100]
 User 42: Label Distribution: [2 6] | counts: [200 100]
 User 43: Label Distribution: [1 2 9] | counts: [300 100 100]
 User 44: Label Distribution: [1 3 6 9] | counts: [100 200 100 100]
 User 45: Label Distribution: [2 6] | counts: [100 200]
 User 46: Label Distribution: [6 8 9] | counts: [100 100 100]
 User 47: Label Distribution: [0 1 3] | counts: [100 200 100]
 User 48: Label Distribution: [4] | counts: [300]
 User 49: Label Distribution: [8 9] | counts: [200 100]
 User 50: Label Distribution: [0 8] | counts: [100 100]
 User 51: Label Distribution: [0 2 3 4 9] | counts: [100 100 100 100 100]
 User 52: Label Distribution: [0 4 9] | counts: [100 100 100]
 User 53: Label Distribution: [5 7 9] | counts: [100 200 100]
 User 54: Label Distribution: [2 4 5 8] | counts: [100 100 100 100]
 User 55: Label Distribution: [0 7 9] | counts: [100 100 100]
 User 56: Label Distribution: [4] | counts: [100]
 User 57: Label Distribution: [0 1 2 4 8] | counts: [100 100 100 100 100]
 User 58: Label Distribution: [0 8 9] | counts: [100 200 100]
 User 59: Label Distribution: [2 9] | counts: [100 100]
 User 60: Label Distribution: [3 6 9] | counts: [200 100 100]
 User 61: Label Distribution: [0] | counts: [100]
 User 62: Label Distribution: [7] | counts: [100]
 User 63: Label Distribution: [3 5 6 7] | counts: [100 100 100 100]
 User 64: Label Distribution: [0 1 2 6 7] | counts: [100 100 100 100 100]
 User 65: Label Distribution: [1 2 3 6] | counts: [100 200 100 100]
 User 66: Label Distribution: [2] | counts: [100]
 User 67: Label Distribution: [1 6] | counts: [100 100]
 User 68: Label Distribution: [2 3 7] | counts: [100 100 100]
 User 69: Label Distribution: [0 2] | counts: [100 100]
 User 70: Label Distribution: [1 4 8 9] | counts: [100 200 100 100]
 User 71: Label Distribution: [4] | counts: [100]
 User 72: Label Distribution: [4 5] | counts: [100 100]
 User 73: Label Distribution: [2 4 8] | counts: [100 100 100]
 User 74: Label Distribution: [0 2 4 5] | counts: [200 100 100 100]
 User 75: Label Distribution: [3] | counts: [100]
 User 76: Label Distribution: [3 5 7] | counts: [100 100 200]
 User 77: Label Distribution: [1 8] | counts: [100 100]
 User 78: Label Distribution: [1 4 6 7] | counts: [100 100 100 200]
 User 79: Label Distribution: [5] | counts: [100]
 User 80: Label Distribution: [0 4] | counts: [200 100]
 User 81: Label Distribution: [1] | counts: [100]
 User 82: Label Distribution: [2 9] | counts: [100 100]
 User 83: Label Distribution: [0 3 7] | counts: [100 100 100]
 User 84: Label Distribution: [2 3 5 9] | counts: [100 200 100 100]
 User 85: Label Distribution: [0 2 8 9] | counts: [100 200 100 100]
 User 86: Label Distribution: [1 4 7] | counts: [100 100 200]
 User 87: Label Distribution: [2 4 5 7] | counts: [100 100 100 100]
 User 88: Label Distribution: [5] | counts: [100]
 User 89: Label Distribution: [4 6 7] | counts: [100 100 100]
 User 90: Label Distribution: [4] | counts: [200]
 User 91: Label Distribution: [1 4 6 9] | counts: [100 100 100 200]
 User 92: Label Distribution: [2 3 4] | counts: [100 100 100]
 User 93: Label Distribution: [2] | counts: [100]
 User 94: Label Distribution: [2] | counts: [100]
 User 95: Label Distribution: [2] | counts: [100]
 User 96: Label Distribution: [3 4 5 9] | counts: [100 100 200 100]
 User 97: Label Distribution: [9] | counts: [100]
 User 98: Label Distribution: [0 2 7 8] | counts: [100 200 100 100]
 User 99: Label Distribution: [4] | counts: [100]
=============== training... ================================
t   0: train_loss = 0.679, norm = 1.996, test_acc = 33.800
t   1: train_loss = 0.416, norm = 1.623, test_acc = 51.622
t   2: train_loss = 0.279, norm = 1.464, test_acc = 62.744
t   3: train_loss = 0.255, norm = 1.445, test_acc = 53.844
t   4: train_loss = 0.232, norm = 1.417, test_acc = 63.156
t   5: train_loss = 0.229, norm = 1.359, test_acc = 54.411
t   6: train_loss = 0.216, norm = 1.356, test_acc = 59.678
t   7: train_loss = 0.217, norm = 1.299, test_acc = 55.611
t   8: train_loss = 0.200, norm = 1.248, test_acc = 52.011
t   9: train_loss = 0.195, norm = 1.277, test_acc = 51.722
training time: 00:05:51.50
'''


############## Label Flip Attack (40%) ################
######## 100 Users/ Clients ###########################

'''
(venv) jahanxb@alsharif-Lambda-Vector:/jahanxbProject/flcode$ python main_fed_noniid_label_flip_attack.py --dataset fmnist --round 10 --tau 10 --gpu 0 --num_users 10 --frac 1 --iid 0
Flipping labels for users: [4 1 5 0]
Before flipping: User 0, Labels: (array([1, 6, 7, 8]), array([300, 100,   0, 200]), array([100, 100, 100, 100]))
After flipping: User 0, Labels: (array([2, 3, 4, 9]), array([200,   0, 100, 300]), array([100, 100, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 1, Labels: (array([4, 7, 8]), array([100, 200,   0]), array([200, 100, 100]))
After flipping: User 1, Labels: (array([2, 3, 6]), array([  0, 200, 100]), array([100, 100, 200]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 0 -> 0
Label 2 flipping to 8 failed
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 200 -> 200
Label 4 was flipped correctly to 6
Label 6 flipped to 4: 0 -> 0
Label 6 flipping to 4 failed
Label 7 flipped to 3: 100 -> 100
Label 7 was flipped correctly to 3
Label 8 flipped to 2: 100 -> 100
Label 8 was flipped correctly to 2
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 4, Labels: (array([1, 2, 5, 6]), array([  0, 300, 200, 100]), array([100, 100, 100, 100]))
After flipping: User 4, Labels: (array([0, 4, 8, 9]), array([200, 100, 300,   0]), array([100, 100, 100, 100]))
Label 1 flipped to 9: 100 -> 100
Label 1 was flipped correctly to 9
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 100 -> 100
Label 5 was flipped correctly to 0
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
Before flipping: User 5, Labels: (array([2, 6]), array([  0, 100]), array([100, 100]))
After flipping: User 5, Labels: (array([4, 8]), array([100,   0]), array([100, 100]))
Label 1 flipped to 9: 0 -> 0
Label 1 flipping to 9 failed
Label 2 flipped to 8: 100 -> 100
Label 2 was flipped correctly to 8
Label 3 flipped to 7: 0 -> 0
Label 3 flipping to 7 failed
Label 4 flipped to 6: 0 -> 0
Label 4 flipping to 6 failed
Label 6 flipped to 4: 100 -> 100
Label 6 was flipped correctly to 4
Label 7 flipped to 3: 0 -> 0
Label 7 flipping to 3 failed
Label 8 flipped to 2: 0 -> 0
Label 8 flipping to 2 failed
Label 9 flipped to 1: 0 -> 0
Label 9 flipping to 1 failed
Label 5 flipped to 0: 0 -> 0
Label 5 flipping to 0 failed
Label 0 flipped to 5: 0 -> 0
Label 0 flipping to 5 failed
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
 User 0: Label Distribution: [2 3 4 9] | counts: [100 100 100 100]
 User 1: Label Distribution: [2 3 6] | counts: [100 100 200]
 User 2: Label Distribution: [2 6] | counts: [200 100]
 User 3: Label Distribution: [8 9] | counts: [100 100]
 User 4: Label Distribution: [0 4 8 9] | counts: [100 100 100 100]
 User 5: Label Distribution: [4 8] | counts: [100 100]
 User 6: Label Distribution: [2 5 6] | counts: [100 100 100]
 User 7: Label Distribution: [3] | counts: [100]
 User 8: Label Distribution: [0 5 6] | counts: [100 200 100]
 User 9: Label Distribution: [8 9] | counts: [100 100]
=============== training... ================================
t   0: train_loss = 0.930, norm = 2.012, test_acc = 14.289
t   1: train_loss = 0.480, norm = 1.362, test_acc = 15.811
t   2: train_loss = 0.356, norm = 1.311, test_acc = 27.300
t   3: train_loss = 0.331, norm = 1.320, test_acc = 18.211
t   4: train_loss = 0.361, norm = 1.404, test_acc = 23.400
t   5: train_loss = 0.337, norm = 1.464, test_acc = 25.811
t   6: train_loss = 0.311, norm = 1.175, test_acc = 23.989
t   7: train_loss = 0.257, norm = 1.193, test_acc = 33.667
t   8: train_loss = 0.248, norm = 1.082, test_acc = 29.411
t   9: train_loss = 0.222, norm = 1.130, test_acc = 36.600
training time: 00:00:41.41

'''