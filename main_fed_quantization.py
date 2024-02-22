import copy
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
import tenseal as ts

from tqdm import tqdm

# from utils.rdp_accountant import compute_rdp, get_privacy_spent
import warnings
warnings.filterwarnings("ignore")
torch.cuda.is_available()


context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60],
                #n_threads=1,
            )
context.generate_galois_keys()
context.global_scale = 2**40


# Example aggregation (simplified for demonstration)
def aggregate_quantized_updates(local_updates):
    """Aggregate quantized updates from local models."""
    aggregated_update = copy.deepcopy(local_updates[0])
    for k in aggregated_update.keys():
        for i in range(1, len(local_updates)):
            aggregated_update[k] += local_updates[i][k]
        aggregated_update[k] = aggregated_update[k] / len(local_updates)
    return aggregated_update



if __name__ == '__main__':
    ################################### hyperparameter setup ########################################
    args = call_parser()
    
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
    ####################################### run experiment ##########################
    
    
    
    #print("global Model: empty state I guesss: ",global_model.get('fc2.bias'))
    
    
    # import torch.quantization
    # quantized_model = torch.quantization.quantize_dynamic(global_model, {torch.nn.Linear}, dtype=torch.qint8)
    
    # print('Quantized ModeL: ',quantized_model)
    
    
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
            
            
            
            #print('model_update: ',model_update.get('fc2.bias'))
            
            
            
            local_updates.append(model_update)
            loss_locals.append(loss)
        norm_med.append(torch.median(torch.stack(delta_norms)).cpu())

        #print('local_update : ',local_updates)
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
        
        net_glob.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        # net_glob_prepared = torch.quantization.prepare(net_glob, inplace=False)
        
        # with torch.no_grad():
        #     for data, _ in ldr_train:  # Use a part of the training data for calibration
        #         net_glob_prepared(data)  # Forward pass for calibration
        #         break

        # net_glob_quantized = torch.quantization.convert(net_glob_prepared, inplace=False)
        
        model_prepared = torch.quantization.prepare(net_glob, inplace=False)
        
        for inputs, _ in ldr_train:
            model_prepared(inputs)
        
        net_glob_quantized = torch.quantization.convert(model_prepared,inplace=False)        
        
        
        '''
        Now do Homomorphic Encryption on it 
        '''
        
        # start_time_he = time.time()

        # # Encrypt the model
        # state_dict = net_glob_quantized.state_dict()
        # for name, tensor in tqdm(state_dict.items()):
        #     try:
        #         state_dict[name] = ts.ckks_tensor(context, tensor)
        #     except:
        #         # Skip non-tensor entries like num_batches_tracked
        #         print(tensor)

        # # Print time to encrypt the model
        # print("Time to encrypt the model: {}s".format(time.time() - start_time_he))
        
        
        '''
        End of Homomorphic Encryption Module
        '''
        
        test_acc_, _ = test_img(net_glob_quantized, dataset_test, args)
        test_acc.append(test_acc_)
        
        train_local_loss.append(sum(loss_locals) / len(loss_locals))
        # print('t {:3d}: '.format(t, ))
        print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
                format(t, train_local_loss[-1], norm_med[-1], test_acc[-1]))
        #print(t,train_local_loss,test_acc)
        
        #print('t {:3d}: train_loss = {:.3f}, norm = {:.3f}, test_acc = {:.3f}'.
        #        format(t, train_local_loss, norm_med, test_acc))

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
    hours, rem = divmod(t2-t1, 3600)
    minutes, seconds = divmod(rem, 60)
    print("training time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    
    
    #print('new Global model: ',global_model.get('fc2.bias'))
    
    
    # Assume local_updates is a list of state_dicts from quantized local models
    #global_model.load_state_dict(aggregate_quantized_updates(local_updates))
    
    #net_glob_quantized.eval()
    
    
        # Measure time to encrypt the model
    # start_time = time.time()

    # # Encrypt the model
    # state_dict = net_glob_quantized.state_dict()
    # for name, tensor in tqdm(state_dict.items()):
    #     try:
    #         state_dict[name] = ts.ckks_tensor(context, tensor)
    #     except:
    #         # Skip non-tensor entries like num_batches_tracked
    #         print(tensor)

    # # Print time to encrypt the model
    # print("Time to encrypt the model: {}s".format(time.time() - start_time))