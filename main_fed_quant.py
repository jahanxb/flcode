# Import necessary packages
import copy
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.quantization

# Assuming other necessary imports and setup functions are correctly defined above this point

if __name__ == '__main__':
    args = call_parser()  # Parse arguments
    
    # Seed setup
    torch.manual_seed(args.seed + args.repeat)
    torch.cuda.manual_seed_all(args.seed + args.repeat)
    np.random.seed(args.seed + args.repeat)
    
    # Data setup
    args, dataset_train, dataset_test, dict_users = data_setup(args)
    # Data loader setup
    trainloader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    testloader = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)
    
    # Model setup
    net_glob = model_setup(args)
    net_glob.train()

    # Initialize for training
    global_model = copy.deepcopy(net_glob.state_dict())
    train_local_loss, test_acc, norm_med = [], [], []

    # Main training loop
    for round in range(args.rounds):
        local_updates, loss_locals, delta_norms = [], [], []
        selected_users = np.random.choice(range(args.num_users), int(args.num_users * args.frac), replace=False)
        
        for user in selected_users:
            local_model = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[user])
            w, loss = local_model.update_weights(model=copy.deepcopy(net_glob), global_round=round)
            local_updates.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # Aggregate updates
        global_weights = aggregate_quantized_updates(local_updates)

        # Load aggregated weights
        net_glob.load_state_dict(global_weights)

        # Prepare and quantize global model
        net_glob.eval()
        net_glob.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        net_glob_prepared = torch.quantization.prepare(net_glob, inplace=False)
        with torch.no_grad():
            for data, _ in DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True):
                net_glob_prepared(data)
                break
        net_glob_quantized = torch.quantization.convert(net_glob_prepared, inplace=False)

        # Optionally: Evaluate quantized model performance
        # test_acc.append(test_model(net_glob_quantized, testloader))

        print(f'Round {round+1}, Avg Loss: {np.mean(loss_locals)}, Test Acc: {test_acc[-1] if test_acc else "Not Tested"}')

    # Final evaluation
    test_acc = test_img(net_glob_quantized, testloader, args)
    print(f'Final Test Accuracy: {test_acc}')
