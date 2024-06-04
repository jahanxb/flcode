import torch
import numpy as np

def aggregation_avg(global_model, local_updates):
    '''
    simple average
    '''
    model_update = {k: local_updates[0][k] *0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] +  local_updates[i][k] for k in global_model.keys()}
    global_model = {k: global_model[k] +  model_update[k]/ len(local_updates) for k in global_model.keys()}
    return global_model



def aggregation_fedavgm(global_model, local_updates, momentum, velocity):
    """
    Federated Averaging with Momentum (FedAvgM)
    """
    model_update = {k: local_updates[0][k] * 0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] + local_updates[i][k] for k in global_model.keys()}
    
    # Compute the average update
    average_update = {k: model_update[k] / len(local_updates) for k in global_model.keys()}
    
    # Update the velocity
    velocity = {k: momentum * velocity[k] + average_update[k] for k in global_model.keys()}
    
    # Apply the velocity to the global model
    global_model = {k: global_model[k] + velocity[k] for k in global_model.keys()}
    
    return global_model, velocity



def aggregation_weighted(global_model, local_updates, weights):
    """
    Weighted Aggregation
    """
    model_update = {k: local_updates[0][k] * 0.0 for k in local_updates[0].keys()}
    for i in range(len(local_updates)):
        model_update = {k: model_update[k] + weights[i] * local_updates[i][k] for k in global_model.keys()}
    
    # Apply the weighted update to the global model
    global_model = {k: global_model[k] + model_update[k] for k in global_model.keys()}
    
    return global_model



def aggregation_fltrust(global_model, local_updates, trusted_update, trust_factors):
    """
    FL-Trust Aggregation
    """
    model_update = {k: torch.zeros_like(v) for k, v in global_model.items()}
    for i in range(len(local_updates)):
        # Calculate trust factor for each client
        trust_factor = trust_factors[i]
        model_update = {k: model_update[k] + trust_factor * local_updates[i][k] for k in global_model.keys()}
    
    global_model = {k: global_model[k] + model_update[k] for k in global_model.keys()}
    return global_model



def aggregation_krum(global_model, local_updates):
    """
    Krum Aggregation
    """
    num_selected_users = len(local_updates)
    distances = np.zeros((num_selected_users, num_selected_users))
    
    for i in range(num_selected_users):
        for j in range(i + 1, num_selected_users):
            distances[i][j] = torch.norm(torch.cat([torch.flatten(local_updates[i][k] - local_updates[j][k]) for k in global_model.keys()])).item()
            distances[j][i] = distances[i][j]
    
    scores = np.zeros(num_selected_users)
    for i in range(num_selected_users):
        sorted_distances = np.sort(distances[i])
        scores[i] = np.sum(sorted_distances[:num_selected_users - 2])
    
    krum_index = np.argmin(scores)
    selected_update = local_updates[krum_index]
    
    global_model = {k: global_model[k] + selected_update[k] for k in global_model.keys()}
    return global_model




def aggregation_median(global_model, local_updates):
    """
    Median Aggregation
    """
    model_update = {k: torch.stack([local_updates[i][k] for i in range(len(local_updates))]).median(dim=0)[0] for k in global_model.keys()}
    global_model = {k: global_model[k] + model_update[k] for k in global_model.keys()}
    return global_model


def aggregation_trimmed_mean(global_model, local_updates, trim_fraction=0.1):
    """
    Trimmed Mean Aggregation
    """
    num_updates = len(local_updates)
    trim_count = int(num_updates * trim_fraction)
    
    model_update = {}
    for k in global_model.keys():
        updates = torch.stack([local_updates[i][k] for i in range(num_updates)])
        sorted_updates, _ = updates.sort(dim=0)
        trimmed_updates = sorted_updates[trim_count:-trim_count]
        model_update[k] = trimmed_updates.mean(dim=0)
    
    global_model = {k: global_model[k] + model_update[k] for k in global_model.keys()}
    return global_model



def aggregation_bulyan(global_model, local_updates, num_to_select):
    """
    Bulyan Aggregation
    """
    def krum_candidate_selection(local_updates, num_to_select):
        num_updates = len(local_updates)
        distances = np.zeros((num_updates, num_updates))
        
        for i in range(num_updates):
            for j in range(i + 1, num_updates):
                distances[i][j] = torch.norm(torch.cat([torch.flatten(local_updates[i][k] - local_updates[j][k]) for k in global_model.keys()])).item()
                distances[j][i] = distances[i][j]
        
        scores = np.zeros(num_updates)
        for i in range(num_updates):
            sorted_distances = np.sort(distances[i])
            scores[i] = np.sum(sorted_distances[:num_to_select])
        
        selected_indices = np.argsort(scores)[:num_to_select]
        return selected_indices
    
    num_updates = len(local_updates)
    if num_to_select > num_updates:
        raise ValueError("num_to_select cannot be greater than the number of local updates")

    candidates = krum_candidate_selection(local_updates, num_to_select)
    if len(candidates) == 0:
        raise RuntimeError("No candidates selected for Bulyan aggregation")
    
    candidate_updates = [local_updates[i] for i in candidates]
    
    trimmed_model_update = {}
    for k in global_model.keys():
        updates = torch.stack([candidate_updates[i][k] for i in range(len(candidate_updates))])
        sorted_updates, _ = updates.sort(dim=0)
        if len(sorted_updates) <= 2:
            raise RuntimeError("Not enough updates for trimming")
        trimmed_updates = sorted_updates[1:-1]  # Trim the smallest and largest
        trimmed_model_update[k] = trimmed_updates.mean(dim=0)
    
    global_model = {k: global_model[k] + trimmed_model_update[k] for k in global_model.keys()}
    return global_model