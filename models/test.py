#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6


import torch.nn.functional as F
from torch.utils.data import DataLoader
import copy


def test_img(net_g, datatest, args):
    net_g = copy.deepcopy(net_g).to(args.device)
    #datatest = datatest.to(args.device)
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.batch_size)
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy, test_loss

