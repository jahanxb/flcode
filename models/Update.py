#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torch
from torch import nn
import copy


class LocalUpdate(object):
    def __init__(self, args):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        
    def sgd(self, net, images, labels):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        net.zero_grad()
        log_probs = net(images)
        loss = self.loss_func(log_probs, labels)
        loss.backward()
        
        # update
        optimizer.step()
        w_new = copy.deepcopy(net.state_dict())
        return w_new, loss.item()

    def local_sgd(self, net, ldr_train):
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.local_lr, momentum=self.args.local_momentum)
        epoch_loss = []
        net.train()
        # ldr_train = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        for s in range(self.args.tau):
            for batch_idx, (images, labels) in enumerate(ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # print(s)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                
                optimizer.step()
                epoch_loss.append(loss.item())
        w_new = copy.deepcopy(net.state_dict())
        return w_new, sum(epoch_loss) / len(epoch_loss)

