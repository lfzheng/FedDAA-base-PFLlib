# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
import torch
import math
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.data_utils import read_client_data
from utils.ALA import ALA
from flcore.optimizers.fedoptimizer import PerAvgOptimizer

class clientDAA(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.eta = args.eta
        self.rand_percent = args.rand_percent
        self.layer_idx = args.layer_idx
        self.lamda = 1
        self.mu = args.mu
        self.L = int(args.L * args.global_rounds)
        self.drlr = args.dr_learning_rate
        self.beta = self.learning_rate
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.optimizer = PerAvgOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.ALA = ALA(self.id, self.loss, train_data, self.batch_size, 
                    self.rand_percent, self.layer_idx, self.eta, self.device)
        self.adaptive_weight = 0
        self.size_weight = 0
        self.weight = 0
        self.alpha = 0
        self.alpha_list = []
    def train(self, R):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                cnt = 0
                p_grad = 0
                ham_1weight = self.hadamard_product_wight(self.ALA.weights, self.layer_idx, R)
                for param, weight in zip(self.model.parameters(), ham_1weight):
                    p_grad += torch.mean(param.grad * param * weight).item() * (self.size_weight-self.adaptive_weight)
                    cnt += 1
                p_grad = p_grad / cnt
                self.alpha = max(0, min(self.alpha - self.drlr * p_grad, 1))

                #step 2
                if type(x) == type([]):
                    x = [None, None]
                    x[0] = x[0].to(self.device)
                    
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step(beta=self.beta)
        self.alpha_list.append(self.alpha)
        if R < self.L:
            self.lamda = (math.cos(R * math.pi / self.L) + 1) / 2
        else:
            self.lamda = 0
        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        

    def local_initialization(self, received_global_model):
        self.ALA.adaptive_local_aggregation(received_global_model, self.model)