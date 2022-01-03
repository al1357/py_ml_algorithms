# -*- coding: utf-8 -*-
from data_iris import read_data
import neural_network as nn

# 1) Get data
(x, y, data) = read_data()

# m x n > n x m
x = x.T
y = y.T

# 2) Train
network_structure = [10, 5, 3]
# train, cv, test sets
sets_distribution = [0.6, 0.2, 0.2]
nn1 = nn.neural_network(network_structure, \
                        x, \
                        y, \
                        sets_distribution, \
                        load_parameters=False)

nn1.learn(gradient_check=False)
#local_nn1 = nn1.__dict__
