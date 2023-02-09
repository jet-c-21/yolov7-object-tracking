# coding: utf-8
"""
Author: Jet C.
GitHub: https://github.com/jet-c-21
Create Date: 2023-02-10
"""
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

X_train = torch.FloatTensor([0., 1., 2.])
print(X_train.is_cuda)

X_train = X_train.to(device)
print(X_train.is_cuda)
