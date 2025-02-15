import torch
import numpy as np
import argparse

from einops import rearrange

from model_snn_withoutcupy import SNASNet
import os
import gc
from scipy.linalg import fractional_matrix_power
import torch.nn as nn
import torch.nn.functional as F


def logdet(K):
    s, ld = np.linalg.slogdet(K)
    return ld

def snn_score(args, trainset, con_mat):
    search_batchsize = 512
    repeat = 2

    train_data = torch.utils.data.DataLoader(trainset, batch_size=search_batchsize,
                                             shuffle=True, pin_memory=True, num_workers=0)
    neuron_type = 'LIFNode'

    with torch.no_grad():

        searchnet = SNASNet(args, con_mat)
        searchnet.to(args.device)

        searchnet.K = np.zeros((search_batchsize, search_batchsize))
        searchnet.num_actfun = 0

        def computing_K_eachtime(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            out = out.view(out.size(0), -1)
            batch_num, neuron_num = out.size()
            x = (out > 0).float()

            # print("1:{}".format(torch.cuda.memory_allocated(0)))
            full_matrix = torch.ones((search_batchsize, search_batchsize)).to(args.device) * neuron_num
            # print("2:{}".format(torch.cuda.memory_allocated(0)))
            sparsity = (x.sum(1) / neuron_num).unsqueeze(1)
            # print("3:{}".format(torch.cuda.memory_allocated(0)))
            norm_K = (sparsity @ (1 - sparsity.t())) + ((1 - sparsity) @ sparsity.t())
            # print("4:{}".format(torch.cuda.memory_allocated(0)))
            rescale_factor = torch.div(0.5 * torch.ones((search_batchsize, search_batchsize)).to(args.device),
                                       norm_K + 1e-3)
            # print("5:{}".format(torch.cuda.memory_allocated(0)))
            K1_0 = (x @ (1 - x.t()))
            K0_1 = ((1 - x) @ x.t())
            # print("6:{}".format(torch.cuda.memory_allocated(0)))
            K_total = (full_matrix - rescale_factor * (K0_1 + K1_0))
            # print("7:{}".format(torch.cuda.memory_allocated(0)))

            searchnet.K = searchnet.K + (K_total.cpu().numpy())
            searchnet.num_actfun += 1

        s = []
        for name, module in searchnet.named_modules():
            if neuron_type in str(type(module)):
                module.register_forward_hook(computing_K_eachtime)

        for j in range(repeat):
            searchnet.K = np.zeros((search_batchsize, search_batchsize))
            searchnet.num_actfun = 0
            data_iterator = iter(train_data)
            inputs, targets = next(data_iterator)
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = searchnet(inputs)
            s.append(logdet(searchnet.K / (searchnet.num_actfun)))
        scores = np.mean(s)
        # print("final score:", scores)

    del searchnet, inputs, targets, outputs, s
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def get_lap_mat(cnt_mat):
    adjacent_mat = np.zeros((4, 4))
    degree_mat = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            if cnt_mat[i, j] > 0:
                adjacent_mat[i, j] = 1
    for k in range(4):
        degree = (cnt_mat[k, :] > 0).sum() - (cnt_mat[:, k] > 0).sum()
        degree_mat[k, k] = degree
    laplace_mat = degree_mat - adjacent_mat
    degree_inverse = fractional_matrix_power(degree_mat, -0.5)
    normed_laplace = degree_inverse @ laplace_mat @ degree_inverse

    return laplace_mat

def get_opt_mat(cnt_mat):
    position = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    operator_vector = np.zeros((1, 6)).squeeze()
    for num, index in enumerate(position):
        operator_vector[num] = cnt_mat[index]

    return operator_vector

class Mlp_embedding(nn.Module):
    def __init__(self):
        super(Mlp_embedding, self).__init__()
        self.embedding = nn.Embedding(5, 80)
        self.transform = nn.Linear(16, 6 * 80)
        self.linear1 = nn.Linear(480, 200)
        self.linear2 = nn.Linear(200, 16)
        self.linear3 = nn.Linear(16, 1)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(16)

    def forward(self, laplace, operator):
        x1 = self.embedding(operator)
        x2 = self.transform(laplace.view(-1, 16))

        x = (x1 + x2.view(-1, 6, 80)).view(-1, 480)
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = torch.sigmoid(self.linear3(x))

        return x

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def max_min(x):
    max = torch.max(x)
    min = torch.min(x)
    return (x - min)/(max - min)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer_embedding(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool='mean',
                 dim_head=32, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.embedding = nn.Embedding(5, 80)
        self.transform = nn.Linear(16, 6 * 80)
        self.linear1 = nn.Linear(480, 96)
        self.linear2 = nn.Linear(200, 16)
        self.linear3 = nn.Linear(96, 1)
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(16)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(480),
            nn.Linear(480, 96),
            nn.ReLU(),
            nn.LayerNorm(96),
            nn.Linear(96, 1),
            nn.Sigmoid()
        )

    def forward(self, laplace, operator):
        x1 = self.embedding(operator)
        x2 = self.transform(laplace.view(-1, 16))
        x = x1 + x2.view(-1, 6, 80)
        x = self.dropout(x)
        x = self.transformer(x).view(-1, 480)
        # x = self.mlp_head(x)
        x = F.relu(self.bn1(self.linear1(x)))
        # x = F.relu(self.bn2(self.linear2(x)))
        x = torch.sigmoid(self.linear3(x))
        return x