import torch
import random
import scipy.io as scio
import numpy as np
from copy import deepcopy
from numbers import Number
from torch_geometric.data import Data
from torch.autograd import Variable

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score
)

'''
This code mainly implements the various components required by the anomaly detection model implemented in this paper, 
including: context anomaly injection module, structure anomaly injection module, random noise addition module, 
and quartile threshold calculation module required by the threshold loss function.
'''

# For different datasets, modify the value to get better auc score
threshold = 95

def eval_roc_auc(label, score):
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def gen_structural_outlier(data, m, n, p=0, directed=False, seed=None):
    if seed:torch.manual_seed(seed)
    new_edges = []
    outlier_idx = torch.randperm(data.num_nodes)[:m * n]
    # connect all m nodes in each clique
    for i in range(n):
        new_edges.append(torch.combinations(outlier_idx[m * i: m * (i + 1)]))
    new_edges = torch.cat(new_edges)
    # drop edges with probability p
    if p != 0:
        indices = torch.randperm(len(new_edges))[:int((1-p) * len(new_edges))]
        new_edges = new_edges[indices]
    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1
    if not directed:new_edges = torch.cat([new_edges, new_edges.flip(1)], dim=0)
    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier

def gen_contextual_outlier(data, n, k, seed=None):
  
    if seed:torch.manual_seed(seed)
    outlier_idx = torch.randperm(data.num_nodes)[:n]

    for i, idx in enumerate(outlier_idx):
        candidate_idx = torch.randperm(data.num_nodes)[:k]
        euclidean_dist = torch.cdist(data.x[idx].unsqueeze(0), data.x[
            candidate_idx])

        max_dist_idx = torch.argmax(euclidean_dist, dim=1)
        max_dist_node = candidate_idx[max_dist_idx]
        data.x[idx] = data.x[max_dist_node]

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    return data, y_outlier

def to_np_array(*arrays, **kwargs):
    array_list = []
    for array in arrays:
        if isinstance(array, Variable):
            if array.is_cuda:
                array = array.cpu()
            array = array.data
        if isinstance(array, torch.Tensor) or isinstance(array, torch.FloatTensor) or isinstance(array, torch.LongTensor) or isinstance(array, torch.ByteTensor) or \
           isinstance(array, torch.cuda.FloatTensor) or isinstance(array, torch.cuda.LongTensor) or isinstance(array, torch.cuda.ByteTensor):
            if array.is_cuda:
                array = array.cpu()
            array = array.numpy()
        if isinstance(array, Number):
            pass
        elif isinstance(array, list) or isinstance(array, tuple):
            array = np.array(array)
        elif array.shape == (1,):
            if "full_reduce" in kwargs and kwargs["full_reduce"] is False:
                pass
            else:
                array = array[0]
        elif array.shape == ():
            array = array.tolist()
        array_list.append(array)
    if len(array_list) == 1:
        array_list = array_list[0]
    return array_list

def remove_edge_random(data, remove_edge_fraction):
    
    data_c = deepcopy(data)
    num_edges = int(data_c.edge_index.shape[1] / 2)
    num_removed_edges = int(num_edges * remove_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data_c.edge_index.T)]
    for i in range(num_removed_edges):
        idx = np.random.choice(len(edges))
        edge = edges[idx]
        edge_r = (edge[1], edge[0])
        edges.pop(idx)
        try:
            edges.remove(edge_r)
        except:
            pass
    data_c.edge_index = torch.LongTensor(np.array(edges).T).to(data.edge_index.device)
    return data_c

def add_random_edge(data, added_edge_fraction=0):
    
    if added_edge_fraction == 0:
        return data
    data_c = deepcopy(data)
    num_edges = int(data.edge_index.shape[1] / 2)
    num_added_edges = int(num_edges * added_edge_fraction)
    edges = [tuple(ele) for ele in to_np_array(data.edge_index.T)]
    added_edges = []
    for i in range(num_added_edges):
        while True:
            added_edge_cand = tuple(np.random.choice(data.x.shape[0], size=2, replace=False))
            added_edge_r_cand = (added_edge_cand[1], added_edge_cand[0])
            if added_edge_cand in edges or added_edge_cand in added_edges:
                if added_edge_cand in edges:
                    assert added_edge_r_cand in edges
                if added_edge_cand in added_edges:
                    assert added_edge_r_cand in added_edges
                continue
            else:
                added_edges.append(added_edge_cand)
                added_edges.append(added_edge_r_cand)
                break

    added_edge_index = torch.LongTensor(np.array(added_edges).T).to(data.edge_index.device)
    data_c.edge_index = torch.cat([data.edge_index, added_edge_index], 1)
    return data_c


def IQR_threshold(data, p):
    
    import numpy as np
    data.sort()

    # Calculate the first quartile (Q1), the second quartile (Q2) and the third quartile (Q3)
    Q1 = np.percentile(data, 25)
    Q2 = np.percentile(data, 50)
    Q3 = np.percentile(data, threshold)
    
    # Calculate interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower edge and upper edge to determine outliers
    lower_thres = Q1 - p * IQR
    upper_thres = Q3 
    
    return lower_thres, upper_thres