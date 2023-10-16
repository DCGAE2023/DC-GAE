import torch
import pickle
import numpy as np
import torch_geometric.transforms as T

from tqdm import tqdm
from model import Model
from sklearn import preprocessing
from utils.GCL.eval import get_split, LREvaluator
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.utils import to_dense_adj, mask_feature
from torch_geometric.datasets import Planetoid, AttributedGraphDataset, DGraphFin
from utils.util import gen_contextual_outlier, gen_structural_outlier, remove_edge_random, add_random_edge


# Load real anomaly dataset
data = torch.load('data/books.pt')
data.y = torch.tensor([int(bool(x)) for x in data.y])
# Books, Weibo or Enron dataset, comment this line
data = T.NormalizeFeatures()(data)

# Load graph dataset and inject manually, and this dataset is based on the PYG library and can be downloaded automatically
# dataset = Planetoid('./data', name='Cora')
# data = dataset[0]
# data = T.NormalizeFeatures()(data)
# data, ys = gen_structural_outlier(data, m=7, n=10) 
# data, ya = gen_contextual_outlier(data, n=70, k=20) 
# data.y = torch.logical_or(ys, ya).long()

# Load graph datasets inj_amazon with synthetic anomalies
# data = torch.load('data/inj_amazon.pt')
# data.y = torch.tensor([int(bool(x)) for x in data.y])

in_dim = data.x.size(1)
hid_dim = 128
dropout = 0.1
model = Model(in_dim, hid_dim, data, dropout)
# Enron dataset set 0.001
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

def classify_test(model, data):
    model.eval()
    z = model.classify_infer(data)
    split = get_split(num_samples=z.size()[0], train_ratio=0.2, test_ratio=0.7)
    result = LREvaluator()(z, data.y, split) 
    return result['ap']

def outiler_test(model, data, loader, epoch):
    model.eval()
    A_hat, X_hat, emb = model.infer(data)
    score, _, _ = model.loss_func(to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(), A_hat, data.x, X_hat, 0)
    score = score.detach().cpu().numpy()
    score = preprocessing.scale(score) 
    auc = roc_auc_score(data.y, score) 
    fpr, tpr, thresholds = roc_curve(data.y, score, drop_intermediate=False) 
    print("Epoch:", '%04d' % (epoch), 'Auc', auc)
    return auc, (fpr, tpr, emb, score) 

def train_ord(epoch, data):
    model.train()
    optimizer.zero_grad()
    A_hat, X_hat, l = model(data, epoch)
    if(True):loss = model.threshold_loss(to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze(), A_hat, data.x, X_hat, epoch)
    else:loss, struct_loss, feat_loss = model.loss_func(to_dense_adj(data.edge_index).squeeze(), A_hat, data.x, X_hat, 0.5)
    loss = torch.mean(loss)
    if(l):loss += l
    loss.backward()
    optimizer.step()        
    print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()))
    
    return loss

def main():
  
    global data  
    input_nodes = torch.arange(0, len(data.y))
    loader = NeighborLoader(data, num_neighbors=[50] * 2, shuffle=True, input_nodes=input_nodes, batch_size=512)
    best = -999
    auc_list = []
    loss_list = []
    
    for epoch in range(0,300):
        
        loss = train_ord(epoch, data)
        if(epoch % 10 == 0):
            score, curve = outiler_test(model, data, loader, epoch)
            auc_list.append(score)
            loss_list.append(loss.item())
            if(best < score):
                best = score
                print(best)
    print("AUC:",best)
    
main()