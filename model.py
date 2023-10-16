import os
import tqdm
import torch
import warnings
import numpy as np
import torch.nn as nn
import GCL.losses as L
import torch.nn.functional as F
from sklearn.ensemble import IsolationForest

import networkx as nx
import torch.nn.functional as F
from scipy import sparse as sp
from scipy import sparse
from utils.util import IQR_threshold
from torch_scatter import scatter
from torch_geometric.nn.models import MLP
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import mask_feature, dropout_node, to_dense_adj, to_networkx


warnings.filterwarnings("ignore")
delta = 0
x_residual = 0
adj_residual = 0

def isolate_score(data):
    
    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, random_state=42)
    clf.fit(data.x)
    scores = clf.decision_function(data.x)
    
    return scores

def knn_score(data):
    
    from pyod.models.knn import KNN
    clf = KNN(n_neighbors=3, method='mean')
    clf.fit(data.x)
    y_pred = clf.labels_
    y_scores = clf.decision_scores_
    
    return y_scores


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, dropout):

        super(Encoder, self).__init__()
        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.dropout = dropout

    def forward(self, x, edge_index):
        
        x1 = F.relu(self.gc1(x, edge_index))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = F.relu(self.gc2(x1, edge_index))
        
        return x2

class Attribute_Decoder(nn.Module):
    
    def __init__(self, nfeat, nhid, dropout):

        super(Attribute_Decoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.dropout = dropout

    def forward(self, x, edge_index):

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, edge_index))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, dropout):

        super(Structure_Decoder, self).__init__()
        self.gc1 = GCNConv(nhid, nhid)
        self.dropout = dropout
    
    def forward(self, x, edge_index):

        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = x @ x.T 

        return x

class LatentContextualRegressor(nn.Module):

    def __init__(self, dim, num_heads):

        super(LatentContextualRegressor, self).__init__()
        
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attns = nn.MultiheadAttention(dim, num_heads)
        self.att_drop = nn.Dropout(p=0.1)

    def forward(self, q, k, v):

        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)
        q = self.att_drop(self.cross_attns(q, k, v)[0])
        q = self.norm_cross(q)
        
        return q

class Model(nn.Module):

    def __init__(self, feat_size, hidden_size, data, dropout):

        super(Model, self).__init__()
        self.num_nodes = len(data.y)
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout)
        self.shared_encoder_ = Encoder(feat_size, hidden_size, dropout)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.attr_decoder_ = Attribute_Decoder(feat_size, hidden_size, dropout)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout)
        self.struct_decoder_ = Structure_Decoder(hidden_size, dropout)
        self.pos_emb = self.laplacian_positional_encoding(data, self.hidden_size)
        data.mask_weight = self.mask_weight = self.mask_weight_calc(data)
            
        self.linear_x = nn.Linear(feat_size, hidden_size)
        self.linear_x_vis = nn.Linear(hidden_size, feat_size)
        self.project = nn.Linear(feat_size, 32)
        self.hidden_size = hidden_size
        
        self.mask_token = nn.Parameter(torch.zeros(1, hidden_size))
        self.lcr = nn.ModuleList([LatentContextualRegressor(hidden_size, 8) for _ in range(1)])
        self.mlp_processor = MLP(in_channels=hidden_size, hidden_channels=hidden_size // 2, out_channels=feat_size, num_layers=2)
        self.mlp_processor_ = MLP(in_channels=hidden_size, hidden_channels=hidden_size // 2, out_channels=hidden_size, num_layers=2)
        self.cross_decoder = LinkPredictor(feat_size, hidden_size, data.x.size(0))
       
    def stru_weight_calc(self, data):
        
        adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.size(0)).squeeze()
        q = int(len(data.y) * 0.1)
        u,s,v = torch.svd_lowrank(adj) 
        s = torch.diag(s)
        s[q+1:,q+1:] = 0
        u = u[:,:q]
        v = v[:,:q]
        adj = torch.matmul(torch.matmul(u, s), v.t())
        score = torch.mean(adj, dim=1)
         
        return score
    
    def sema_weight_calc(self, data):

        q = int(len(data.y) * 0.1)
        u,s,v = torch.svd_lowrank(data.x)
        s = torch.diag(s)
        s[q+1:,q+1:] = 0
        u = u[:,:q] 
        v = v[:,:q]
        x = torch.matmul(torch.matmul(u, s), v.t())
        
        diff_attribute = torch.pow(x - data.x, 2)
        score = torch.sqrt(torch.sum(diff_attribute, 1))

        return score
    
    # Implement fusion mask weight score calculation
    def mask_weight_calc(self, data):
        
        # sema_score = torch.tensor(isolate_score(data))
        # sema_score = torch.tensor(knn_score(data))
        sema_score = self.sema_weight_calc(data)
        stru_score = self.stru_weight_calc(data)    

        sema_score = (sema_score - torch.min(sema_score)) / (torch.max(sema_score) - torch.min(sema_score))
        stru_score = (stru_score - torch.min(stru_score)) / (torch.max(stru_score) - torch.min(stru_score))

        score = sema_score + 1 - stru_score
        score = (score - torch.min(score)) / (torch.max(score) - torch.min(score))
        score = torch.tanh(3 * score)
        print(torch.mean(score))
        
        return score
    
    
    def laplacian_positional_encoding(self, data, pos_size):
        
        g = to_networkx(data) 
        A = nx.adjacency_matrix(g).todense().astype(float) 
        in_degree = [item[1] for item in g.in_degree()] 
        N = sp.diags(np.array(in_degree).clip(1) ** -0.5, dtype=float) 
        L = sp.eye(data.num_nodes) - N * A * N 
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_size+1, which='SR', tol=1e-2) 
        EigVec = EigVec[:, EigVal.argsort()]
        x_pos = torch.from_numpy(EigVec[:,1:pos_size+1]).float()

        return x_pos
        
    def forward(self, data, epoch):

        mask_mode = 0 
        compose = 0 
        extra_loss = True 
        x, edge_index = data.x, data.edge_index 

        if(mask_mode == 0): mask = torch.bernoulli(data.mask_weight).to(torch.bool) 
        else: _, _, mask = dropout_node(data.edge_index, p=0.46) #

        # Get the corresponding part of the visible node
        set = torch.arange(0,data.x.size(0))
        subset = set[mask]
        subgraph = data.subgraph(subset)
        x_vis = self.shared_encoder(subgraph.x, subgraph.edge_index)
        
        # Predicts mask nodes based on visible node representations
        x_masked = self.mask_token.expand(len(data.x[~mask]),-1) 
        for blk in self.lcr:x_masked = blk(x_masked, x_vis, x_vis)
        
        if(compose == 0):
            x = torch.zeros_like(self.linear_x(data.x))
            x[mask == 1] = x_vis
            x[mask == 0] = x_masked
            x_hat = self.attr_decoder(x, edge_index)
            # For graph datasets with synthetic anomalies, it is recommended to use the following code
            x_hat = self.mlp_processor(x)
        
        else:
            x_hat = torch.zeros_like(data.x)
            x_hat[mask == 1] = self.linear_x_vis(x_vis)
            x_hat[mask == 0] = self.mlp_processor(x_masked)
        
        adj = self.cross_decoder(x_hat, x_hat)
        struct_reconstructed = adj

        # Alignment loss
        if(extra_loss):
            z = self.shared_encoder_(data.x, edge_index)
            loss = self.align_loss(x_masked, z[~mask]) 
            return struct_reconstructed, x_hat, loss
        else:
            return struct_reconstructed, x_hat, None
            
    def infer(self, data):
        
        x, edge_index = data.x, data.edge_index
        x = self.shared_encoder(x, edge_index)
        x_hat = self.attr_decoder(x, edge_index)
        struct_reconstructed = self.cross_decoder(x_hat, x_hat)

        return struct_reconstructed, x_hat, x
    
    def classify_infer(self, data):
        
        x, edge_index = data.x, data.edge_index
        x = self.shared_encoder(x, edge_index)

        return x

    # Threshold loss and huber loss
    def threshold_loss(self, adj, A_hat, attrs, X_hat, epoch):

        beta = 0.5
        
        def hu_loss(x,y,delta):
        
            return  torch.abs(x-y) * delta - 0.5 * delta ** 2
        
        if(True):
            x_delta = 0
            adj_delta = 0
            global delta, x_residual, adj_residual
            def msle_loss(x, y):
                log_residual = torch.log(abs(x-y)+1) ** 2
                return log_residual
            
            x_dist = torch.mean(torch.abs(attrs - X_hat), dim=1) 
            adj_dist = torch.mean(torch.abs(adj - A_hat), dim=1)

            if(epoch == 0): 
                _, x_delta = IQR_threshold(x_dist.detach().cpu().numpy(), p=1.5)
                _, adj_delta = IQR_threshold(adj_dist.detach().cpu().numpy(), p=1.5)
            elif(epoch % 10 == 0): 
                x_residual, adj_residual = x_residual / 10, adj_residual / 10
                _, x_delta = IQR_threshold(x_residual.detach().cpu().numpy(), p=1.5)
                _, adj_delta = IQR_threshold(adj_residual.detach().cpu().numpy(), p=1.5)
                x_residual = 0
                adj_residual = 0
            else:
                x_residual += x_dist
                adj_residual += adj_dist
            
            adj_loss = torch.where(torch.abs(adj - A_hat) < adj_delta, 0.5 * ((A_hat - adj)**2), msle_loss(adj, A_hat))
            x_loss = torch.where(torch.abs(attrs - X_hat) < x_delta, 0.5 * ((X_hat - attrs)**2), msle_loss(attrs, X_hat))
            loss = beta * torch.mean(adj_loss, dim=1) + (1 - beta) * torch.mean(x_loss, dim=1)
    
        return loss
    
    def loss_func(self, adj, A_hat, attrs, X_hat, alpha):
        
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)
    
        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)
       
        cost =  (1-alpha) * attribute_reconstruction_errors + alpha * structure_reconstruction_errors
        
        return cost, structure_cost, attribute_cost
    
    def kl_loss(self, x, y):
        
        p = F.softmax(x, dim=1) 
        q = F.softmax(y, dim=1) 

        log_p = torch.log(p) 
        log_q = torch.log(q) 
        kl = torch.mean(log_p - log_q, dim=1) 

        total_kl = torch.sum(kl) 
        mean_kl = torch.mean(kl) 

        return mean_kl
    
    def sce_loss(self, x, y, alpha=3):

        x = F.normalize(x, p=2, dim=-1) 
        y = F.normalize(y, p=2, dim=-1)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        return loss
    
    def loss_func_(self, adj, A_hat, attrs, X_hat, alpha=1):

        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)

        attribute_reconstruction_errors = self.sce_loss(attrs, X_hat)
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

        return cost, structure_cost, attribute_cost
    
    def align_loss(self, attrs, X_hat):
      
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        return attribute_cost

class LinkPredictor(torch.nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        
        super(LinkPredictor, self).__init__()
        self.hidden_channels = hidden_channels
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j 
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        self.lin_final = nn.Linear(self.hidden_channels, x_i.shape[0])
        x = self.lin_final(x)
        return torch.sigmoid(x)