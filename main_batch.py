import torch
import torch_geometric.transforms as T

from tqdm import tqdm
from model_batch import Model
from sklearn import preprocessing
from utils.GCL.eval import get_split, LREvaluator
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.utils import to_dense_adj, mask_feature
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import DGraphFin
from utils.util import gen_contextual_outlier, gen_structural_outlier, remove_edge_random, add_random_edge 

'''
This code is only for anomaly detection experiments on large-scale datasets such as DGraphFin. 
The code aims to split the input graph data into batches and perform training and testing.
'''

# this dataset is based on the PYG library and can be downloaded automatically
dataset = DGraphFin(root='./data/DGraph')
data = dataset[0]
mask = (data.y == 0) | (data.y == 1)
mask = mask.bool()

in_dim = data.x.size(1) 
hid_dim = 64 
dropout = 0.1 
model = Model(in_dim, hid_dim, data, dropout)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

def classify_test(model, loader):
    model.eval()
    z = model.classify_batch(loader)[mask]
    split = get_split(num_samples=z.size()[0], train_ratio=0.2, test_ratio=0.7) 
    result = LREvaluator()(z, data.y[mask], split) 
    return result['ap']

def outiler_test(model, data, loader, epoch):
    model.eval()
    score = model.infer_batch(loader)
    score = score.detach().cpu().numpy()
    score = preprocessing.scale(score) 
    auc = roc_auc_score(data.y[mask], score[mask]) 
    fpr, tpr, thresholds = roc_curve(data.y[mask], score[mask]) 
    print("Epoch:", '%04d' % (epoch), 'Auc', auc)
    return auc, (fpr, tpr) 

def train_batch(loader, epoch):
    model.train()
    total_examples = total_loss = 0
    for batch in tqdm(loader):
        optimizer.zero_grad() 
        batch_size = batch.batch_size
        A_hat, X_hat, l = model(batch, epoch)
        if(True):loss = model.threshold_loss(to_dense_adj(batch.edge_index, max_num_nodes=batch.x.size(0)).squeeze(), A_hat, batch.x, X_hat, epoch)
        else:loss, struct_loss, feat_loss = model.loss_func(to_dense_adj(batch.edge_index, max_num_nodes=batch.x.size(0)).squeeze(), A_hat, batch.x, X_hat, 0.5)
        loss = torch.mean(loss)
        if(l): loss += l
        loss.backward() 
        optimizer.step()      
        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples

def main():
    
    input_nodes = torch.arange(0,len(data.y))
    loader = NeighborLoader(data, num_neighbors=[60] * 2, shuffle=True, input_nodes=input_nodes, batch_size=128)
    best = -999
    
    for epoch in range(1,50):
        train_batch(loader, epoch)
        if(epoch % 10 == 0):
            score, curve = outiler_test(model, data, loader, epoch)
            if(best < score):
                best = score 
            print(best)
    
main()