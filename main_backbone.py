import torch
import pickle
import sklearn
import numpy as np
import scipy.io as scio
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.nn.models import MLP
from utils.pygod.pygod.metric import eval_roc_auc
from utils.util import remove_edge_random, add_random_edge
from sklearn.metrics import roc_auc_score, roc_curve
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.datasets import Planetoid, AttributedGraphDataset, DGraphFin, EllipticBitcoinDataset, DeezerEurope
from utils.pygod.pygod.generator import gen_contextual_outlier, gen_structural_outlier
from utils.pygod.pygod.detector import DOMINANT, CoLA, GUIDE, DONE, CONAD, GAAN, GAE, AnomalyDAE, AdONE, SCAN, Radar, ANOMALOUS

'''
This code implements the anomaly detection of traditional anomaly detection models such as LOF, IF 
and deep learning graph anomaly detection models such as Dominant, DONE and other baseline models.
'''

device = torch.device('cpu')

# real anomaly detection datasets
data = torch.load('data/disney.pt')
y = data.y.bool()

# DgraphFin dataset, and this dataset is based on the PYG library and can be downloaded automatically
# dataset = DGraphFin(root='./data/DGraph')
# data = dataset[0] 
# mask = (data.y == 0) | (data.y == 1)
# mask = mask.bool()

# Manually inject exception datasets, and this dataset is based on the PYG library and can be downloaded automatically
# dataset = Planetoid('./data', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]
# data, ys = gen_structural_outlier(data, m=70, n=10) 
# data, ya = gen_contextual_outlier(data, n=700, k=20) 
# data.y = torch.logical_or(ys, ya).long() 

# Dataset with synthetic anomalies
# data = torch.load('data/inj_amazon.pt')
# data.y = torch.tensor([int(bool(x)) for x in data.y])
# data = T.NormalizeFeatures()(data)

# LOF and Isolated Forest Detection Algorithms
from pyod.models.lof import LOF
from sklearn.ensemble import IsolationForest
# clf = LOF()
clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.1, random_state=42)
clf.fit(data.x)
test_scores = clf.decision_function(data.x)
roc = round(roc_auc_score(data.y, test_scores), ndigits=4)
print(roc)

detector = DOMINANT(hid_dim=64, num_layers=2, epoch=100)
# detector = GUIDE(hid_a=64, hid_s=64, num_layers=2, epoch=100)
# detector = CoLA(hid_dim=64, num_layers=2, epoch=100)
# detector = DONE(hid_dim=64, num_layers=2, epoch=100)
# detector = CONAD(hid_dim=64, num_layers=2, epoch=100)
# detector = GAE(hid_dim=64, num_layers=2, epoch=100)
# detector = GAAN(hid_dim=64, num_layers=2, epoch=100)
# detector = AnomalyDAE(hid_dim=64, num_layers=2, epoch=100)
# detector = AdONE(hid_dim=64, num_layers=2, epoch=100)
# detector = SCAN(eps=0.5, mu=2, contamination=0.1, verbose=0)
# detector = Radar(gamma=1.0, weight_decay=0.0, lr=0.004, epoch=100, gpu=-1, contamination=0.1, verbose=0)
# detector = ANOMALOUS(gamma=1.0, weight_decay=0.0, lr=0.004, epoch=100, gpu=-1, contamination=0.1, verbose=0)
detector.fit(data)
pred, score, prob = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=False)

print('Labels:')
print(pred)

print('Raw scores:')
print(score)

print('Probability:')
print(prob)

f1 = sklearn.metrics.f1_score(data.y, pred)
print(f1)

auc_score = eval_roc_auc(data.y, score)
print('AUC Score:', auc_score)

ap = sklearn.metrics.average_precision_score(data.y, score)
print("ap:",ap)