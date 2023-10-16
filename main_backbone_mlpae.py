import torch
import torch_geometric.transforms as T

from torch import nn
from sklearn.metrics import roc_auc_score
from torch_geometric.datasets import TUDataset, Planetoid, DeezerEurope
from utils.pygod.pygod.generator import gen_contextual_outlier, gen_structural_outlier

'''
This code implements the traditional anomaly detection model based on MLP+AE as the baseline model 
for the graph anomaly detection datasets involved in this paper.
'''

class MLPAutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(MLPAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    x_hat = model(data.x)
    loss = criterion(x_hat, data.x)
    loss.backward()
    optimizer.step()

def test(model, data):
    model.eval()
    with torch.no_grad():
      x_hat = model(data.x)
      score = torch.mean((data.x - x_hat)**2, dim=1)  
    return roc_auc_score(data.y, score)


# Load real anomaly dataset
# data = torch.load('data/books.pt')
# data.y = torch.tensor([int(bool(x)) for x in data.y])
# # Books, Weibo or Enron dataset, comment this line
# data = T.NormalizeFeatures()(data)

# Load graph datasets inj_amazon with synthetic anomalies
# data = torch.load('data/inj_amazon.pt')
# data.y = torch.tensor([int(bool(x)) for x in data.y])

# this dataset is based on the PYG library and can be downloaded automatically
dataset = Planetoid('./data', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0]
data = T.NormalizeFeatures()(data)
data, ys = gen_structural_outlier(data, m=7, n=10) 
data, ya = gen_contextual_outlier(data, n=70, k=20) 
data.y = torch.logical_or(ys, ya).long()

# Initialize the MLP autoencoder
model = MLPAutoEncoder(input_dim=dataset.num_features, hidden_dim=64, z_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
criterion = nn.MSELoss()

# Training loop
for epoch in range(100):
    train(model, data, optimizer, criterion)
    auc_score = test(model, data)
    print(f'Epoch: {epoch+1}, AUC: {auc_score:.4f}')
