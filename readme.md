
# Requirements

```python
networkx                      3.1
numpy                         1.24.3
pandas                        2.0.1
pyod                          1.1.0
scikit-learn                  1.2.2
scipy                         1.10.1
six                           1.16.0
torch                         1.13.1+cu116
torch-cluster                 1.6.1+pt113cu116
torch-geometric               2.3.1
torch-scatter                 2.1.1+pt113cu116
torch-sparse                  0.6.17+pt113cu116
torch-spline-conv             1.2.2+pt113cu116
tqdm                          4.65.0
```



# Dataset



**It should be noted that the disney, books, weibo, enron, reddit, cora and inj_amazon datasets have been attached to the folder data. The  DGraphFin dataset are based on the PYG library and can be automatically downloaded when the code is running.**

| Dataset | Type                                   | location           |
| ------- | -------------------------------------- | ------------------ | 
| Disney  | Real anomaly detection dataset         | data/              | 95                                            |
| Books   | Real anomaly detection dataset         | data/              | 98                                            |
| Reddit  | Real anomaly detection dataset         | data/              | 95                                            |
| Enron   | Real anomaly detection dataset         | data/              | 99                                            |
| Weibo   | Real anomaly detection dataset         | data/              | 90                                            |
| Dgraph  | Real anomaly detection dataset         | Automatic download | 99                                            |
| Cora    | Graph dataset with synthetic anomalies | data/              | 95                                            |
| Amazon  | Graph dataset with synthetic anomalies | data/              | 95                                            |



# Running the experiments



#### Run MGAD algorithm with not splitting batches

````python
python main.py 
````



#### Run MGAD algorithm with  splitting batches

````python
python main_batch.py # Applies to big datases, such as Dgraph
````



#### Run backbone algorithm

````python
python main_backbone.py # Run baseline model such as Dominant and DONE
python main_backbone_mlpae.py # Run baseline model MLPAE
````

| Algorithm  | Backbone   |
| ---------- | ---------- |
| LOF        | -          |
| IF         | -          |
| MLPAE      | MLP+AE     |
| GAE        | GNN+AE     |
| ONE        | MF         |
| AdONE      | MLP+AE     |
| AnomalyDAE | GNN+AE     |
| GAAN       | GAN        |
| GUIDE      | GNN+AE     |
| CONAD      | GNN+AE+SSL |

![]https://github.com/DCGAE2023/DC-GAE/blob/master/Nodes_time.png
