import argparse
import os
import sys
import torch
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
from torch_geometric.datasets import Planetoid,WebKB,Actor,WikipediaNetwork, LINKXDataset
from torch_geometric.data import DataLoader,Data,ClusterData,ClusterLoader
from torch_geometric.loader import RandomNodeSampler
from torch_geometric.utils import to_dense_adj
from models import *
from utils import *
from ogb.nodeproppred import NodePropPredDataset
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(1234)
np.random.seed(1234)
################### Arguments parameters ###################################
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    default="texas",
    choices=["pubmed","penn94","ogbn-arxiv","arxiv-year"],
    help="You can choose between pubmed, penn94, ogbn-arxiv",
)
parser.add_argument(
    "--cuda",
    default="cuda:0",
    choices=["cuda:0","cuda:1","cpu"],
    help="You can choose between cuda:0, cuda:1, cpu",
)
parser.add_argument(
        "--hidden_channels", type=int, default=16, help="Hidden channels for the unsupervised model"
)
parser.add_argument(
        "--hidden_channels_unsupervised", type=int, default=16, help="Hidden channels for the unsupervised model"
)
parser.add_argument(
        "--lr_unsupervised", type=float, default=0.03, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd_unsupervised", type=float, default=0, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout rate"
    )
parser.add_argument(
        "--lr", type=float, default=0.01, help="Outer learning rate of model"
    )
parser.add_argument(
        "--wd", type=float, default=5e-4, help="Outer weight decay rate of model"
    )
parser.add_argument(
        "--epochs_unsupervised", type=int, default=100, help="Epochs for the unsupervised model"
    )
parser.add_argument(
        "--epochs", type=int, default=200, help="Epochs for the model"
    )
parser.add_argument(
        "--unsupervised_mode", type=bool, default=False, help="Use ODwire unsupervised o supervised"
)
parser.add_argument(
        "--n_layers", type=int, default=10, help="Number of hops"
    )
parser.add_argument(
        "--num_centers", type=int, default=5, help="Number of centers"
)
parser.add_argument(
        '--log_file', nargs='?')
args = parser.parse_args()
################### Importing the dataset ###################################
if args.dataset == "pubmed":
    dataset = Planetoid(root='./data',name='pubmed')
    data = dataset[0]
elif args.dataset == "penn94":
    dataset = LINKXDataset(root='./data',name='penn94')
    data = dataset[0]
elif args.dataset == "ogbn-arxiv":
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    print(dataset)
    print(dataset.num_classes)
    print(dataset.graph.keys())
    split_index = dataset.get_idx_split()
    # Parse to tensor
    data = Data(x=torch.from_numpy(dataset.graph['node_feat']).float(),
                edge_index=torch.from_numpy(dataset.graph['edge_index']).long(),
                y=torch.from_numpy(dataset.labels).long())
    data.train_mask = torch.from_numpy(split_index['train']).bool()
    data.val_mask = torch.from_numpy(split_index['valid']).bool()
    data.test_mask = torch.from_numpy(split_index['test']).bool()
    dataset.num_features = dataset.graph['node_feat'].shape[1]
    dataset.num_classes = dataset.labels.max() + 1
elif args.dataset == 'arxiv-year':
    dataset = NodePropPredDataset(name='ogbn-arxiv')
    print(dataset)
    print(dataset.num_classes)
    print(dataset.graph.keys())
    dataset.name = 'arxiv_year'
    split_index = dataset.get_idx_split()
    label = even_quantile_labels(
        dataset.graph['node_year'].flatten(), 5, verbose=False)
    dataset.label = torch.as_tensor(label).reshape(-1, 1)
    # Parse to tensor
    data = Data(x=torch.from_numpy(dataset.graph['node_feat']).float(),
                edge_index=torch.from_numpy(dataset.graph['edge_index']).long(),
                y= dataset.label.long().squeeze(1))
    
    dataset.num_features = dataset.graph['node_feat'].shape[1]
    dataset.num_classes = dataset.label.max() + 1
    print(dataset.name)
    
print()
# Let's see the values of the dataset
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
print()
print(data) 
dataset.graph['edge_index'] = None
print('===========================================================================================================')
adj = to_dense_adj(data.edge_index)[0] # Convert the sparse adjacency matrix to a dense adjacency matrix
################### CUDA ###################################
device = torch.device(args.cuda)
data = data.to(device)   
print("Device: ",device)
adj = to_dense_adj(data.edge_index)[0].to(device)
edge_indexs = []
edge_indexs.append(data.edge_index.to('cpu').clone())
data.edge_index = None
################### Training the model in a unsupervised way ###################################
if args.unsupervised_mode:  
    print('Unsupervised mode')
    edge_indexs = None
    data.edge_index = None
    data.x = None
    data.y = None
    unsupervised_model = DJ_unsupervised(adj_dim=adj.shape[0],
                                         num_centers=args.num_centers).to(device)
    unsupervised_model.train()   
    optimizer = torch.optim.Adam(unsupervised_model.parameters(),
                                 lr=args.lr_unsupervised,
                                 weight_decay=args.wd_unsupervised)
    for epoch in range(args.epochs_unsupervised):
        new_adj = None
        loss , new_adj = train_adj(adj,data,unsupervised_model,optimizer)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print('===========================================================================================================')
    print('Final loss: ',loss.item())
    print('===========================================================================================================')
    adj = new_adj.detach().squeeze(0).numpy()
    # Store the new adjacency matrix
    np.save(dataset.name+'_new_adj.npy',adj)
    # Terminates the execution
    print('Men vaig a dormir')
    sys.exit(0)
else:
    # Load the new adjacency matrix
    adj = np.load(dataset.name+'_new_adj.npy')
# Parse adj to Tensor
adj = torch.tensor(adj).to('cpu')

# Get the edge_indexs
previus_adj = torch.zeros(adj.shape).to('cpu')
for i in range(args.n_layers):
    adj_j = torch.zeros_like(adj)
    top_min = torch.topk(adj, i, dim=1, largest=False, sorted=True)
    adj_j.scatter_(1, top_min.indices, 1)
    adj_j = adj_j - previus_adj
    edge_indexs.append(torch.nonzero(adj_j).t().to('cpu'))
    previus_adj = adj_j
#del unsupervised_model
del adj
del previus_adj
################### Training the model in a supervised way ###################################
results = []
seeds = [12381, 45891, 63012, 32612, 91738]
for i in range(5):
    if args.dataset == "penn94":
        #data = dataset[0]
        train_mask = data.train_mask[:,i]
        val_mask = data.val_mask[:,i]
        test_mask = data.test_mask[:,i]
    elif args.dataset == "ogbn-arxiv" or args.dataset == "arxiv-year":
        train_mask, val_mask, test_mask = rand_train_test_idx(data.y,seed=seeds[i])
        train_mask = train_mask.to('cpu')
        val_mask = val_mask.to('cpu')
        test_mask = test_mask.to('cpu')
    else:
        print("Error: Dataset not found")
        exit()
    print('===========================================================================================================')
    print('Split: ',i)
    print('===========================================================================================================')
    model = DJ_supervised(in_channels=dataset.num_features,
                                hidden_channels=args.hidden_channels,
                                n_jumps=args.n_layers,
                                out_channels=dataset.num_classes,
                                drop_out=args.dropout).to(device)
    
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    test_acc = 0
    adj = edge_indexs
    adj = [f.to(device) for f in adj]
    #adj = edge_indexs[0].to(device)
    for epoch in range(args.epochs):
        loss,acc_train = train(adj,data,model,train_mask,optimizer,criterion)
        acc_val = val(adj,data,model,val_mask)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {acc_train:.4f}, Val Acc: {acc_val:.4f}, Test Acc: {test_acc:.4f}')
        acc_test = test(adj,data,model,test_mask)
        if acc_test > test_acc:
            test_acc = acc_test
    print('===========================================================================================================')
    print('Test Accuracy: ',test_acc)
    print('===========================================================================================================')
    results.append(test_acc)
    del model
acc_mean = np.mean(results)*100
acc_std = np.std(results)*100
print('===========================================================================================================')
print('Report: ',acc_mean,'+-',acc_std)
print('===========================================================================================================')
print(' Configuration: ',args)
print('===========================================================================================================')

# Escribimos el resultado en un fichero de log
if args.log_file is not None:
    with open(args.log_file, 'a') as file:
        file.write('' + acc_mean.astype('str') + 
                ';' + acc_std.astype('str') + 
                ';' + args.dataset +
                ';' + str(args.num_centers) +
                ';' + str(args.n_layers) +
                ';' + str(args.lr) +
                ';' + str(args.dropout) +
                ';' + str(args.hidden_channels) +
                    '\n')
