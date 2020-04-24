import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from models import GCNLink, GCNClusterNet, GCNDeep
from utils import make_normalized_adj, negative_sample, edge_dropout,load_data
from modularity import make_modularity_matrix, greedy_modularity_communities
from loss_functions import loss_modularity
import time


K = 5 #number of clusters
num_cluster_iter = 1
np.random.seed(24)
torch.manual_seed(24)

universities = ['cornell','washington','wisconsin','texas']
'''Choose a University'''

uni = universities[0]
print('dataset:'+uni)

adj_train, bin_adj_train, features_train = load_data('data/{}/'.format(uni), '{}_train_{:.2f}'.format(uni, 0.4))
adj_all, bin_adj_all, features_test = load_data('data/{}/'.format(uni), '{}'.format(uni))

n = bin_adj_all.shape[0]
nfeat = features_test.shape[1]

# Model and optimizer
'''GCN two stage'''
model_ts = GCNLink(no_features=nfeat,
            no_hidden=50,
            no_output=50,
            dropout=0.5)
'''ClusterNet Model'''
model_cluster = GCNClusterNet(no_features=nfeat,
            no_hidden=50,
            no_output=50,
            dropout=0.5,
            K = K,
            cluster_temp = 30)
#uses GCNs to predict the cluster membership of each node
'''GCN end to end'''
model_gcn = GCNDeep(no_features=nfeat,
            no_hidden=50,
            no_output=K,
            dropout=0.5)
optimizer = optim.Adam(model_cluster.parameters(), lr=0.005, weight_decay=5e-4)
losses = []
losses_test = []

mod_train = make_modularity_matrix(bin_adj_train)
# mod_valid = make_modularity_matrix(bin_adj_valid)
mod_all = make_modularity_matrix(bin_adj_all)

loss_fn = loss_modularity
test_object = mod_all
train_object = mod_train
'''ClusterNet'''
clusterNet_start = time.time()
best_train_val = 100
for t in range(1001):
    #link prediction setting: get loss with respect to training edges only
    r, dist = model_cluster(features_train, adj_train, num_cluster_iter)
    loss = loss_fn(r, bin_adj_train, train_object)
    loss = -loss
    optimizer.zero_grad()
    loss.backward()
    #increase number of clustering iterations after 500 updates to fine-tune
    if t == 500:
        num_cluster_iter = 5
    #every 100 iterations, look and see if we've improved on the best training loss
    #seen so far. Keep the solution with best training value.
    if t % 100 == 0:
        #round solution to discrete partitioning
        r = torch.softmax(100*r, dim=1)
        #evalaute test loss -- note that the best solution is
        #chosen with respect training loss. Here, we store the test loss
        #of the currently best training solution
        loss_test = loss_fn(r, bin_adj_all, test_object)
        #training loss, to do rounding after
        if loss.item() < best_train_val:
            best_train_val = loss.item()
            curr_test_loss = loss_test.item()
            #convert distances into a feasible (fractional x)
            x_best = torch.softmax(dist*100, 0).sum(dim=1)
            x_best = 2*(torch.sigmoid(4*x_best) - 0.5)
            if x_best.sum() > K:
                x_best = K*x_best/x_best.sum()
    losses.append(loss.item())
    optimizer.step()
clusterNet_end = time.time()
print('ClusterNet value', curr_test_loss)
print('Timing', str(clusterNet_end - clusterNet_start))

#input 
'''GCN-2stage'''
def train_twostage(model_ts):
    optimizer_ts = optim.Adam(model_ts.parameters(), lr=0.005, weight_decay=5e-4)
    edges = adj_train.indices().t()
    for t in range(300):
        adj_input = make_normalized_adj(edge_dropout(edges, 0.2), n)
        edges_eval, labels = negative_sample(edges, 1, bin_adj_train)
        preds = model_ts(features_train, adj_input, edges_eval)
        loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
        optimizer_ts.zero_grad()
        loss.backward()
        optimizer_ts.step()
train_twostage(model_ts)
#predict probability that all unobserved edges exist
indices = torch.tensor(np.arange(n))
to_pred = torch.zeros(n**2, 2)
to_pred[:, 1] = indices.repeat(n)
for i in range(n):
    to_pred[i*n:(i+1)*n, 0] = i
to_pred = to_pred.long()
preds = model_ts(features_train, adj_train, to_pred)
preds = nn.Sigmoid()(preds).view(n, n)
preds = bin_adj_train + (1 - bin_adj_train)*preds
r = greedy_modularity_communities(preds, K)
GCN_2stage_end = time.time()
print('two stage value', loss_fn(r, bin_adj_all, test_object).item())
print('Timing', str(GCN_2stage_end- clusterNet_end))


'''GCN e2e'''
best_train_val = 0
for t in range(1000):
    best_train_loss = 100
    r = model_gcn(features_train, adj_train)
    r = torch.softmax(r, dim = 1)
    loss = -loss_fn(r, bin_adj_train, train_object)
    optimizer.zero_grad()
    loss.backward()
    if t % 100 == 0:
        r = torch.softmax(100*r, dim=1)
        loss_test = loss_fn(r, bin_adj_all, test_object)
        losses_test.append(loss_test.item())
        if loss.item() < best_train_val:
            curr_test_loss = loss_test.item()
            best_train_val = loss.item()
    losses.append(loss.item())
    optimizer.step()
GCN_e2e_end = time.time()
print('GCN-implicit value', curr_test_loss)
print('Timing', str(GCN_e2e_end - GCN_2stage_end))

'''No prediction value'''
preds = bin_adj_train
r = greedy_modularity_communities(preds, K)
end = time.time()
print('no prediction value', loss_fn(r, bin_adj_all, test_object).item())
print('Timing', str(end - GCN_e2e_end))