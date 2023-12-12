#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import numpy as np
import itertools
import sys

from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

from models import TimeEncoder, GATModel, LinkPredictor
from helpers import TemporalLoader, create_mask_in_batches, create_subset, make_time_relative, prepare_data, append_to_csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# In[8]:


@torch.no_grad()
def test(gnn, link_pred, time_enc, loader):
    gnn.eval()
    link_pred.eval()
    time_enc.eval()

    true_labels = torch.Tensor().to(device)
    predictions = torch.Tensor().to(device)
    batches = 0
    for batch in tqdm(loader, desc='Batches'):
        batches += 1
        mask = (batch.edge_y < 0)
        node_embs = gnn(batch.node_id, batch.node_type, batch.edge_index[:,mask], batch.edge_attr[mask], batch.time[mask], batch.edge_type[mask])
        mask = (batch.edge_y >= 0) & (batch.edge_type == 1)
        link_preds = link_pred(node_embs[batch.edge_index[0,mask]], node_embs[batch.edge_index[1,mask]], batch.time[mask]).sigmoid()

        predictions = torch.cat((predictions, link_preds[:,0].detach()), dim=0)
        true_labels = torch.cat((true_labels, batch.edge_y[mask]), dim=0)
    if batches == 0:
        return test(gnn, link_pred, time_enc, loader)
    loss = criterion(predictions, true_labels)
    auc = roc_auc_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    ap = average_precision_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    return loss, auc, ap


# In[79]:


# hyperparameters
batch_size = 512
num_epochs = 5
emb_dim = 100
hidden_dim = 100
time_dim = 100
identity_dim = 100
learning_rate = 0.0001
plot = False


# In[80]:


# experimental setting
relative_times = [True, False]
semi_transductives = [True, False]
heterogenous_msg_passings = [True, False]
# dataset_paths = ["data/act-mooc/graph.pt", "data/junyi/graph_sub.pt"]
dataset_paths = sys.argv[1:]


# In[81]:


for i in range(0, 3):
    for relative_time, semi_transductive, heterogenous_msg_passing, dataset_path in itertools.product(relative_times, semi_transductives, heterogenous_msg_passings, dataset_paths):
        print(f"Combination: relative_time={relative_time}, semi_transductive={semi_transductive}, "
              f"heterogenous_msg_passing={heterogenous_msg_passing}, dataset_path={dataset_path}")
        data = prepare_data(dataset_path, relative_time=False).to(device)
        
        time_array = data.time.cpu().numpy()
        quantile_70 = np.quantile(time_array, 0.7)
        quantile_85 = np.quantile(time_array, 0.85)
        
        # Create masks for splitting the data
        train_mask = time_array < quantile_70
        val_mask = time_array < quantile_85
        test_mask = np.ones_like(time_array, dtype=bool)
        
        # Create subsets
        train_data = create_subset(data, train_mask)
        val_data = create_subset(data, val_mask)
        test_data = create_subset(data, test_mask)
        
        train_loader = TemporalLoader(train_data, 0, device, batch_size=batch_size)
        val_loader = TemporalLoader(val_data, quantile_70, device, batch_size=batch_size)
        test_loader = TemporalLoader(test_data, quantile_85, device, batch_size=batch_size)

        time_enc = TimeEncoder(time_dim).to(device)
        gnn = GATModel(identity_dim, hidden_dim, emb_dim, time_dim+data.edge_attr.size(1), time_enc, data.node_id[data.node_type == 1], device, heterogenous_msg_passing, semi_transductive).to(device)
        link_pred = LinkPredictor(emb_dim, time_enc).to(device)
        optimizer = torch.optim.Adam(set(time_enc.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()), lr=learning_rate)
        num_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"] if p.requires_grad)
        print(f'{num_params} params')

        criterion = torch.nn.BCEWithLogitsLoss()
        train_losses = []
        val_losses = []
        val_aucs = []
        for epoch in range(0, num_epochs):
            gnn.train()
            link_pred.train()
            time_enc.train()
            losses = np.array([])
            for batch in tqdm(train_loader, desc='Training Batches'):
                batch.to(device)
                # forward pass with edges that are older than this batch
                mask = batch.edge_y < 0
                node_embs = gnn(batch.node_id, batch.node_type, batch.edge_index[:,mask], batch.edge_attr[mask], batch.time[mask], batch.edge_type[mask])
                
                # first predict for all dropout edges
                mask = (batch.edge_y == 1) & (batch.edge_type == 1)
                positive_edges = batch.edge_index[:, mask]
                if positive_edges.size(1) == 0: continue
                pos_out = link_pred(node_embs[positive_edges[0]], node_embs[positive_edges[1]], batch.time[mask])
                loss = criterion(pos_out, torch.ones_like(pos_out))
        
                # sample the same number of 0 edges from edge_index
                negative_indices = torch.nonzero((batch.edge_y == 0) & (batch.edge_type == 1)).squeeze()
                negative_indices = negative_indices[torch.randperm(negative_indices.size(0))][:positive_edges.size(1)]
                negative_edges = batch.edge_index[:, negative_indices]
                neg_out = link_pred(node_embs[negative_edges[0]], node_embs[negative_edges[1]], batch.time[negative_indices])
                loss += criterion(neg_out, torch.zeros_like(neg_out))
        
                # backward pass and gradient update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses = np.append(losses, loss.detach().cpu()/2)

            val_loss, val_auc, val_ap = test(gnn, link_pred, time_enc, val_loader)
            train_losses.append(losses.mean())
            val_losses.append(val_loss.detach().cpu())
            val_aucs.append(val_auc)
            print(f'Epoch: {epoch}, Train Loss: {losses.mean()}, Val Loss: {val_loss}, Val AUC: {val_auc}')
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.plot(val_aucs, label='Val AUC')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Per Epoch')
            plt.legend()
            plt.show()
            
        train_loss, train_auc, train_ap = test(gnn, link_pred, time_enc, train_loader)
        test_loss, test_auc, test_ap = test(gnn, link_pred, time_enc, test_loader)
        
        append_to_csv("results.csv", dataset_path, relative_time, semi_transductive, heterogenous_msg_passing,
                      batch_size, num_epochs, learning_rate, emb_dim, hidden_dim, time_dim, identity_dim,
                      train_loss.item(), val_loss.item(), test_loss.item(), train_auc, val_auc, test_auc, train_ap, val_ap, test_ap, i, num_params)


# In[ ]:




