import torch
from tqdm import tqdm
from torch_geometric.data import Data
import csv
import os
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay
import numpy as np

def prepare_data(dataset_path, relative_time=False):
    data = torch.load(dataset_path)
    data = data.to_homogeneous()
    data.x = torch.zeros(data.num_nodes, 1)
    data.edge_attr = data.edge_attr.float()
    data.time = data.time.float()

    if relative_time:
        data = make_time_relative(data)
        
    return data

def append_to_csv(filename, dataset_path, relative_time, semi_transductive, heterogenous_msg_passing,
                  batch_size, num_epochs, learning_rate, emb_dim, hidden_dim, time_dim, identity_dim,
                  train_loss, val_loss, test_loss, train_AUC, val_AUC, test_AUC, train_ap, val_ap, test_ap, run, num_params):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['timestamp', "dataset_path", 'relative_time', "semi_transductive", "heterogenous_msg_passing",
                      "batch_size", "num_epochs", "learning_rate", "emb_dim", "hidden_dim", "time_dim", "identity_dim",
                      'train_loss', 'val_loss', 'test_loss', 'train_AUC', 'val_AUC', 'test_AUC',
                      'train_AP', 'val_AP', 'test_AP', 'run', 'num_params']
            writer.writerow(header)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = [timestamp, dataset_path, relative_time, semi_transductive, heterogenous_msg_passing,
                batch_size, num_epochs, learning_rate, emb_dim, hidden_dim, time_dim, identity_dim,
                train_loss, val_loss, test_loss, train_AUC, val_AUC, test_AUC, train_ap, val_ap, test_ap,run, num_params]
        writer.writerow(data)

def append_to_csv_tgn(filename, dataset_path, batch_size, num_epochs, learning_rate, memory_dim, time_dim, embedding_dim, dropout, 
                  train_loss, val_AUC, test_AUC, val_ap, test_ap, run):
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['timestamp', "dataset_path", "batch_size", "num_epochs", "learning_rate", "memory_dim", "time_dim", "embedding_dim", "dropout", 
                      'train_loss', 'val_AUC', 'test_AUC', 'val_AP', 'test_AP', 'run']
            writer.writerow(header)

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        data = [timestamp, dataset_path, batch_size, num_epochs, learning_rate, memory_dim, time_dim, embedding_dim, dropout, 
                  train_loss, val_AUC, test_AUC, val_ap, test_ap, run]
        writer.writerow(data)

def create_mask_in_batches(edge_index, nodes, mask_batch_size, device):
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=device)
    for i in range(0, len(nodes), mask_batch_size):
        batch_nodes = nodes[i:i + mask_batch_size]
        batch_mask = ((edge_index[0].unsqueeze(1) == batch_nodes).any(dim=1)) | \
                     ((edge_index[1].unsqueeze(1) == batch_nodes).any(dim=1))
        mask |= batch_mask
    return mask

# Function to create a new Data object from the original data and a mask
def create_subset(data, mask):
    # Filter data using the mask
    subset = Data()
    for key, item in data:
        if key in ['node_id', 'node_type', 'x']:
            subset[key] = item
        elif key in ['edge_index']:
            subset[key] = item[:,mask]
        else:
            subset[key] = item[mask]
    return subset

def make_time_relative(data):
    unique_node_ids = data.node_id[data.node_type==0].unique()
    min_times_per_node = {node_id: data.time[data.edge_index[0,:] == node_id].min() for node_id in unique_node_ids}
    
    # Subtract the minimum time from each node's times
    for node_id, min_time in min_times_per_node.items():
        data.time[data.edge_index[0,:] == node_id] -= min_time
        data.time[data.edge_index[1,:] == node_id] -= min_time

    return data

class TemporalLoader:
    def __init__(self, data, start_time, device, batch_size=32, mask_batch_size=256):
        self.data = data
        self.time_order = torch.argsort(self.data.time)
        self.start_time = start_time
        self.start_index = torch.nonzero(self.data.time[self.time_order]>=start_time)[0]
        self.index = 0
        self.batch_size = batch_size
        self.length = len(data.edge_y) - self.start_index
        self.device = device
        self.mask_batch_size = mask_batch_size

    def __iter__(self):
        return self

    def __len__(self):
        return self.length // self.batch_size

    def reset(self):
        self.index = 0

    def __next__(self):
        if self.index + self.batch_size >= self.length:
            self.index = 0
            raise StopIteration

        # these edges are predicted
        mask = torch.zeros(self.data.edge_index.size(1), dtype=torch.bool).to(self.device)
        mask[self.time_order[self.index+self.start_index:self.index+self.start_index+self.batch_size]] = 1

        # but add neighbors that may be useful for the prediction = edges containing the same IDs in the time before
        before_mask = torch.zeros(self.data.edge_index.size(1), dtype=torch.bool).to(self.device)
        before_mask[self.time_order[:self.index+self.start_index]] = 1
        first_time = self.data.time[self.time_order[self.index+self.start_index]]

        edge_nodes = self.data.edge_index[:,mask].unique()
        neighbor_mask = create_mask_in_batches(self.data.edge_index, edge_nodes, self.mask_batch_size, self.device)

        mask |= (neighbor_mask & before_mask)
        
        batch = create_subset(self.data, mask)
        batch.edge_y[batch.time < first_time] = -1
        self.index += self.batch_size
        return batch

