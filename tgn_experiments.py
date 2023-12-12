import os.path as osp

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
)
from tqdm import tqdm

from helpers import append_to_csv_tgn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

# hyperparameters
batch_size = 1024
num_epochs = 5
learning_rate = 0.001
memory_dim = 100
time_dim = 100
embedding_dim = 100
dropout = 0.1

# experimental setting
dataset_path = "data/junyi/graph_sub.pt"#"data/act-mooc/graph.pt" # 
data = torch.load(dataset_path)
relative_time = False

# data preparation
del data[('resource', 'rev_accesses', 'user')]
data = data.to_homogeneous()
if relative_time:
    unique_node_ids = data.node_id[data.node_type==0].unique()
    min_times_per_node = {node_id: data.time[data.edge_index[0,:] == node_id].min() for node_id in unique_node_ids}
    
    # Subtract the minimum time from each node's times
    for node_id, min_time in min_times_per_node.items():
        data.time[data.edge_index[0,:] == node_id] -= min_time
        data.time[data.edge_index[1,:] == node_id] -= min_time

data_temp = TemporalData(
    src=data.edge_index[0,:].to(torch.long),
    dst=data.edge_index[1,:].to(torch.long),
    t=data.time.to(torch.long),
    msg=data.edge_attr.to(torch.float),
    y=data.edge_y.to(torch.long)
)
data = data_temp

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:
data = data.to(device)

train_data, val_data, test_data = data.train_val_test_split(val_ratio=0.15, test_ratio=0.15)

train_loader = TemporalDataLoader(train_data, batch_size=batch_size, neg_sampling_ratio=1.0)
val_loader = TemporalDataLoader(val_data, batch_size=batch_size, neg_sampling_ratio=1.0)
test_loader = TemporalDataLoader(test_data, batch_size=batch_size, neg_sampling_ratio=1.0)
neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

import numpy as np
time_array = data.t.cpu().numpy()
quantile_70 = np.quantile(time_array, 0.7)
quantile_85 = np.quantile(time_array, 0.85)
for batch in tqdm(train_loader):
    assert batch.t.min() <= quantile_70
for batch in tqdm(val_loader):
    assert batch.t.min() >= quantile_70
    assert batch.t.max() <= quantile_85
for batch in tqdm(test_loader):
    assert batch.t.min() >= quantile_85

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=dropout, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

criterion = torch.nn.BCEWithLogitsLoss()

def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        positive_edges = batch.edge_index[:, batch.y == 1]
        pos_out = link_pred(z[assoc[positive_edges[0]]], z[assoc[positive_edges[1]]])
        loss = criterion(pos_out, torch.ones_like(pos_out))
        
        negative_indices = torch.nonzero(batch.y == 0).squeeze()
        negative_indices = negative_indices[torch.randperm(negative_indices.size(0))][:positive_edges.size(1)]
        negative_edges = batch.edge_index[:, negative_indices]
        neg_out = link_pred(z[assoc[negative_edges[0]]], z[assoc[negative_edges[1]]])
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    true_labels = torch.Tensor().to(device)
    predictions = torch.Tensor().to(device)
    for batch in tqdm(loader):
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device), data.msg[e_id].to(device))
        
        out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])

        y_pred = out.sigmoid().cpu()
        y_true = batch.y
        
        predictions = torch.cat((predictions, y_pred.detach()), dim=0)
        true_labels = torch.cat((true_labels, batch.y), dim=0)

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    auc = roc_auc_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    ap = average_precision_score(true_labels.cpu().numpy(), predictions.cpu().numpy())
    #return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())
    return ap, auc

for i in range(0, 3):
    memory = TGNMemory(data.num_nodes, data.msg.size(-1), memory_dim, time_dim,
                   message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
                   aggregator_module=LastAggregator(),).to(device)

    gnn = GraphAttentionEmbedding(in_channels=memory_dim, out_channels=embedding_dim, msg_dim=data.msg.size(-1), time_enc=memory.time_enc,).to(device)
    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(set(memory.parameters()) | set(gnn.parameters()) | set(link_pred.parameters()), lr=learning_rate)
    for epoch in range(0, num_epochs):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        val_ap, val_auc = test(val_loader)
        test_ap, test_auc = test(test_loader)
        print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')
    
    append_to_csv_tgn("results_tgn.csv", dataset_path, batch_size, num_epochs, learning_rate, memory_dim, time_dim, embedding_dim, dropout,
                  loss, val_auc, test_auc, val_ap, test_ap, i)


