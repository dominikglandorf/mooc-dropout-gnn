import torch
from torch import Tensor
from torch_geometric.nn.conv import GATConv
from torch.nn import Linear, Embedding

class TimeEncoder(torch.nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        self.time_lin = Linear(1, time_dim)

    def forward(self, t: Tensor):
        return self.time_lin(t.view(-1, 1)).cos()

class GATModel(torch.nn.Module):
    def __init__(self, emb_dim, hidden_channels, out_channels, edge_dim, time_enc, resource_ids, device, heterogenous_msg_passing=True, semi_transductive=True):
        super().__init__()
        self.conv1_user_to_resource = GATConv(emb_dim, hidden_channels, edge_dim=edge_dim)
        self.conv2_user_to_resource = GATConv(hidden_channels, out_channels, edge_dim=edge_dim)
        
        if heterogenous_msg_passing:
            self.conv1_resource_to_user = GATConv(emb_dim, hidden_channels, edge_dim=edge_dim)
            self.conv2_resource_to_user = GATConv(hidden_channels, out_channels, edge_dim=edge_dim)
            
        self.time_enc = time_enc
        if semi_transductive:
            self.mapping = {nid: i+1 for i, nid in enumerate(resource_ids)}
            self.embedding = Embedding(len(resource_ids)+1, emb_dim)
        else:
            self.mapping = {nid: 1 for i, nid in enumerate(resource_ids)}
            self.embedding = Embedding(2, emb_dim)
            
        self.device=device
        self.heterogenous_msg_passing = heterogenous_msg_passing

    def forward(self, node_id: Tensor, node_type: Tensor, edge_index: Tensor, edge_attr: Tensor, t: Tensor, edge_type: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        # edge_attr: Edge features
        # t: Timestamp of edges
        time_enc = self.time_enc(t)
        edge_attr = torch.cat([time_enc, edge_attr], dim=-1)
        

        node_ids = torch.tensor([self.mapping.get(nid, 0) for nid in node_id]).to(self.device)
        x = self.embedding(node_ids)

        mask = edge_type == 0
        
        if self.heterogenous_msg_passing:
            x_resources = self.conv1_user_to_resource(x, edge_index[:, mask], edge_attr[mask]).relu()
            x_users = self.conv1_resource_to_user(x, edge_index[:, ~mask], edge_attr[~mask]).relu()
            users = node_type == 0
            h = torch.zeros_like(x_users)
            h[users] = x_users[users]
            h[~users] = x_resources[~users]
        else:
            h = self.conv1_user_to_resource(x, edge_index, edge_attr).relu()

        if self.heterogenous_msg_passing:
            h_resources = self.conv2_user_to_resource(h, edge_index[:, mask], edge_attr[mask]).relu()
            h_users = self.conv2_resource_to_user(h, edge_index[:, ~mask], edge_attr[~mask]).relu()
            o = torch.zeros_like(h_users)
            o[users,:] = h_users[users,:]
            o[~users,:] = h_resources[~users,:]
        else: 
            o = self.conv2_user_to_resource(h, edge_index, edge_attr).relu()
        
        return o

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, time_enc):
        super().__init__()
        self.time_enc = time_enc
        dim = 2 * in_channels + time_enc.time_dim # 2*
        self.lin = Linear(dim, dim)
        self.lin_final = Linear(dim, 1)

    def forward(self, z_src, z_dst, t):
        time_enc = self.time_enc(t)
        input = torch.cat([z_src, z_dst, time_enc], dim=-1)
        h = self.lin(input)
        h = h.relu()
        
        return self.lin_final(h)