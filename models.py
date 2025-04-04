import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv

class MatrixGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(MatrixGCN, self).__init__()
        self.conv1d = torch.nn.Conv1d(1, 64, 24, stride=24)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)

    def forward(self, x, edge_index):
        x = x.transpose(1,2).flatten(1)  # Flatten the matrix features
        x = self.conv1d(x.unsqueeze(1))
        x = x.view(x.size(0),-1)
        #print(x.shape)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class MatrixGAT(torch.nn.Module):
  def __init__(self, in_channels, hidden_channels, embedding_dim, heads=1):
    super().__init__()
    self.conv1d = torch.nn.Conv1d(1, 64, 24, stride=24)
    self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
    self.gat2 = GATv2Conv(hidden_channels*heads, embedding_dim, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.001,
                                      weight_decay=5e-4)

  def forward(self, x, edge_index):
    x = x.transpose(1,2).flatten(1)  # Flatten the matrix features
    x = self.conv1d(x.unsqueeze(1))
    x = x.view(x.size(0),-1)
    # x = F.dropout(x, p=0.5, training=self.training)
    x = self.gat1(x, edge_index)
    x = F.relu(x)
    # x = F.dropout(h, p=0.5, training=self.training)
    x = self.gat2(x, edge_index)

    return x