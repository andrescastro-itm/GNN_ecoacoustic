import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv


class MatrixGATVAE_MT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_dim, num_nodes, num_classes, heads=1):
        super(MatrixGATVAE_MT, self).__init__()
        
        # Encoder components
        self.conv1d = torch.nn.Conv1d(1, 64, 24, stride=24)
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads)
        
        # Mean and log variance layers for the latent distribution
        self.bn = torch.nn.BatchNorm1d(hidden_channels)
        self.fc_mu = GATv2Conv(hidden_channels*heads, latent_dim, heads=1)
        self.fc_logvar = GATv2Conv(hidden_channels*heads, latent_dim, heads=1)
        
        # Decoder components for node features
        self.decoder_fc1 = torch.nn.Linear(latent_dim, hidden_channels)
        self.num_rec = int(in_channels*24/64)
        self.decoder_fc2 = torch.nn.Linear(hidden_channels, self.num_rec)
        
        # Decoder components for adjacency matrix
        self.num_nodes = num_nodes
        self.adj_decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_nodes)
        )

        # Clasificador
        self.classifier = torch.nn.Sequential(
            # torch.nn.Linear(latent_dim, num_classes),
            torch.nn.Linear(latent_dim, max(8,int(latent_dim/4))),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),  # Agregamos dropout para mejor generalización
            torch.nn.Linear(max(8,int(latent_dim/4)), num_classes),
        )
        
    def encode(self, x, edge_index):
        # # Process input features
        x = x.transpose(1, 2).flatten(1)  # Flatten the matrix features
        x = self.conv1d(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        
        # GCN encoding
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        
        # Get latent distribution parameters
        x = self.bn(x)
        mu = self.fc_mu(x, edge_index)
        logvar = self.fc_logvar(x, edge_index)
        
        return mu
    
    # def reparameterize(self, mu, logvar):
    #     return mu + torch.randn_like(logvar) * torch.exp(logvar)
    
    # def decode(self, z):
    #     # Decode node features from latent space
    #     h = F.relu(self.decoder_fc1(z))
    #     node_reconstruction = torch.sigmoid(self.decoder_fc2(h))
        
    #     # Decode adjacency matrix from latent space
    #     adj_logits = self.adj_decoder(z)
    #     # Create adjacency predictions - a matrix where each row i contains scores for edges from node i to all nodes
    #     adj_matrix = torch.sigmoid(torch.matmul(adj_logits, adj_logits.transpose(0, 1)))
        
    #     return node_reconstruction, adj_matrix
    
    def forward(self, inputs):
        # Encode to get latent distribution
        mu = self.encode(inputs[0], inputs[1])
        # mu, logvar = torch.tanh(mu), torch.tanh(logvar)
        # Sample from the latent distribution
        # z = s/elf.reparameterize(mu, logvar)
        # print(z)
        # Decode to get reconstructions
        # node_reconstruction, adj_reconstruction = self.decode(z)
        # class_logits = self.classifier(z)#mu
        
        return mu
    
    # def edge_index_to_adj_matrix(self, edge_index, num_nodes):
    #     """Convert edge_index to dense adjacency matrix"""
    #     adj_matrix = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    #     adj_matrix[edge_index[0], edge_index[1]] = 1.0
    #     return adj_matrix
    
    # def loss_function(self, node_reconstruction, adj_reconstruction, class_logits, labels, x_original, edge_index, mu, logvar, alpha=1.0, beta=1.0):
    #     # Node feature reconstruction loss (MSE)
    #     # print(f"true reconstructed= {node_reconstruction.shape}, original={x_original.transpose(1, 2).flatten(1).shape}")
    #     feature_loss = F.mse_loss(node_reconstruction, x_original.transpose(1, 2).flatten(1))
        
    #     # Adjacency matrix reconstruction loss (BCE)
    #     true_adj = self.edge_index_to_adj_matrix(edge_index, self.num_nodes)
        
    #     # Pérdida de clasificación
    #     class_loss = F.cross_entropy(class_logits, labels)
    #     # Convert predictions to log probabilities
    #     # log_prediction = F.log_softmax(class_logits, dim=1)

    #     # KL divergence loss (for soft labels)
    #     # class_loss = F.kl_div(log_prediction, labels, reduction='batchmean')

    #     # Binary cross entropy for adjacency matrix
    #     # We can add class weights if the graph is sparse
    #     # print(f"{adj_reconstruction.double()=}")
    #     # print(f"{true_adj.double()=}")
    #     adj_loss = F.mse_loss(adj_reconstruction.double(), true_adj.double())
        
    #     # KL Divergence loss
    #     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    #     kl_loss = kl_loss/labels.shape[0]
        
    #     # Total loss with weighting factors
    #     total_loss = feature_loss + alpha * adj_loss + beta * kl_loss + class_loss
        
    #     return total_loss, feature_loss, alpha*adj_loss, beta*kl_loss, class_loss
    