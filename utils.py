import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import scipy.sparse

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, to_undirected, is_undirected
from torch_geometric.nn import knn_graph, radius_graph


def is_connected(data):
    # Convert PyTorch Geometric graph to NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Check if the graph is connected
    return nx.is_connected(G)

def edge_creation_nodeinfo(node_information, mode = 'knn', k_neigh = 7, radius = 530_000):
    dataset = []
    for i in range(node_information.shape[0]):
        if mode == 'knn':
            edge_index = knn_graph(node_information[i].flatten(1), k = k_neigh)
        elif mode == 'radius':
            edge_index = radius_graph(node_information[i].flatten(1), r=radius)
        edge_index = to_undirected(edge_index)
        graph = Data(x = node_information[i], edge_index = edge_index)
        dataset.append(graph)

    return dataset

def edge_creation_coverinfo(cover_information, node_information,mode = 'knn', k_neigh = 7, radius = 0.5):
    dataset = []
    if mode == 'knn':
        edge_index = knn_graph(cover_information, k = k_neigh)
    elif mode == 'radius':
        edge_index = radius_graph(cover_information, r=radius)
    for i in range(node_information.shape[0]):
        edge_index = to_undirected(edge_index)
        graph = Data(x = node_information[i], edge_index = edge_index)
        dataset.append(graph)

    return dataset

def haversine_distance_matrix(lats, lons):
    R = 6371  # Earth's radius in kilometers
    
    # Convert to radians
    lats, lons = np.radians(lats), np.radians(lons)
    
    # Compute differences
    dlat = lats[:, np.newaxis] - lats
    dlon = lons[:, np.newaxis] - lons
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lats[:, np.newaxis]) * np.cos(lats) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def adjacency_to_edge_index(adjacency_matrix):
    """
    Convert an adjacency matrix to an edge_index tensor for PyTorch Geometric.
    
    Args:
    adjacency_matrix (np.ndarray): A 2D numpy array representing the adjacency matrix.
    
    Returns:
    torch.LongTensor: A 2xN tensor representing the edge_index.
    """
    # Find the indices of the non-zero elements
    edges = np.array(np.nonzero(adjacency_matrix))
    
    # Convert to PyTorch tensor and ensure Long datatype
    edge_index = torch.tensor(edges, dtype=torch.long)
    
    return edge_index

def edge_creation_geoDistance(node_information, df_map, dist_thres=1.0):
    lats = df_map.latitude_IG.values
    lons = df_map.longitud_IG.values

    # Compute the distance matrix
    distance_matrix = haversine_distance_matrix(lats, lons)

    adjacency_matrix = (distance_matrix <= dist_thres).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)  # Set diagonal to 0

    dataset = []
    edge_index = adjacency_to_edge_index(adjacency_matrix)
    for i in range(node_information.shape[0]):
        edge_index = to_undirected(edge_index)
        graph = Data(x = node_information[i], edge_index = edge_index)
        dataset.append(graph)

    return dataset, adjacency_matrix

def plot_distance_matrix_heatmap(matrix, title="Distance Matrix Heatmap"):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap="YlOrRd", square=True, cbar_kws={'label': 'Distance (km)'})
    plt.title(title)
    plt.xlabel("Coordinate Index")
    plt.ylabel("Coordinate Index")
    plt.show()

def edge_index_to_adjacency(edge_index):
    """
    Convert an edge_index to an adjacency matrix.
    
    Args:
    edge_index (np.ndarray): A 2D numpy array representing the edges.

    Returns:
    np.ndarray: A NxN tensor representing the adjacency matrix.
    """
    # Convert to adjacency matrix
    num_nodes = edge_index.max().item() + 1
    adj_matrix = scipy.sparse.coo_matrix((torch.ones(edge_index.size(1)), edge_index), shape=(num_nodes, num_nodes)).toarray()
    
    return adj_matrix





