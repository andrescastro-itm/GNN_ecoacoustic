import networkx as nx

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx, to_undirected
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


