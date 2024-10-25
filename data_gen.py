from scipy.stats import multivariate_normal
import numpy as np


def gen_nm_cluster(center, var, dim=2, size=10):
    cov = np.eye(dim) * var
    cluster = multivariate_normal.rvs(center, cov, size)
    return cluster



def generate_tree_data(d, angle, dim=2, data_size=10, var=1):
    """ 
    dist:   distance between each parent-child cluster center
    angle:  angle between children and parent
    dim:    dimensionality of dataset
    """
    # The center of each cluster of data
    center = [[0.0, 0.0]]
    level_1_centers = [
        [0.0, d],    # above
        [d, 0.0],    # right
        [0.0, -d],   # below
        [-d, 0.0],   # left
    ]

    dsin = d * np.sin(angle)
    dcos = d * np.cos(angle)
    level_2_centers = [
        [-dsin, d+dcos], [dsin, d+dcos],    # top children
        [d+dcos, dsin], [d+dcos, -dsin],    # right children
        [-dsin, -d-dcos], [dsin, -d-dcos],  # bot children
        [-d-dcos, dsin], [-d-dcos, -dsin],  # left children
    ]

    centers = np.array(center + level_1_centers + level_2_centers)

    # Generate a data cluster around each center
    data_size = data_size
    data = np.array([gen_nm_cluster(center, var, dim, data_size) for center in centers])
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    
    # Generate labels for each cluster
    dataLabels = np.array([[label for _ in range(data_size)] for label in range(centers.shape[0])])
    dataLabels = dataLabels.reshape((dataLabels.shape[0] * dataLabels.shape[1]))

    return data, dataLabels



def generate_linear_hierarchical_data(n_samples, n_clusters, n_dim, start, end):
    """ 
    Generate a "hierarchical" dataset where we basically generate clusters from "left" to "right" 
    """
    # n_samples must be a integer multiple of n_clusters
    points_per_cluster = n_samples
    cluster_distances = np.abs(end - start) / n_clusters                            # distance between each cluster
    cluster_centers = [start + i * cluster_distances for i in range(n_clusters)]    # centers of each cluster

    # Generate data
    data = []
    for center in cluster_centers:
        cluster_center = np.ones(n_dim) * center
        cov = (cluster_distances / 3) 
        # cov = 1
        cluster_cov = np.identity(n_dim) * cov
        #cluster_cov = np.identity(n_dim)
        cluster_data = multivariate_normal.rvs(mean=cluster_center, cov=cluster_cov, size=points_per_cluster)
        data.append(cluster_data)

    # Generate labels
    labels = []
    for i in range(n_clusters):
        labels = labels + [i for _ in range(points_per_cluster)]

    return np.array(data), np.array(labels)



def generate_uniform_clusters(n_samples, n_dim, b1, b2):
    # n_samples = 500
    # n_dim = 10
    # b1 = 1000000
    # b2 = 100

    # Bounds for first distribution
    lb_1 = np.ones(n_dim) * (-b1)
    ub_1 = np.ones(n_dim) * (-b2)
    widths_1 = ub_1 - lb_1 

    # Bounds for 2nd distribution
    lb_2 = np.ones(n_dim) * b2
    ub_2 = np.ones(n_dim) * b1
    widths_2 = ub_2 - lb_2

    # Draw samples 
    samples_1 = uniform.rvs(loc=lb_1, scale=widths_1, size=(n_samples, n_dim))
    samples_2 = uniform.rvs(loc=lb_2, scale=widths_2, size=(n_samples, n_dim))

    dataX = np.concatenate((samples_1, samples_2), axis=0)
    labels = [0 for _ in range(n_samples)] + [1 for _ in range(n_samples)]
    dataLabels = np.array(labels)

    return dataX, dataLabels


def generate_hierarchical_D_V(cluster_size, n_children, depth):
    """
    Generates a D (distance) and V (affinity) matrix for a tree-like/hierarchical dataset
    where we have 'n_children' children per node, with a depth of 'depth'

    depth = 0 means only a root, etc..
    """
    tree_nodes = np.sum([np.power(n_children, p) for p in range(depth)])
    size = cluster_size * tree_nodes
    D = np.zeros((size, size))
    V = np.zeros((size, size))

    # Center cluster


    
    return D, V









"""
Below code that generates a custom D, V matrix where the distances 
reflect a tree-like hierarchical data structure
"""
from queue import Queue
import numpy as np

def find_children(node: int, n_children: int, n_nodes: int):
    """ 
    node:               The current node
    n_children:         The number of children per node
    n_nodes:            Total nr. of nodes
    total_n_children:   Total nr. of children for this node

    Find the children of this node, breath-first style
    """
    children = Queue(maxsize=n_nodes)        # list of children to return
    q = Queue(maxsize=n_nodes)               # queue of children to explore further   
    q.put(node)                              # Initialize with root node

    while not q.empty():
        curr_node = q.get()                      # current node were exploring
                     
        for i in range(1, n_children + 1):       # For each children, add them
            child = n_children * curr_node + i   # idx of child node
            if child >= n_nodes: 
                break

            children.put(child)                 
            q.put(child)

    return children 


def find_depth(node: int, n_children: int):
    """ 
    Find the depth of node 'node'
    The depth is the number of divisions by n_children we need to get to the root (node 0)
    If node is a multiple of n_children, then (node - 1) / n_children
    """
    depth = 0
    while node > 0:
        depth += 1                                          

        if node % n_children == 0:
            node = int(np.floor((node - 1) / n_children))
        else:
            node = int(np.floor(node / n_children))
    
    return depth


def node_distances(node: int, D: np.array, dist: int, n_children: int, children: Queue):
    """ 
    Computes distances between node
    Assuming breath-first construction of distance matrid
    """
    # Computes distances from current node to all children
    start_depth = find_depth(node, n_children)          # depth of current (parent) node
    while not children.empty():
        target = children.get()
        tgt_depth = find_depth(target, n_children)      # depth of target node
        depth = tgt_depth - start_depth                 # relative depth
        
        # depth of target node relative to (current) node * dist
        D[node, target] = depth * dist


def test_non_clustered_tree():
    """ 
    Tests the code that generates a tree with clusters of 1 node per vertex
    """
    dist = 10
    depth = 2
    n_children = 2
    n_nodes = sum(np.power(n_children, d) for d in range(depth + 1))
    D = np.zeros((n_nodes, n_nodes))

    # Compute distances from each node to every other node
    for n in range(n_nodes):
        # Compute distances n to non-children nodes
        # Note that this is just the distance of (parent to other nodes) + (n to parent)
        parent = int(np.floor((n - 1) / n_children)) if n % n_children == 0 else int(np.floor(n / n_children))
        if parent >= 0:
            D[n, (n + 1):] = dist + D[parent, (n + 1):]

        # Compute distances n to its children
        children = find_children(n, n_children, n_nodes)
        node_distances(n, D, dist, n_children, children)

    # Reflect distances across diagional to complete distance matrix
    D = D + D.T

    return D


def generate_Tree_D(cluster_size: int, n_children: int, n_nodes: int, mu: float, sigma: float, dist):
    """ 
    Generata Distance D and Affinity V matrix for a tree-like hierarchical dataset
    with clusters around each node/vertex of the tree
    """
    # (0) Initialize distance matrix
    D = np.zeros((n_nodes, cluster_size, n_nodes, cluster_size))

    # (1) Initialize D for inbetween cluster distances
    for n in range(n_nodes):
        samples = sigma * np.random.randn(cluster_size, cluster_size) + mu
        samples = np.abs(samples)                                               # take abs value to get positive distances
        distances = np.triu(samples, 1)             
        distances = distances + distances.T                                     # distances are squared for use in gaussian probabilities
        D[n, :, n, :] = distances

    # (2) Compute distances between node clusters
    for n in range(n_nodes - 1):
        # Compute distances to parents children (if we have one)
        parent = int(np.floor((n - 1) / n_children)) if n % n_children == 0 else int(np.floor(n / n_children))
        if parent >= 0:
            dist = D[parent, :, n, :]                                           # Distance from (me to parent) == (parent to me)
            dist_to_parent = np.stack([dist] * (n_nodes - n - 1), axis=1)       # We must match dimensions of parent_dists
            parent_dists = D[parent, :, (n + 1):, :]                            # Distances between parent, and non-children nodes of us 
            D[n, :, (n + 1):, :] = dist_to_parent + parent_dists

        # Compute distances to children
        children = find_children(n, n_children, n_nodes)
        start_depth = find_depth(n, n_children)                                 # depth of current (parent) node
        while not children.empty():
            target = children.get()
            tgt_depth = find_depth(target, n_children)                          # depth of target node
            depth = tgt_depth - start_depth                                     # relative depth
            samples = sigma * np.random.randn(cluster_size, cluster_size) + mu  # initial samples for cluster distances
            samples = np.abs(samples)                                           # take abs value to get positive distances
            distances = np.triu(samples, 1)                                     # upper triangulate distance matrix
            distances = distances + distances.T                                 # turn into symmetric distance matrix
            D[n, :, target, :] = (depth * dist) + samples                       # add tree-path distance to sampled distances


    # (3) Reshape and return
    D = D.reshape((n_nodes * cluster_size, n_nodes * cluster_size))
    D = np.triu(D, 1)
    D = D + D.T
    return D


def generate_Tree_V(D: np.ndarray, mean: float, var: float):
    """ 
    Compute the affinity (probability) matrix given a distance (D) matrix
    """
    V = np.exp(-np.square(D - mean) / (2 * var))    # apply gaussian probability function onto squared distances
    np.fill_diagonal(V, 0)                          # reset diagonal entries to 0
    V = V / np.sum(V)                               # normalize so we get probabilities
    return V