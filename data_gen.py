from scipy.stats import multivariate_normal
import numpy as np



def gen_nm_cluster(center, var, dim=2, size=10):
    cov = np.eye(dim) * var
    cluster = multivariate_normal.rvs(center, cov, size)
    return cluster



def generate_tree_data(d, angle, dim=2, var=1):
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
    data_size = 10
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