import numpy as np
import cloud_linalg
from sklearn.cluster import MeanShift, estimate_bandwidth

def perform_meanshift(cloud1, cloud2):
    # The following bandwidth can be automatically detected using
    full_cloud = np.vstack((cloud1, cloud2))
    bandwidth = estimate_bandwidth(full_cloud, quantile=0.2, n_samples=500)

    # Create MeanShift instance
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # Perform mean shift
    ms.fit(full_cloud)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    n_clusters_ = labels.max()+1

    return full_cloud, cluster_centers, n_clusters_, labels
    
# load a cloud and return the new clusters
def load_cloud(nr, cl1, cl2, path="C:/Users/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal"):
    cloud1 = np.loadtxt(str(path + '/' + str(nr) + '_' + cl1 + '.txt'), delimiter=',')
    cloud2 = np.loadtxt(str(path + '/' + str(nr) + '_' + cl2 + '.txt'), delimiter=',')
    A = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    B = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    #find cluster centers
    A_center = cloud_linalg.get_cloud_center(A)
    B_center = cloud_linalg.get_cloud_center(B)
    full_cloud, new_cluster_centers = perform_meanshift(A, B)
    
    return [A_center, B_center], assign_to_cluster(full_cloud, new_cluster_centers), new_cluster_centers
    
def assign_to_cluster(cloud, cluster_centers):
    nr_clouds = len(cluster_centers)
    new_clouds = {}
    for i in range(0, nr_clouds):
        new_clouds[i] = []
        
    for point in cloud:
        shortest_dist = 1000
        shortest_dist_loc = -1
        for c in range(0, nr_clouds):
            dist = np.linalg.norm(point-cluster_centers[c])
            if dist < shortest_dist:
                shortest_dist_loc = c
                shortest_dist = dist
        new_clouds[shortest_dist_loc].append(point)
        
    return new_clouds
