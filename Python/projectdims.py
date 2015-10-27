import numpy as np


def project_cloud3dto2d(cloud):
    cloud_2d = []
    for point in cloud:
        point = np.asarray(point)
        point = np.delete(point, 3)
        cloud_2d.append(project3dto2d(point))
    return cloud_2d


#project 3d point to 2d
def project3dto2d(point):
    #Kinect intrinsics as given in PCL
    kinect_intrinsics = np.asarray([[525.0, 0.0, 319.5], [0.0, 525, 239.5], [0.0, 0.0, 1.0]])
    m = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    proj_matrix = np.dot(kinect_intrinsics, m)
#    point_3d = point_3d + Eigen::Vector4f (0, 0, 0, 1);  //??
    projected_point = np.dot(proj_matrix, point)
    projected_point = projected_point / projected_point [2]
    projected_point = np.delete(projected_point, 2)
    return projected_point;
