import numpy as np

def get_cloud_center(cloud):
    #get center of the cloud
    x_mean = sum(cloud[:,0]) / len(cloud[:,0])
    y_mean = sum(cloud[:,1]) / len(cloud[:,1])
    z_mean = sum(cloud[:,2]) / len(cloud[:,2])
    cloud_center = np.array([x_mean, y_mean, z_mean])
    return cloud_center

#fit plane according to Ax + By + Cz + D = 0
#find plane according to the least squares solution
#solve for Z = -Ax - By - D
#General form of solution:
#[Ax By D] [t] = [Z]

def fitPlaneLstsq(cloud):
    #create A matrix (X, Y and D)
    A = np.column_stack((cloud[:,0],cloud[:,1],cloud[:,3]))
    b = cloud[:,2]
    return np.linalg.lstsq(A, b)[0]

#Alternate fit plane according to the least squares solution
#Solve Ax + By + Cz  = -D
def fitPlaneLstsqAlt(cloud):
    #create A matrx (X, Y and Z)
    A = np.column_stack((cloud[:,0],cloud[:,1],cloud[:,2]))
    b = cloud[:,3]
    return np.linalg.lstsq(A, b)[0]

#fit plane, using the SVD
def fitPlaneSVD(cloud):
    new_cloud = np.column_stack((cloud[:,0], cloud[:,1], cloud[:,2]))

    #get SVD
    [U, S, V] = np.linalg.svd(new_cloud)
    V = V.conj().T
    return V[-1]

#project cloudpoints. Only for least squares at the moment
def project_cloud2(cloud, plane):
    new_cloud = np.column_stack((cloud[:,0],cloud[:,1],cloud[:,2]))
    projected_cloud = []
    for row in new_cloud:
        d = (np.dot(row, plane)) / np.sqrt(np.linalg.norm(plane))
        normalized_plane = plane / np.linalg.norm(plane)
        proj =d*normalized_plane
        projected_cloud.append(row - proj)
    return np.asarray(projected_cloud)

def get_d(plane_normal, cloud):
    #Ax + By + Cz = -D
    cloud_center = get_cloud_center(cloud)
    return -plane_normal.dot(cloud_center)

#calulate the direction of the line and return the normalized direction
def plane_intersection(plane1, plane2, cloud1, cloud2):
    new_cloud1 = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    cloud_center1 = get_cloud_center(new_cloud1)
    plane1_d  = np.dot(cloud_center1, plane1)

    new_cloud2 = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    cloud_center2 = get_cloud_center(new_cloud2)
    plane2_d = np.dot(cloud_center2, plane2)

    plane3 = np.cross(plane1, plane2)

    A = np.row_stack((plane1, plane2, plane3))
    b = np.asarray([plane1_d, plane2_d, 0])
    point_on_line = np.linalg.solve(A, b)

    direction = np.cross(plane1, plane2)
    return direction/np.linalg.norm(direction), point_on_line

def proj_cloud(cloud, plane_normal):
    #get center of plane
    new_cloud = np.column_stack((cloud[:,0],cloud[:,1],cloud[:,2]))
    cloud_center = get_cloud_center(new_cloud)
    plane =  plane_normal / np.linalg.norm(plane_normal)
    proj_cloud = []

    for point in new_cloud:
        v = point-cloud_center
        dist = v.dot(plane)
        proj_cloud.append(point-(dist*plane))
    return np.asarray(proj_cloud)

def project_cloud(cloud, plane):
    new_cloud = np.column_stack((cloud[:,0],cloud[:,1],cloud[:,2]))
    #normalize plane
    plane =  plane / np.linalg.norm(plane)
    vnorm = np.square(plane)
    normal_vector = plane / np.linalg.norm(plane)
    point_in_plane = plane / vnorm

    points_from_point_in_plane = new_cloud - point_in_plane
    proj_onto_normal_vector = np.dot(points_from_point_in_plane,
                                     normal_vector)
    proj_onto_plane = (points_from_point_in_plane -
                       proj_onto_normal_vector[:, None]*normal_vector)

    return point_in_plane + proj_onto_plane

#assumes direction vector is unit vector
def proj_cloud_line(cloud, direction, offset):
    #(x,y,z).(a,b,c) (a,b,c) = (ax+by+cz)(a,b,c)
    new_cloud = np.column_stack((cloud[:,0], cloud[:,1], cloud[:,2]))
    proj_cloud = []
    for point in new_cloud:
        projection = (np.dot(point, direction)*direction) + offset
        proj_cloud.append(projection)
    return np.asarray(proj_cloud)

#find the location of the joint
#take the center of the cloud
#assume the point closest to this is the location of the joint
def find_joint_location(cloud1, cloud2, line1, line2):
    cloud1_ = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    cloud2_ = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    center_cloud1 = get_cloud_center(cloud1_)
    center_cloud2 = get_cloud_center(cloud2_)
    average_c1c2 = (center_cloud1 + center_cloud2) / 2

    min_dist = 1000
    min_dist_point = [0,0,0]
    for point in line1:
        dist = np.linalg.norm(point-average_c1c2)
        if dist < min_dist:
            min_dist = dist
            min_dist_point = point

    for point in line2:
        dist = np.linalg.norm(point-average_c1c2)
        if dist < min_dist:
            min_dist = dist
            min_dist_point = point

    return min_dist_point
    
def custom_joint_loc(cloud1, cloud2):
    cloud1_normal = fitPlaneLstsqAlt(cloud1)
    cloud2_normal = fitPlaneLstsqAlt(cloud2)

    intersection_direction, offset = plane_intersection(cloud1_normal, cloud2_normal, cloud1, cloud2)
    proj_cloud1_line = proj_cloud_line(cloud1, intersection_direction, offset)
    proj_cloud2_line = proj_cloud_line(cloud2, intersection_direction, offset)

    joint_loc = find_joint_location(cloud1, cloud2, proj_cloud1_line, proj_cloud2_line)
    return joint_loc
