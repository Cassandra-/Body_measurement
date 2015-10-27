import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import argparse
import os
import sys
from sklearn.cluster import MeanShift, estimate_bandwidth

#load pointcloud data
#is expected to be stored as csv:
#X1,Y1,Z1,1
#X2,Y2,Z2,1
#etc...
def loadfile(filename):
    return np.loadtxt(filename, delimiter=',')

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
    #create A matrx (X, Y and D)
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

def perform_meanshift(cloud1, cloud2):
    # The following bandwidth can be automatically detected using
    full_cloud = np.vstack((cloud1, cloud2))
    bandwidth = estimate_bandwidth(full_cloud, quantile=0.2, n_samples=500)

    # Create MeanShift instance
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # Perform mean shift
    ms.fit(full_cloud)

    return full_cloud, ms.cluster_centers_

#plot everything
def plot_results(cloud1, cloud2, proj_plane1, proj_plane2, arm_line_proj, forearm_line_proj, joint_loc):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # plot points in 3D
    ax.plot(cloud1[:,0],cloud1[:,1],cloud1[:,2],'o', color='blue')
    ax.plot(cloud2[:,0],cloud2[:,1],cloud2[:,2],'o', color='green')

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')
    ax2.plot(cloud1[:,0],cloud1[:,1],cloud1[:,2],'o', color='blue')
    ax2.plot(cloud2[:,0],cloud2[:,1],cloud2[:,2],'o', color='green')
    ax2.plot(proj_plane1[:,0],proj_plane1[:,1],proj_plane1[:,2],'o', color='red')
    ax2.plot(proj_plane2[:,0],proj_plane2[:,1],proj_plane2[:,2],'o', color='magenta')

    fig5 = plt.figure()
    ax5 = fig5.gca(projection='3d')

    ax5.plot(cloud1[:,0],cloud1[:,1],cloud1[:,2],'o', color='blue')
    ax5.plot(cloud2[:,0],cloud2[:,1],cloud2[:,2],'o', color='green')
    ax5.plot(arm_line_proj[:,0], arm_line_proj[:,1], arm_line_proj[:,2], 'o', color='red')
    ax5.plot(forearm_line_proj[:,0], forearm_line_proj[:,1], forearm_line_proj[:,2], 'o', color='magenta')
    ax5.plot([joint_loc[0]], [joint_loc[1]], [joint_loc[2]], 'o', color='yellow')
    plt.show()

#draw a line between two points
#accepts tuples
def drawlimb(point1, point2, ax2=None):
    ax2.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

def drawlimb_2d(point1, point2, ax2=None):
    ax2.plot([point1[0], point2[0]], [point1[1], point2[1]])

def drawskeleton_2d(filename):
    fig3 = plt.figure()
    ax3 = fig3.gca()

    lines = [line.rstrip('\n').split(',') for line in open(filename)]
    joints = []
    for line in lines:
        if len(line) == 3:
            line = map(float, line)
            line = project3dto2d(line)
            ax3.plot([line[0]],[line[1]],'o')
            joints.append(line)

    #draw limbs
    drawlimb_2d(joints[10], joints[20], ax2=ax3)
    drawlimb_2d(joints[10], joints[19], ax2=ax3)
    drawlimb_2d(joints[10], joints[26], ax2=ax3)
    drawlimb_2d(joints[10], joints[25], ax2=ax3)
    drawlimb_2d(joints[26], joints[23], ax2=ax3)
    drawlimb_2d(joints[25], joints[24], ax2=ax3)
    drawlimb_2d(joints[21], joints[22], ax2=ax3)

    drawlimb_2d(joints[23], joints[8], ax2=ax3)
    drawlimb_2d(joints[24], joints[9], ax2=ax3)
    drawlimb_2d(joints[9], joints[2], ax2=ax3)
    drawlimb_2d(joints[8], joints[6], ax2=ax3)
    drawlimb_2d(joints[2], joints[0], ax2=ax3)
    drawlimb_2d(joints[6], joints[4], ax2=ax3)

    drawlimb_2d(joints[26], joints[12], ax2=ax3)
    drawlimb_2d(joints[12], joints[14], ax2=ax3)
    drawlimb_2d(joints[25], joints[16], ax2=ax3)
    drawlimb_2d(joints[16], joints[18], ax2=ax3)
    drawlimb_2d(joints[19], joints[21], ax2=ax3)
    drawlimb_2d(joints[20], joints[22], ax2=ax3)

    drawlimb_2d(joints[26], joints[9], ax2=ax3)
    drawlimb_2d(joints[25], joints[8], ax2=ax3)
    drawlimb_2d(joints[8], joints[9], ax2=ax3)
    plt.show()

#draw skeleton
def drawskeleton(filename):
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')

    lines = [line.rstrip('\n').split(',') for line in open(filename)]
    joints = []
    for line in lines:
        if len(line) == 3:
            line = map(float, line)
            ax2.plot([line[0]],[line[1]],[line[2]],'o')
            joints.append(line)

    #draw limbs
    drawlimb(joints[10], joints[20], ax2=ax2)
    drawlimb(joints[10], joints[19], ax2=ax2)
    drawlimb(joints[10], joints[26], ax2=ax2)
    drawlimb(joints[10], joints[25], ax2=ax2)
    drawlimb(joints[26], joints[23], ax2=ax2)
    drawlimb(joints[25], joints[24], ax2=ax2)
    drawlimb(joints[21], joints[22], ax2=ax2)

    drawlimb(joints[23], joints[8], ax2=ax2)
    drawlimb(joints[24], joints[9], ax2=ax2)
    drawlimb(joints[9], joints[2], ax2=ax2)
    drawlimb(joints[8], joints[6], ax2=ax2)
    drawlimb(joints[2], joints[0], ax2=ax2)
    drawlimb(joints[6], joints[4], ax2=ax2)

    drawlimb(joints[26], joints[12], ax2=ax2)
    drawlimb(joints[12], joints[14], ax2=ax2)
    drawlimb(joints[25], joints[16], ax2=ax2)
    drawlimb(joints[16], joints[18], ax2=ax2)
    drawlimb(joints[19], joints[21], ax2=ax2)
    drawlimb(joints[20], joints[22], ax2=ax2)

    drawlimb(joints[26], joints[9], ax2=ax2)
    drawlimb(joints[25], joints[8], ax2=ax2)
    drawlimb(joints[8], joints[9], ax2=ax2)
    plt.show()


def project_cloud3dto2d(cloud):
    cloud_2d = []
    for point in cloud:
        point = np.asarray(point)
        point = np.delete(point, 3)
        cloud_2d.append(project3dto2d(point))
    return cloud_2d


#project 3d point to 2d
def project3dto2d(point):
    #Kinect intrinsics
    kinect_intrinsics = np.asarray([[525.0, 0.0, 319.5], [0.0, 525, 239.5], [0.0, 0.0, 1.0]])
    m = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    proj_matrix = np.dot(kinect_intrinsics, m)
#    point_3d = point_3d + Eigen::Vector4f (0, 0, 0, 1);  //??
    projected_point = np.dot(proj_matrix, point)
    projected_point = projected_point / projected_point [2]
    projected_point = np.delete(projected_point, 2)
    return projected_point;


def loadfiles(filenum, new_path=None):
    if new_path is not None:
        path = new_path
    fig=plt.figure()
    fig2d=plt.figure()

    for i in os.listdir(path):
        fullpath = os.path.join(path, i)
        if os.path.isfile(fullpath) and i.startswith(str(str(filenum) + '_')):
            if 'skeleton' in fullpath:
                drawskeleton(fullpath)
                drawskeleton_2d(fullpath)
            else:
                data = np.loadtxt(fullpath, delimiter=',')
                if len(data) > 0:
                    fig = add_plot(data, fig=fig)
                    fig2d = add_plot_2d(data, fig=fig2d)


def add_plot(data, fig=None):
    ax = fig.gca(projection='3d')
    # plot points in 3D
    ax.plot(data[:, 0], data[:, 1], data[:, 2], '.')
    plt.show()
    return fig


def add_plot_2d(data, fig=None):
    data = np.asarray(project_cloud3dto2d(data))
    ax4 = fig.gca()
    # plot points in 3D
    ax4.plot(data[:, 0], data[:, 1], '.')
    plt.show()
    return fig


def load_cloud(nr, cl1, cl2, path="C:/Users/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal"):
    cloud1 = np.loadtxt(str(path + '/' + str(nr) + '_' + cl1 + '.txt'), delimiter=',')
    cloud2 = np.loadtxt(str(path + '/' + str(nr) + '_' + cl2 + '.txt'), delimiter=',')
    A = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    B = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    #find cluster centers
    A_center = get_cloud_center(A)
    B_center = get_cloud_center(B)
    print "Original cloud center A: ", A_center
    print "Original cloud center B: ", B_center
    full_cloud, new_cluster_centers = perform_meanshift(A, B)
    
    return assign_to_cluster(full_cloud, new_cluster_centers)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-filenr', metavar='How many simulations should be run?', type=int)
    parser.parse_args()

    #load file
    #loadfiles(5, new_path="/media/cassandra/Local Disk/Documents and Settings/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal")
    #loadfiles(5, new_path="C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python\Data\Test 4 - Normal")
    
    load_cloud(5, "Leftarm", "LeftForearm")

'''
    print "EXECUTING SCRIPT!"

    #pass data to script
    arm_data = np.loadtxt("24_Larm.txt", delimiter=',')
    forearm_data = np.loadtxt("24_Lforearm.txt", delimiter=',')

    arm_plane_normal = fitPlaneLstsqAlt(arm_data)
    forearm_plane_normal = fitPlaneLstsqAlt(forearm_data)

    intersection_direction, offset = plane_intersection(arm_plane_normal, forearm_plane_normal, arm_data, forearm_data)
    proj_arm_line = proj_cloud_line(arm_data, intersection_direction, offset)
    proj_forearm_line = proj_cloud_line(forearm_data, intersection_direction, offset)

    proj_arm = proj_cloud(arm_data, arm_plane_normal)
    proj_forearm = proj_cloud(forearm_data, forearm_plane_normal)

    joint_loc = find_joint_location(arm_data, forearm_data, proj_arm_line, proj_forearm_line)
    #plot_results(arm_data, forearm_data, proj_arm, proj_forearm, proj_arm_line, proj_forearm_line, joint_loc)
'''