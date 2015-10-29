import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
import projectdims
import cloud_meanshift
import cloud_linalg
from itertools import cycle

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
def drawlimb(point1, point2, ax=None):
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]])

def drawlimb_2d(point1, point2, ax=None):
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]])

def drawskeleton_2d(filename, fig=None, ax=None):
    if (fig is None) or (ax is None):
        fig = plt.figure()
        ax = fig.gca()

    lines = [line.rstrip('\n').split(',') for line in open(filename)]
    joints = []
    nr = 0
    for line in lines:
        if len(line) == 3:
            line = map(float, line)
            line = projectdims.project3dto2d(line)
            if nr not in [1, 3, 4, 5, 7, 11, 13, 15, 17]:
                ax.plot([line[0]],[line[1]],'o')
            joints.append(line)
            nr = nr + 1

    #draw limbs
    drawlimb_2d(joints[10], joints[20], ax=ax)
    drawlimb_2d(joints[10], joints[19], ax=ax)
    drawlimb_2d(joints[10], joints[26], ax=ax)
    drawlimb_2d(joints[10], joints[25], ax=ax)
    drawlimb_2d(joints[26], joints[23], ax=ax)
    drawlimb_2d(joints[25], joints[24], ax=ax)
    drawlimb_2d(joints[21], joints[22], ax=ax)

    drawlimb_2d(joints[23], joints[8], ax=ax)
    drawlimb_2d(joints[24], joints[9], ax=ax)
    drawlimb_2d(joints[9], joints[2], ax=ax)
    drawlimb_2d(joints[8], joints[6], ax=ax)
    drawlimb_2d(joints[2], joints[0], ax=ax)
    drawlimb_2d(joints[6], joints[4], ax=ax)

    drawlimb_2d(joints[26], joints[12], ax=ax)
    drawlimb_2d(joints[12], joints[14], ax=ax)
    drawlimb_2d(joints[25], joints[16], ax=ax)
    drawlimb_2d(joints[16], joints[18], ax=ax)
    drawlimb_2d(joints[19], joints[21], ax=ax)
    drawlimb_2d(joints[20], joints[22], ax=ax)

    drawlimb_2d(joints[26], joints[9], ax=ax)
    drawlimb_2d(joints[25], joints[8], ax=ax)
    drawlimb_2d(joints[8], joints[9], ax=ax)
    plt.show()

#draw skeleton
def drawskeleton(filename, fig=None, ax=None):
    if (fig is None) or (ax is None):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    lines = [line.rstrip('\n').split(',') for line in open(filename)]
    joints = []
    nr = 0
    for line in lines:
        if len(line) == 3:
            line = map(float, line)
            if nr not in [1, 3, 4, 5, 7, 11, 13, 15, 17]:
                ax.plot([line[0]],[line[1]],[line[2]],'o')
            joints.append(line)
            nr = nr + 1

    #draw limbs
    drawlimb(joints[10], joints[20], ax=ax)
    drawlimb(joints[10], joints[19], ax=ax)
    drawlimb(joints[10], joints[26], ax=ax)
    drawlimb(joints[10], joints[25], ax=ax)
    drawlimb(joints[26], joints[23], ax=ax)
    drawlimb(joints[25], joints[24], ax=ax)
    drawlimb(joints[21], joints[22], ax=ax)

    drawlimb(joints[23], joints[8], ax=ax)
    drawlimb(joints[24], joints[9], ax=ax)
    drawlimb(joints[9], joints[2], ax=ax)
    drawlimb(joints[8], joints[6], ax=ax)
    drawlimb(joints[2], joints[0], ax=ax)
    drawlimb(joints[6], joints[4], ax=ax)

    drawlimb(joints[26], joints[12], ax=ax)
    drawlimb(joints[12], joints[14], ax=ax)
    drawlimb(joints[25], joints[16], ax=ax)
    drawlimb(joints[16], joints[18], ax=ax)
    drawlimb(joints[19], joints[21], ax=ax)
    drawlimb(joints[20], joints[22], ax=ax)

    drawlimb(joints[26], joints[9], ax=ax)
    drawlimb(joints[25], joints[8], ax=ax)
    drawlimb(joints[8], joints[9], ax=ax)
    plt.show()

#used to be loadfiles
def visualize_clouds(filenum, new_path=None):
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
                    
    return fig, fig2d
    
#used to be loadfiles
def visualize_clouds_3d(filenum, new_path=None):
    if new_path is not None:
        path = new_path
    fig=plt.figure()
    ax = fig.gca(projection='3d')
  
    for i in os.listdir(path):
        fullpath = os.path.join(path, i)
        if os.path.isfile(fullpath) and i.startswith(str(str(filenum) + '_')):
            if 'skeleton' in fullpath:
                drawskeleton(fullpath, fig=fig, ax=ax)
            else:
                data = np.loadtxt(fullpath, delimiter=',')
                if len(data) > 0:
                    # plot 3d
                    ax.plot(data[:, 0], data[:, 1], data[:, 2], '.')
    plt.show()

#used to be loadfiles
def visualize_clouds_2d(filenum, new_path=None):
    if new_path is not None:
        path = new_path
    fig=plt.figure()
    ax = fig.gca()

    for i in os.listdir(path):
        fullpath = os.path.join(path, i)
        if os.path.isfile(fullpath) and i.startswith(str(str(filenum) + '_')):
            if 'skeleton' in fullpath:
                drawskeleton_2d(fullpath, fig=fig, ax=ax)
            else:
                data = np.loadtxt(fullpath, delimiter=',')
                if len(data) > 0:
                    data = np.asarray(projectdims.project_cloud3dto2d(data))
                    # plot points in 3D
                    ax.plot(data[:, 0], data[:, 1], '.')
    plt.show()
    
#used to be loadfiles
def visualize_clouds_2d_new_joint(filenum, limb, new_path=None):
    if new_path is not None:
        path = new_path
    fig=plt.figure()
    ax = fig.gca()
    cloud1 = None
    cloud2 = None
    
    if limb == "Larm":
        for i in os.listdir(path):
            fullpath = os.path.join(path, i)
            if os.path.isfile(fullpath) and i.startswith(str(str(filenum) + '_')):
                if 'LeftArm' in fullpath:
                    cloud1 = np.loadtxt(fullpath, fig=fig, ax=ax)
                elif 'LeftForearm' in fullpath:
                    cloud2 = np.loadtxt(fullpath, fig=fig, ax=ax)
                if (cloud1 is not None) and (cloud2 is not None):
                    #calc joint
                    new_joint_loc = cloud_linalg.custom_joint_loc(cloud1, cloud2)


def add_plot(data, fig=None):
    ax = fig.gca(projection='3d')
    # plot points in 3D
    ax.plot(data[:, 0], data[:, 1], data[:, 2], '.')
    plt.show()
    return fig


def add_plot_2d(data, fig=None):
    data = np.asarray(projectdims.project_cloud3dto2d(data))
    ax4 = fig.gca()
    # plot points in 3D
    ax4.plot(data[:, 0], data[:, 1], '.')
    plt.show()
    return fig
    
def plot_old_new_clusters(filenr, limb1, limb2, path="C:/Users/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal"):
    cloud1 = np.loadtxt(str(path + '/' + str(filenr) + '_' + limb1 + '.txt'), delimiter=',')
    cloud2 = np.loadtxt(str(path + '/' + str(filenr) + '_' + limb2 + '.txt'), delimiter=',')
    A = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    B = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    A_center = cloud_linalg.get_cloud_center(A)
    B_center = cloud_linalg.get_cloud_center(B)
    full_cloud, new_cluster_centers, n_clusters_, labels = cloud_meanshift.perform_meanshift(A, B)
    
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = new_cluster_centers[k]
        ax.plot(full_cloud[my_members, 0], full_cloud[my_members, 1], full_cloud[my_members, 2], '.')
        ax.plot([cluster_center[0]], [cluster_center[1]], [cluster_center[2]],'o', markersize=14)
    
    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d')

    ax2.plot(A[:,0], A[:,1], A[:,2], '.')
    ax2.plot([A_center[0]], [A_center[1]], [A_center[2]],'o', markersize=14)
    
    ax2.plot(B[:,0], B[:,1], B[:,2], '.')
    ax2.plot([B_center[0]], [B_center[1]], [B_center[2]],'o', markersize=14)

    plt.show()
    
def plot_old_new_cluster_centers(filenr, limb1, limb2, path="C:/Users/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal"):
    cloud1 = np.loadtxt(str(path + '/' + str(filenr) + '_' + limb1 + '.txt'), delimiter=',')
    cloud2 = np.loadtxt(str(path + '/' + str(filenr) + '_' + limb2 + '.txt'), delimiter=',')
    A = np.column_stack((cloud1[:,0],cloud1[:,1],cloud1[:,2]))
    B = np.column_stack((cloud2[:,0],cloud2[:,1],cloud2[:,2]))
    A_center = cloud_linalg.get_cloud_center(A)
    B_center = cloud_linalg.get_cloud_center(B)
    full_cloud, new_cluster_centers, n_clusters_, labels = cloud_meanshift.perform_meanshift(A, B)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(full_cloud[:,0], full_cloud[:,1], full_cloud[:,2], '.')
    ax.plot([A_center[0]], [A_center[1]], [A_center[2]],'o', markersize=14)
    ax.plot([B_center[0]], [B_center[1]], [B_center[2]],'o', markersize=14)
    ax.plot([new_cluster_centers[0][0]], [new_cluster_centers[0][1]], [new_cluster_centers[0][2]],'o', markersize=14)
    ax.plot([new_cluster_centers[1][0]], [new_cluster_centers[1][1]], [new_cluster_centers[1][2]],'o', markersize=14)

    plt.show()