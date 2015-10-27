import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
import projectdims

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
            line = projectdims.project3dto2d(line)
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
