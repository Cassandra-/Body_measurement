import numpy as np
import argparse
import cloud_meanshift
import cloud_plot
import cloud_linalg
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-filenr', metavar='Pass file number (1, 2, 3, etc)', type=int)
    parser.add_argument('-limb1', metavar='Pass name of first limb', type=str)
    parser.add_argument('-limb2', metavar='Pass name of second limb', type=str)
    parser.add_argument('-dirpath', metavar='Pass directory path where files are located', type=str)
    args = parser.parse_args()
    
    filenr = None
    limb1 = None
    limb2 = None
    dirpath = "C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python\Data\Test 4 - Normal"
    
    '''
    if(vars(args)['filenr'] is not None):
	filenr = vars(args)['filenr']
    else:
        print "Need a filenr! Exiting..."
        sys.exit()
    if(vars(args)['limb1'] is not None):
      	limb1 = vars(args)['limb1']
    else:
        print "Need limb1! Exiting..."
        sys.exit()
    if(vars(args)['limb2'] is not None):
      	limb2 = vars(args)['limb2']
    else:
        print "Need limb2! Exiting..."
        sys.exit()
    if(vars(args)['dirpath'] is not None):
      	dirpath = vars(args)['dirpath']
    else:
        print "No dirpath passed! Using default: ", dirpath
    '''
    #visualize clouds
    #visualize_clouds(5, new_path="/media/cassandra/Local Disk/Documents and Settings/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal")
    cloud_plot.visualize_clouds_3d(5, new_path="C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python\Data\Test 4 - Normal")
    cloud_plot.visualize_clouds_2d(5, new_path="C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python\Data\Test 4 - Normal")
        
    filenr = 5
    limb1 = "LeftArm"
    limb2 = "LeftForearm"
#    old_cluster_centers, new_clusters, new_cluster_centers = cloud_meanshift.load_cloud(filenr, limb1, limb2)
    
    #cloud_plot.plot_old_new_clusters(filenr, limb1, limb2, path=dirpath)
    #cloud_plot.plot_old_new_cluster_centers(filenr, limb1, limb2, path=dirpath)

'''
    print "EXECUTING SCRIPT!"

    #pass data to script
    arm_data = np.loadtxt("C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python/24_Larm.txt", delimiter=',')
    forearm_data = np.loadtxt("C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python/24_Lforearm.txt", delimiter=',')

    arm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(arm_data)
    forearm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(forearm_data)

    intersection_direction, offset = cloud_linalg.plane_intersection(arm_plane_normal, forearm_plane_normal, arm_data, forearm_data)
    proj_arm_line = cloud_linalg.proj_cloud_line(arm_data, intersection_direction, offset)
    proj_forearm_line = cloud_linalg.proj_cloud_line(forearm_data, intersection_direction, offset)

    proj_arm = cloud_linalg.proj_cloud(arm_data, arm_plane_normal)
    proj_forearm = cloud_linalg.proj_cloud(forearm_data, forearm_plane_normal)

    joint_loc = cloud_linalg.find_joint_location(arm_data, forearm_data, proj_arm_line, proj_forearm_line)
    cloud_plot.plot_results(arm_data, forearm_data, proj_arm, proj_forearm, proj_arm_line, proj_forearm_line, joint_loc)
'''