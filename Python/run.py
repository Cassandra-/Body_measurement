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
    limb3 = None
    limb4 = None
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

    filenr = 5
    limb1 = "LeftArm"
    limb2 = "LeftForearm"
    limb3 = "RightArm"
    limb4 = "RightForearm"
    
    #visualize clouds
    #visualize_clouds(5, new_path="/media/cassandra/Local Disk/Documents and Settings/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal")
    cloud_plot.visualize_clouds_3d(filenr, new_path=dirpath)
    cloud_plot.visualize_clouds_2d(filenr, new_path=dirpath)
    
    # Left arm calc new elbow joint
    
    left_arm_data = np.loadtxt(str(dirpath + '/' + str(filenr) + '_' + limb1 + '.txt'), delimiter=',')
    left_forearm_data = np.loadtxt(str(dirpath + '/' + str(filenr) + '_' + limb2 + '.txt'), delimiter=',')
    left_arm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(left_arm_data)
    left_forearm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(left_forearm_data)

    left_intersection_direction, left_offset = cloud_linalg.plane_intersection(left_arm_plane_normal, left_forearm_plane_normal, left_arm_data, left_forearm_data)
    left_proj_arm_line = cloud_linalg.proj_cloud_line(left_arm_data, left_intersection_direction, left_offset)
    left_proj_forearm_line = cloud_linalg.proj_cloud_line(left_forearm_data, left_intersection_direction, left_offset)

    left_proj_arm = cloud_linalg.proj_cloud(left_arm_data, left_arm_plane_normal)
    left_proj_forearm = cloud_linalg.proj_cloud(left_forearm_data, left_forearm_plane_normal)

    left_joint_loc = cloud_linalg.find_joint_location(left_arm_data, left_forearm_data, left_proj_arm_line, left_proj_forearm_line)
    cloud_plot.plot_results(left_arm_data, left_forearm_data, left_proj_arm, left_proj_forearm, left_proj_arm_line, left_proj_forearm_line, left_joint_loc)
    
    # Right arm calc new joint
    right_arm_data = np.loadtxt(str(dirpath + '/' + str(filenr) + '_' + limb3 + '.txt'), delimiter=',')
    right_forearm_data = np.loadtxt(str(dirpath + '/' + str(filenr) + '_' + limb4 + '.txt'), delimiter=',')
    right_arm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(right_arm_data)
    right_forearm_plane_normal = cloud_linalg.fitPlaneLstsqAlt(right_forearm_data)

    right_intersection_direction, right_offset = cloud_linalg.plane_intersection(right_arm_plane_normal, right_forearm_plane_normal, right_arm_data, right_forearm_data)
    right_proj_arm_line = cloud_linalg.proj_cloud_line(right_arm_data, right_intersection_direction, right_offset)
    right_proj_forearm_line = cloud_linalg.proj_cloud_line(right_forearm_data, right_intersection_direction, right_offset)

    right_proj_arm = cloud_linalg.proj_cloud(right_arm_data, right_arm_plane_normal)
    right_proj_forearm = cloud_linalg.proj_cloud(right_forearm_data, right_forearm_plane_normal)

    right_joint_loc = cloud_linalg.find_joint_location(right_arm_data, right_forearm_data, right_proj_arm_line, right_proj_forearm_line)
    cloud_plot.plot_results(right_arm_data, right_forearm_data, right_proj_arm, right_proj_forearm, right_proj_arm_line, right_proj_forearm_line, right_joint_loc)
    
    # Plot new elbow joints
    cloud_plot.visualize_clouds_3d(filenr, new_path=dirpath, point_left=left_joint_loc, point_right=right_joint_loc)
    cloud_plot.visualize_clouds_2d(filenr, new_path=dirpath, point_left=left_joint_loc, point_right=right_joint_loc)
        
        
    # Mean shift stuff
#    old_cluster_centers, new_clusters, new_cluster_centers = cloud_meanshift.load_cloud(filenr, limb1, limb2)
    
    cloud_plot.plot_old_new_clusters(filenr, limb1, limb2, path=dirpath)
    cloud_plot.plot_old_new_cluster_centers(filenr, limb1, limb2, path=dirpath)
    
    # Show mean shifted clusters
    # calulcate new joints
    # plot
