import numpy as np
import argparse
import cloud_meanshift
import cloud_plot
import cloud_linalg
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-filenr', metavar='How many simulations should be run?', type=int)
    parser.parse_args()

    #visualize clouds
    #visualize_clouds(5, new_path="/media/cassandra/Local Disk/Documents and Settings/Cassandra/Documents/GitHub/Body_measurement/Python/Data/Test 4 - Normal")
    #cloud_plot.visualize_clouds(5, new_path="C:\Users\Cassandra\Documents\GitHub\Body_measurement\Python\Data\Test 4 - Normal")
    
    cloud_meanshift.load_cloud(5, "Leftarm", "LeftForearm")

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