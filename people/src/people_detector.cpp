/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 * 
 * @author: Koen Buys, Anatoly Baksheev
 */

#include <pcl/gpu/people/people_detector.h>
#include <pcl/gpu/people/label_common.h>
#include <pcl/common/eigen.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/common/centroid.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/intersections.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
//#include <pcl/gpu/people/conversions.h>
//#include <pcl/gpu/people/label_conversion.h>
//#include <pcl/gpu/people/label_segment.h>
#include <pcl/gpu/people/label_tree.h>
#include <pcl/gpu/people/probability_processor.h>
#include <pcl/gpu/people/organized_plane_detector.h>
#include <pcl/console/print.h>
#include "internal.h"
// our headers
#include "pcl/gpu/people/label_blob2.h"   //this one defines the blob structure
#include "pcl/gpu/people/label_common.h"  //this one defines the LUT's
#include "pcl/gpu/people/person_attribs.h"

// std
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

// PCL specific includes
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/console/print.h>

#include <pcl/common/eigen.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>

#include <pcl/common/time.h>

#define AREA_THRES      200 // for euclidean clusterization 1 
#define AREA_THRES2     100 // for euclidean clusterization 2
#define CLUST_TOL_SHS   0.1
#define DELTA_HUE_SHS   5

using namespace std;
using namespace pcl;
using namespace pcl::gpu::people;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::people::PeopleDetector::PeopleDetector () :
    fx_ (525.f),
    fy_ (525.f),
    cx_ (319.5f),
    cy_ (239.5f),
    delta_hue_tolerance_ (5),
    dt_ (0.5),
    alpha_tracking_ (0.5),
    beta_tracking_ (0.005),
    active_tracking_ (false)
{
  PCL_DEBUG("[pcl::gpu::people::PeopleDetector] : (D) : Constructor called\n");

  // Create a new organized plane detector
  org_plane_detector_ = OrganizedPlaneDetector::Ptr (new OrganizedPlaneDetector ());

  // Create a new probability_processor
  probability_processor_ = ProbabilityProcessor::Ptr (new ProbabilityProcessor ());

  // Create a new person attribs
  person_attribs_ = PersonAttribs::Ptr (new PersonAttribs ());

  // Just created, indicates first time callback (allows for tracking features to start from second frame)
  first_iteration_ = true;

  //set the velocities (for tracking)
  for (int i = 0; i < num_parts_all; i++)
  {
    joints_velocity_[i] = Eigen::Vector4f (0.0, 0.0, 0.0, 0.0);
  }

  //stores the body centroid (value not used yet)
  mean_vals_ = Eigen::Vector4f (0.0, 0.0, 0.0, 0.0);

  // allocation buffers with default sizes
  // if input size is other than the defaults, 
  // then the buffers will be reallocated at processing time.
  // This cause only penalty for first frame ( one reallocation of each buffer )
  allocate_buffers ();
}

/*
 *Enables/disables tracking
 */
void
pcl::gpu::people::PeopleDetector::setActiveTracking (bool value)
{
  active_tracking_ = value;

}

/**
 * Set alpha parameter for tracking
 */
void
pcl::gpu::people::PeopleDetector::setAlphaTracking (float value)
{
  alpha_tracking_ = value;

}

/**
 * Set beta parameter for tracking
 */
void
pcl::gpu::people::PeopleDetector::setBetaTracking (float value)
{
  beta_tracking_ = value;

}

void
pcl::gpu::people::PeopleDetector::setIntrinsics (float fx,
                                                 float fy,
                                                 float cx,
                                                 float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
}

/** @brief This function prepares the needed buffers on both host and device **/
void
pcl::gpu::people::PeopleDetector::allocate_buffers (int rows,
                                                    int cols)
{
  device::Dilatation::prepareRect5x5Kernel (kernelRect5x5_);

  cloud_host_.width = cols;
  cloud_host_.height = rows;
  cloud_host_.points.resize (cols * rows);
  cloud_host_.is_dense = false;

  cloud_host_color_.width = cols;
  cloud_host_color_.height = rows;
  cloud_host_color_.resize (cols * rows);
  cloud_host_color_.is_dense = false;

  hue_host_.width = cols;
  hue_host_.height = rows;
  hue_host_.points.resize (cols * rows);
  hue_host_.is_dense = false;

  depth_host_.width = cols;
  depth_host_.height = rows;
  depth_host_.points.resize (cols * rows);
  depth_host_.is_dense = false;

  flowermat_host_.width = cols;
  flowermat_host_.height = rows;
  flowermat_host_.points.resize (cols * rows);
  flowermat_host_.is_dense = false;

  cloud_device_.create (rows, cols);
  hue_device_.create (rows, cols);

  depth_device1_.create (rows, cols);
  depth_device2_.create (rows, cols);
  fg_mask_.create (rows, cols);
  fg_mask_grown_.create (rows, cols);
}

/**
 * Calculates 2D coordinates of the joint
 * @param[in] point_3d 3d position of the joint
 * @return 2D coordinates
 */
Eigen::Vector3f
pcl::gpu::people::PeopleDetector::project3dTo2d (Eigen::Vector4f point_3d)
{
  Eigen::Vector3f projected_point;

  Eigen::MatrixXf intrinsics (3, 3);

  intrinsics << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
  Eigen::MatrixXf m (3, 4);
  m << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0;
  Eigen::MatrixXf projection_matrix = intrinsics * m;
  point_3d = point_3d + Eigen::Vector4f (0, 0, 0, 1);  //??
  projected_point = projection_matrix * point_3d;
  projected_point /= projected_point (2);
  return projected_point;

}

int
pcl::gpu::people::PeopleDetector::process (const Depth& depth,
                                           const Image& rgba)
{
  int cols;
  allocate_buffers (depth.rows (), depth.cols ());

  depth_device1_ = depth;

  const device::Image& i = (const device::Image&) rgba;
  device::computeHueWithNans (i, depth_device1_, hue_device_);
  //TODO Hope this is temporary and after porting to GPU the download will be deleted  
  hue_device_.download (hue_host_.points, cols);

  device::Intr intr (fx_, fy_, cx_, cy_);
  intr.setDefaultPPIfIncorrect (depth.cols (), depth.rows ());

  device::Cloud& c = (device::Cloud&) cloud_device_;
  device::computeCloud (depth, intr, c);
  cloud_device_.download (cloud_host_.points, cols);

  // uses cloud device, cloud host, depth device, hue device and other buffers
  return process ();
}

int
pcl::gpu::people::PeopleDetector::process (const pcl::PointCloud<PointTC>::ConstPtr &cloud)
{
  allocate_buffers (cloud->height, cloud->width);

  const float qnan = std::numeric_limits<float>::quiet_NaN ();

  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud_host_.points[i].x = cloud->points[i].x;
    cloud_host_.points[i].y = cloud->points[i].y;
    cloud_host_.points[i].z = cloud->points[i].z;

    bool valid = isFinite (cloud_host_.points[i]);

    hue_host_.points[i] = !valid ? qnan : device::computeHue (cloud->points[i].rgba);
    depth_host_.points[i] = !valid ? 0 : static_cast<unsigned short> (cloud_host_.points[i].z * 1000);  //m -> mm
  }
  cloud_device_.upload (cloud_host_.points, cloud_host_.width);
  hue_device_.upload (hue_host_.points, hue_host_.width);
  depth_device1_.upload (depth_host_.points, depth_host_.width);

  // uses cloud device, cloud host, depth device, hue device and other buffers
  return process ();
}

int
pcl::gpu::people::PeopleDetector::process ()
{

  part_t base_joint = Neck;  //the joint from which the blob-tree is built later

  int cols = cloud_device_.cols ();
  int rows = cloud_device_.rows ();

  rdf_detector_->process (depth_device1_, cloud_host_, AREA_THRES);

  const RDFBodyPartsDetector::BlobMatrix& sorted = rdf_detector_->getBlobMatrix ();

  //Selecting the base joint for building the skeleton-tree (in Neck is missing we use one of the Chest blobs)
  if (sorted[Neck].size () == 0)
  {
    if (sorted[Lchest].size () != 0)
    {
      base_joint = Lchest;
    }
    else if (sorted[Rchest].size () != 0)
    {
      base_joint = Rchest;
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // if we found a base_joint display the tree, and continue with processing
  if (sorted[base_joint].size () != 0)
  {
    int c = 0;
    Tree2 t;
    buildTree (sorted, cloud_host_, base_joint, c, t);

    const std::vector<int>& seed = t.indices.indices;

    std::fill (flowermat_host_.points.begin (), flowermat_host_.points.end (), 0);
    {
      //ScopeTime time("shs");    
      shs5 (cloud_host_, seed, &flowermat_host_.points[0]);
    }

    fg_mask_.upload (flowermat_host_.points, cols);
    device::Dilatation::invoke (fg_mask_, kernelRect5x5_, fg_mask_grown_);

    device::prepareForeGroundDepth (depth_device1_, fg_mask_grown_, depth_device2_);

    //// //////////////////////////////////////////////////////////////////////////////////////////////// //
    //// The second label evaluation

    rdf_detector_->process (depth_device2_, cloud_host_, AREA_THRES2);
    const RDFBodyPartsDetector::BlobMatrix& sorted2 = rdf_detector_->getBlobMatrix ();

    //brief Test if the second tree is build up correctly
    if (sorted2[base_joint].size () > 0)
    {
      Tree2 t2;
      buildTree (sorted2, cloud_host_, base_joint, c, t2);
      int par = 0;
      for (int f = 0; f < NUM_PARTS; f++)

        mean_vals_ = Eigen::Vector4f (0.0, 0.0, 0.0, 0.0);

      for (int i = 0; i < num_parts_all; i++)
      {
        //resetting
        mean_vals_ += skeleton_joints_[i];
        if (active_tracking_)
        {
          skeleton_joints_prev_[i] = skeleton_joints_[i];

        }

        skeleton_joints_[i] = Eigen::Vector4f (-1, -1, -1, -1);

      }

      mean_vals_ = mean_vals_ / num_parts_all;

      estimateJoints (sorted2, t2, base_joint, 0);
      skeleton_blobs_[base_joint] = sorted2[base_joint][0];

      for (int i = 0; i < num_parts_labeled; i++)
      {
        if (sorted2[i].size () > 0 && skeleton_joints_[i][0] == -1.0)
        {

          skeleton_joints_[i] = sorted2[i][0].mean;
          skeleton_blobs_[i] = (sorted2[i][0]);

        }
      }

      //we use ALL chest blobs (not only the largest one), important for the shoulder calculation
      if (sorted2[Lchest].size () > 0)
        skeleton_blobs_[Lchest] = * (sorted2[Lchest].data ());
      if (sorted2[Rchest].size () > 0)
        skeleton_blobs_[Rchest] = * (sorted2[Rchest].data ());

      calculateAddtionalJoints ();

      if (active_tracking_)
        alphaBetaTracking ();

      static int counter = 0;  // TODO move this logging to PeopleApp
      return 2;
    }
    return 1;
    //output: Tree2 and PointCloud<XYZRGBL> 
  }
  return 0;
}

void
pcl::gpu::people::PeopleDetector::getMaxDist (const pcl::PointCloud<PointXYZ> &cloud,
                                              Eigen::Vector4f &pivot_pt,
                                              Eigen::Vector4f &max_pt)
{
  float max_dist = -FLT_MAX;
  int max_idx = -1;
  float dist;

  // If the data is dense, we don't need to check for NaN
  if (cloud.is_dense)
  {
    for (size_t i = 0; i < cloud.points.size (); ++i)
    {
      Eigen::Vector4f pt;
      pt[0] = cloud.points[i].x;
      pt[1] = cloud.points[i].y;
      pt[2] = cloud.points[i].z;
      pt[3] = 0.0;
      dist = (pivot_pt - pt).norm ();
      if (dist > max_dist)
      {
        max_idx = int (i);
        max_dist = dist;
      }
    }
  }
  // NaN or Inf values could exist => check for them
  else
  {
    for (size_t i = 0; i < cloud.points.size (); ++i)
    {
      // Check if the point is invalid
      if (!pcl_isfinite(cloud.points[i].x) || !pcl_isfinite(cloud.points[i].y) || !pcl_isfinite(cloud.points[i].z))
        continue;
      Eigen::Vector4f pt;
      pt[0] = cloud.points[i].x;
      pt[1] = cloud.points[i].y;
      pt[2] = cloud.points[i].z;
      pt[3] = 0.0;
      dist = (pivot_pt - pt).norm ();
      if (dist > max_dist)
      {
        max_idx = int (i);
        max_dist = dist;
      }
    }
  }

  if (max_idx != -1)
  {
    max_pt[0] = cloud.points[max_idx].x;
    max_pt[1] = cloud.points[max_idx].y;
    max_pt[2] = cloud.points[max_idx].z;
    max_pt[3] = 0.0;
  }

  else
    max_pt = Eigen::Vector4f (0, 0, 0, 0);
}

/**
 * Calculates/corrects the position of following joints:
 * Shoulders, ELbows(if the Elbow-Blob is missing) and hips
 */
void
pcl::gpu::people::PeopleDetector::calculateAddtionalJoints ()
{

  Eigen::Vector4f intersection_point;
  Eigen::Vector4f min;
  Eigen::Vector4f max;
  pcl::getMinMax3D (cloud_host_, skeleton_blobs_[Rchest].indices, min, max);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_initial (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Rchest].indices.indices));
  pcl::PassThrough<pcl::PointXYZ> pass;

  //Right shoulder
  pass.setInputCloud (cloud_initial);
  pass.setFilterFieldName ("y");
  double minval = min[1];  //y
  double maxval = max[1];  //y
  pass.setFilterLimits (-100.0, minval + 0.1);
  pass.filter (*cloud_filtered);

  pcl::compute3DCentroid (*cloud_filtered, intersection_point);
  skeleton_joints_[Rshoulder] = intersection_point;

  //Left shoulder
  pcl::getMinMax3D (cloud_host_, skeleton_blobs_[Lchest].indices, min, max);

  pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<PointXYZ> (*cloud_filtered));  //resetting
  cloud_initial = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Lchest].indices.indices));

  pass.setInputCloud (cloud_initial);
  pass.setFilterFieldName ("y");
  minval = min[1];  //y
  maxval = max[1];  //y
  pass.setFilterLimits (-100.0, minval + 0.1);

  pass.filter (*cloud_filtered);

  pcl::compute3DCentroid (*cloud_filtered, intersection_point);
  skeleton_joints_[Lshoulder] = intersection_point;

  //Calculating the Elbows, v2

  std::vector<int> indices;
  if (skeleton_joints_[Lelbow][0] == -1)
  {
    cloud_initial = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Larm].indices.indices));
	//export pointclouds so I can read it in python --> if not possible, try to understand this all
    getMaxDist (*cloud_initial, skeleton_joints_[Lshoulder], intersection_point);
    //skeleton_joints_[Lelbow]=intersection_point;
    pass.setFilterFieldName ("y");
    minval = min[1];  //y
    maxval = max[1];  //y
    pass.setFilterLimits (float (intersection_point[1] - 0.05), float (intersection_point[1] - 0.05));
    pass.filter (*cloud_initial);
    pcl::compute3DCentroid (*cloud_initial, intersection_point);
    skeleton_joints_[Lelbow] = intersection_point;

  }
  if (skeleton_joints_[Relbow][0] == -1)
  {
    cloud_initial = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Rarm].indices.indices));

    getMaxDist (*cloud_initial, skeleton_joints_[Rshoulder], intersection_point);
    //skeleton_joints_[Relbow]=intersection_point;
    pass.setFilterFieldName ("y");
    minval = min[1];  //y
    maxval = max[1];  //y
    pass.setFilterLimits (float (intersection_point[1] - 0.05), float (intersection_point[1] - 0.05));
    pass.filter (*cloud_initial);
    pcl::compute3DCentroid (*cloud_initial, intersection_point);
    skeleton_joints_[Relbow] = intersection_point;

  }

  //Calculating the hips

  //Right hip
  cloud_initial = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Rhips].indices.indices));
  pass.setInputCloud (cloud_initial);
  pass.setFilterFieldName ("y");
  pcl::getMinMax3D (cloud_host_, skeleton_blobs_[Rhips].indices, min, max);
  minval = min[1];  //y
  maxval = max[1];  //y
  pass.setFilterLimits (maxval - 0.1, 100.0);
  pass.filter (*cloud_filtered);
  pcl::compute3DCentroid (*cloud_filtered, intersection_point);
  skeleton_joints_[Rhips] = intersection_point;

  //Left hip
  cloud_initial = pcl::PointCloud<pcl::PointXYZ>::Ptr (new pcl::PointCloud<pcl::PointXYZ> (cloud_host_, skeleton_blobs_[Lhips].indices.indices));
  pass.setInputCloud (cloud_initial);
  pass.setFilterFieldName ("y");
  pcl::getMinMax3D (cloud_host_, skeleton_blobs_[Lhips].indices, min, max);
  minval = min[1];  //y
  maxval = max[1];  //y
  pass.setFilterLimits (maxval - 0.1, 100.0);
  pass.filter (*cloud_filtered);
  pcl::compute3DCentroid (*cloud_filtered, intersection_point);
  skeleton_joints_[Lhips] = intersection_point;

}

/*
 * Joint tracking based on their previous positions and velocities
 */
void
pcl::gpu::people::PeopleDetector::alphaBetaTracking ()
{

  float outlier_thresh = 1.8;  //Outliers above this threshold are dismissed (old value is used)
  float outlier_thresh2 = 0.1;  //Used for special handling of the hands. If the hands position change is higher then thresh2, tracking parameters are set higher (position change happens faster)
  float outlier_thresh3 = 0.15;  //for hand correction

  for (int i = 0; i < num_parts_all; i++)
  {

    Eigen::Vector4f measured = skeleton_joints_[i];
    Eigen::Vector4f prev = skeleton_joints_prev_[i];
    Eigen::Vector4f velocity_prev = joints_velocity_[i];

    bool valid = (measured - (prev)).norm () < outlier_thresh;
    bool valid2 = (measured - (prev + velocity_prev)).norm () < outlier_thresh2;

    //hand 

    //hand correction
    //Left hand
    if (i == Lhand && skeleton_joints_[Lforearm][0] != -1 && skeleton_joints_[Lelbow][0] != -1)
    {
      //Expected position based on forearm and elbow positions
      Eigen::Vector4f dir = skeleton_joints_[Lforearm] - skeleton_joints_[Lelbow];
      if (dir.norm () < 0.04)
        dir *= 2.0;
      Eigen::Vector4f corrected = skeleton_joints_[Lforearm] + dir;
      //If the measured position is too far away from the expected position we use the expected position
      if ( ( (corrected - measured).norm () > outlier_thresh3 && !valid2) || (corrected - measured).norm () > 0.3)
        measured = corrected;
    }
    //Right hand (same as left hand)
    if (i == Rhand && skeleton_joints_[Rforearm][0] != -1 && skeleton_joints_[Relbow][0] != -1)
    {
      Eigen::Vector4f dir = skeleton_joints_[Rforearm] - skeleton_joints_[Relbow];
      if (dir.norm () < 0.04)
        dir *= 2.0;
      Eigen::Vector4f corrected = skeleton_joints_[Rforearm] + dir;
      if ( ( (corrected - measured).norm () > outlier_thresh3 && !valid2) || (corrected - measured).norm () > 0.3)
        measured = corrected;

    }

    valid = (measured - (prev)).norm () < outlier_thresh;

    //going through all dimensions
    for (int dim = 0; dim < 3; dim++)
    {

      float xm = measured[dim];

      float xk = prev[dim] + (velocity_prev[dim] * dt_);    //predicted position
      float vk = velocity_prev[dim];    //velocity update
      float rk = xm - xk;    //difference between measured and predicted

      //new position and velocity
      if (prev[dim] == -1.0 || prev[dim] == 0.0)    //if the old values are invalid, use the new ones
      {
        xk = measured[dim];
        vk = 0.0;
      }
      else
      {
        if (valid && ! (xm == -1) && ! (xm == 0))    //Apply tracking
        {
          //If its the hands, arms and elbows and the joint has moved fast
          //we increase the tracking parameters, so that the position changes faster
          //the reason is that hand position might change much faster then the position of other joints
          if (!valid2 && i != Lhand && i != Rhand && i != Lforearm && i != Rforearm && i != Lelbow && i != Relbow && i != Lfoot && i != Rfoot)
          {
            xk += alpha_tracking_ * rk * 0.2;
            vk += (beta_tracking_ * rk * 0.5) / dt_;

          }
          else
          {

            if (i == Lhand || i == Rhand)          //Increase the velocity update if it is a hand (because the hand position changes much faster)
            {
              xk += alpha_tracking_ * rk;
              vk += (beta_tracking_ * 5.0 * rk) / dt_;

            }
            else
            {          //If it's not a hand use usual tracking parameters

              xk += alpha_tracking_ * rk;
              vk += (beta_tracking_ * rk) / dt_;

            }
          }

        }
        else
        {

          xk = prev[dim];
          vk = velocity_prev[dim];

        }
      }

      skeleton_joints_[i][dim] = xk;
      joints_velocity_[i][dim] = vk;
    }
  }
}

/**
 *Goes recursively through all the children of the tree and calculate the positions of the joints
 *@param[in] tree the blob tree which we search to estimate the optimal blob for every joint
 *@param[in] part_label label of the part we a building the tree from (e.g. Neck)
 *@param[in] id of the blob from which we build the tree (usually 0. There might be several blobs with the same label, they are sorted by the size.)
 */
int
pcl::gpu::people::PeopleDetector::estimateJoints (const std::vector<std::vector<Blob2, Eigen::aligned_allocator<Blob2> > >& sorted,
                                                  Tree2& tree,
                                                  int part_label,
                                                  int part_lid)
{
  int nr_children = LUT_nr_children[part_label];
  tree.parts_lid[part_label] = part_lid;

  const Blob2& blob = sorted[part_label][part_lid];

  // iterate over the number of pixels that are part of this label
  //const std::vector<int>& indices = blob.indices.indices;
  //tree.indices.indices.insert(tree.indices.indices.end(), indices.begin(), indices.end());

  if (nr_children == 0)
    return 0;

  // iterate over all possible children
  for (int i = 0; i < nr_children; i++)
  {
    // check if this child has a valid child_id, leaf test should be redundant
    if (blob.child_id[i] != NO_CHILD && blob.child_id[i] != LEAF)
    {
      //tree.total_dist_error += blob.child_dist[i];
      skeleton_joints_[blob.child_label[i]] = sorted[blob.child_label[i]][blob.child_lid[i]].mean;
      skeleton_blobs_[blob.child_label[i]] = sorted[blob.child_label[i]][blob.child_lid[i]];
      estimateJoints (sorted, tree, blob.child_label[i], blob.child_lid[i]);
    }
  }
  return 0;
}

int
pcl::gpu::people::PeopleDetector::processProb (const pcl::PointCloud<PointTC>::ConstPtr &cloud)
{
  allocate_buffers (cloud->height, cloud->width);

  const float qnan = std::numeric_limits<float>::quiet_NaN ();

  for (size_t i = 0; i < cloud->points.size (); ++i)
  {
    cloud_host_color_.points[i].x = cloud_host_.points[i].x = cloud->points[i].x;
    cloud_host_color_.points[i].y = cloud_host_.points[i].y = cloud->points[i].y;
    cloud_host_color_.points[i].z = cloud_host_.points[i].z = cloud->points[i].z;
    cloud_host_color_.points[i].rgba = cloud->points[i].rgba;

    bool valid = isFinite (cloud_host_.points[i]);

    hue_host_.points[i] = !valid ? qnan : device::computeHue (cloud->points[i].rgba);
    depth_host_.points[i] = !valid ? 0 : static_cast<unsigned short> (cloud_host_.points[i].z * 1000);  //m -> mm
  }
  cloud_device_.upload (cloud_host_.points, cloud_host_.width);
  hue_device_.upload (hue_host_.points, hue_host_.width);
  depth_device1_.upload (depth_host_.points, depth_host_.width);

  // uses cloud device, cloud host, depth device, hue device and other buffers
  return processProb ();
}

int
pcl::gpu::people::PeopleDetector::processProb ()
{
  int cols = cloud_device_.cols ();
  int rows = cloud_device_.rows ();

  PCL_DEBUG("[pcl::gpu::people::PeopleDetector::processProb] : (D) : called\n");

  // First iteration no tracking can take place
  if (first_iteration_)
  {
    //Process input pointcloud with RDF
    rdf_detector_->processProb (depth_device1_);

    probability_processor_->SelectLabel (depth_device1_, rdf_detector_->labels_, rdf_detector_->P_l_);
  }
  // Join probabilities from previous result
  else
  {
    // Backup P_l_1_ value in P_l_prev_1_;
    rdf_detector_->P_l_prev_1_.swap (rdf_detector_->P_l_1_);
    // Backup P_l_2_ value in P_l_prev_2_;
    rdf_detector_->P_l_prev_2_.swap (rdf_detector_->P_l_2_);

    //Process input pointcloud with RDF
    rdf_detector_->processProb (depth_device1_);

    // Create Gaussian Kernel for this iteration, in order to smooth P_l_2_
    float* kernel_ptr_host;
    int kernel_size = 5;
    float sigma = 1.0;
    kernel_ptr_host = probability_processor_->CreateGaussianKernel (sigma, kernel_size);
    DeviceArray<float> kernel_device (kernel_size * sizeof(float));
    kernel_device.upload (kernel_ptr_host, kernel_size * sizeof(float));

    // Output kernel for verification
    PCL_DEBUG("[pcl::gpu::people::PeopleDetector::processProb] : (D) : kernel:\n");
    for (int i = 0; i < kernel_size; i++)
      PCL_DEBUG("\t Entry %d \t: %lf\n", i, kernel_ptr_host[i]);

    if (probability_processor_->GaussianBlur (depth_device1_, rdf_detector_->P_l_2_, kernel_device, rdf_detector_->P_l_Gaus_Temp_, rdf_detector_->P_l_Gaus_)
        != 1)
      PCL_ERROR("[pcl::gpu::people::PeopleDetector::processProb] : (E) : Gaussian Blur failed\n");

    // merge with prior probabilities at this line
    probability_processor_->CombineProb (depth_device1_, rdf_detector_->P_l_Gaus_, 0.5, rdf_detector_->P_l_, 0.5, rdf_detector_->P_l_Gaus_Temp_);
    PCL_DEBUG("[pcl::gpu::people::PeopleDetector::processProb] : (D) : CombineProb called\n");

    // get labels
    probability_processor_->SelectLabel (depth_device1_, rdf_detector_->labels_, rdf_detector_->P_l_Gaus_Temp_);
  }

  // This executes the connected components
  rdf_detector_->processSmooth (depth_device1_, cloud_host_, AREA_THRES);
  // This creates the blobmatrix
  rdf_detector_->processRelations (person_attribs_);

  // Backup this value in P_l_1_;
  rdf_detector_->P_l_1_.swap (rdf_detector_->P_l_);

  const RDFBodyPartsDetector::BlobMatrix& sorted = rdf_detector_->getBlobMatrix ();

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // if we found a neck display the tree, and continue with processing
  if (sorted[Neck].size () != 0)
  {
    int c = 0;
    Tree2 t;
    buildTree (sorted, cloud_host_, Neck, c, t, person_attribs_);

    const std::vector<int>& seed = t.indices.indices;

    std::fill (flowermat_host_.points.begin (), flowermat_host_.points.end (), 0);
    {
      //ScopeTime time("shs");
      shs5 (cloud_host_, seed, &flowermat_host_.points[0]);
    }

    fg_mask_.upload (flowermat_host_.points, cols);
    device::Dilatation::invoke (fg_mask_, kernelRect5x5_, fg_mask_grown_);

    device::prepareForeGroundDepth (depth_device1_, fg_mask_grown_, depth_device2_);

    //// //////////////////////////////////////////////////////////////////////////////////////////////// //
    //// The second label evaluation

    rdf_detector_->processProb (depth_device2_);
    // TODO: merge with prior probabilities at this line

    // get labels
    probability_processor_->SelectLabel (depth_device1_, rdf_detector_->labels_, rdf_detector_->P_l_);
    // This executes the connected components
    rdf_detector_->processSmooth (depth_device2_, cloud_host_, AREA_THRES2);
    // This creates the blobmatrix
    rdf_detector_->processRelations (person_attribs_);

    // Backup this value in P_l_2_;
    rdf_detector_->P_l_2_.swap (rdf_detector_->P_l_);

    const RDFBodyPartsDetector::BlobMatrix& sorted2 = rdf_detector_->getBlobMatrix ();

    //brief Test if the second tree is build up correctly
    if (sorted2[Neck].size () != 0)
    {
      Tree2 t2;
      buildTree (sorted2, cloud_host_, Neck, c, t2, person_attribs_);

      //Buuilding the tree beginning from the Neck
      Tree2 t3;
      cerr << " " << t3 << std::endl;

      buildTree (sorted2, cloud_host_, Neck, 0, t3, this->person_attribs_);

      for (int i = 0; i < num_parts_labeled; i++)
      {

        skeleton_joints_[i] = Eigen::Vector4f (-1, -1, -1, -1);
      }

      for (int i = 0; i < num_parts_labeled; i++)
      {
        if (sorted2[i].size () != 0)
        {
          skeleton_joints_[i] = sorted2[i].data ()->mean;
        }
      }

      int par = 0;
      for (int f = 0; f < NUM_PARTS; f++)
      {
        if (t2.parts_lid[f] == NO_CHILD)
        {
          cerr << "1;";
          par++;
        }
        else
          cerr << "0;";
      }
      std::cerr << std::endl;
      static int counter = 0;  // TODO move this logging to PeopleApp

      //cerr << t2.nr_parts << ";" << par << ";" << t2.total_dist_error << ";" << t2.norm_dist_error << ";" << counter++ << ";" << endl;
      first_iteration_ = false;
      return 2;
    }
    first_iteration_ = false;
    return 1;
    //output: Tree2 and PointCloud<XYZRGBL>
  }
  first_iteration_ = false;
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace
{
  void
  getProjectedRadiusSearchBox (int rows,
                               int cols,
                               const pcl::device::Intr& intr,
                               const pcl::PointXYZ& point,
                               float squared_radius,
                               int &minX,
                               int &maxX,
                               int &minY,
                               int &maxY)
  {
    int min, max;

    float3 q;
    q.x = intr.fx * point.x + intr.cx * point.z;
    q.y = intr.fy * point.y + intr.cy * point.z;
    q.z = point.z;

    // http://www.wolframalpha.com/input/?i=%7B%7Ba%2C+0%2C+b%7D%2C+%7B0%2C+c%2C+d%7D%2C+%7B0%2C+0%2C+1%7D%7D+*+%7B%7Ba%2C+0%2C+0%7D%2C+%7B0%2C+c%2C+0%7D%2C+%7Bb%2C+d%2C+1%7D%7D

    float coeff8 = 1;                                   //K_KT_.coeff (8);
    float coeff7 = intr.cy;                             //K_KT_.coeff (7);
    float coeff4 = intr.fy * intr.fy + intr.cy * intr.cy;  //K_KT_.coeff (4);

    float coeff6 = intr.cx;                             //K_KT_.coeff (6);
    float coeff0 = intr.fx * intr.fx + intr.cx * intr.cx;  //K_KT_.coeff (0);

    float a = squared_radius * coeff8 - q.z * q.z;
    float b = squared_radius * coeff7 - q.y * q.z;
    float c = squared_radius * coeff4 - q.y * q.y;

    // a and c are multiplied by two already => - 4ac -> - ac
    float det = b * b - a * c;

    if (det < 0)
    {
      minY = 0;
      maxY = rows - 1;
    }
    else
    {
      float y1 = (b - sqrt (det)) / a;
      float y2 = (b + sqrt (det)) / a;

      min = (int) std::min (floor (y1), floor (y2));
      max = (int) std::max (ceil (y1), ceil (y2));
      minY = std::min (rows - 1, std::max (0, min));
      maxY = std::max (std::min (rows - 1, max), 0);
    }

    b = squared_radius * coeff6 - q.x * q.z;
    c = squared_radius * coeff0 - q.x * q.x;

    det = b * b - a * c;
    if (det < 0)
    {
      minX = 0;
      maxX = cols - 1;
    }
    else
    {
      float x1 = (b - sqrt (det)) / a;
      float x2 = (b + sqrt (det)) / a;

      min = (int) std::min (floor (x1), floor (x2));
      max = (int) std::max (ceil (x1), ceil (x2));
      minX = std::min (cols - 1, std::max (0, min));
      maxX = std::max (std::min (cols - 1, max), 0);
    }
  }

  float
  sqnorm (const pcl::PointXYZ& p1,
          const pcl::PointXYZ& p2)
  {
    float dx = (p1.x - p2.x);
    float dy = (p1.y - p2.y);
    float dz = (p1.z - p2.z);
    return dx * dx + dy * dy + dz * dz;
  }
}

void
pcl::gpu::people::PeopleDetector::shs5 (const pcl::PointCloud<PointT> &cloud,
                                        const std::vector<int>& indices,
                                        unsigned char *mask)
{
  pcl::device::Intr intr (fx_, fy_, cx_, cy_);
  intr.setDefaultPPIfIncorrect (cloud.width, cloud.height);

  const float *hue = &hue_host_.points[0];
  double squared_radius = CLUST_TOL_SHS * CLUST_TOL_SHS;

  std::vector<std::vector<int> > storage (100);

  // Process all points in the indices vector
  int total = static_cast<int> (indices.size ());
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int k = 0; k < total; ++k)
  {
    int i = indices[k];
    if (mask[i])
      continue;

    mask[i] = 255;

    int id = 0;
#ifdef _OPENMP
    id = omp_get_thread_num();
#endif
    std::vector<int>& seed_queue = storage[id];
    seed_queue.clear ();
    seed_queue.reserve (cloud.size ());
    int sq_idx = 0;
    seed_queue.push_back (i);

    PointT p = cloud.points[i];
    float h = hue[i];

    while (sq_idx < (int) seed_queue.size ())
    {
      int index = seed_queue[sq_idx];
      const PointT& q = cloud.points[index];

      if (!pcl::isFinite (q))
        continue;

      // search window                  
      int left, right, top, bottom;
      getProjectedRadiusSearchBox (cloud.height, cloud.width, intr, q, squared_radius, left, right, top, bottom);

      int yEnd = (bottom + 1) * cloud.width + right + 1;
      int idx = top * cloud.width + left;
      int skip = cloud.width - right + left - 1;
      int xEnd = idx - left + right + 1;

      for (; xEnd < yEnd; idx += 2 * skip, xEnd += 2 * cloud.width)
        for (; idx < xEnd; idx += 2)
        {
          if (mask[idx])
            continue;

          if (sqnorm (cloud.points[idx], q) <= squared_radius)
          {
            float h_l = hue[idx];

            if (fabs (h_l - h) < DELTA_HUE_SHS)
            {
              seed_queue.push_back (idx);
              mask[idx] = 255;
            }
          }
        }

      sq_idx++;
    }
  }
}

