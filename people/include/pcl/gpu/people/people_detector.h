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
 * @author: Koen Buys
 */

#ifndef PCL_GPU_PEOPLE_PERSON_H_
#define PCL_GPU_PEOPLE_PERSON_H_

#include <iostream>
#include <sstream>
#include <fstream>

#include <pcl/point_types.h>
#include <pcl/console/print.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/people/label_common.h>
#include <pcl/gpu/people/tree.h>
#include <pcl/common/eigen.h>

#include <pcl/gpu/people/label_tree.h>
#include <pcl/gpu/people/person_attribs.h>
//#include <opencv2/core/core.hpp>

#include <pcl/gpu/people/bodyparts_detector.h>
#include <pcl/gpu/people/face_detector.h>
#include <pcl/gpu/people/organized_plane_detector.h>
#include <pcl/gpu/people/probability_processor.h>

namespace pcl
{
  namespace gpu
  {
    namespace people
    {
      /*
       struct OtherDetector
       {
       typedef boost::shared_ptr<OtherDetector> Ptr;
       };
       */
      //part_t const body_parts[] =  {Lfoot,Lleg, Lknee,Lthigh,Rfoot,Rleg,Rknee,Rthigh,Rhips,Lhips,Neck,Rarm,Relbow,Rforearm,Rhand,Larm,Lelbow,Lforearm,Lhand,FaceLB,FaceRB,FaceLT,FaceRT,Rchest,Lchest};
      //char* const body_parts_str[] = {"Lfoot","Lleg", "Lknee","Lthigh","Rfoot","Rleg","Rknee","Rthigh","Rhips","Lhips","Neck","Rarm","Relbow","Rforearm","Rhand","Larm","Lelbow","Lforearm","Lhand","FaceLB","FaceRB","FaceLT","FaceRT","Rchest","Lchest"};
      const int num_parts_labeled = 25;

      const int num_parts_all = 27;    //including shoulders

      class PCL_EXPORTS PeopleDetector
      {
        public:
          typedef boost::shared_ptr<PeopleDetector> Ptr;

          typedef pcl::PointXYZRGBA PointTC;
          typedef pcl::PointXYZ PointT;
          typedef DeviceArray2D<unsigned short> Depth;
          typedef DeviceArray2D<pcl::RGB> Image;

          // ALL THE DETECTOR OBJECTS
          RDFBodyPartsDetector::Ptr rdf_detector_;
          OrganizedPlaneDetector::Ptr org_plane_detector_;
          FaceDetector::Ptr face_detector_;
          //OtherDetector::Ptr          other_detector_;

          // ALL THE OTHER PEOPLE STUFF
          PersonAttribs::Ptr person_attribs_;
          ProbabilityProcessor::Ptr probability_processor_;

          /** \brief Class constructor. */
          PeopleDetector ();

          /** \brief Class destructor. */
          ~PeopleDetector ()
          {
          }

          /** \brief User must set non standard intrinsics */
          void
          setIntrinsics (float fx,
                         float fy,
                         float cx = -1,
                         float cy = -1);

          void
          setActiveTracking (bool value);

          void
          setAlphaTracking (float value);

          void
          setBetaTracking (float value);

          /** \brief Possible will be removed because of extra overheads */
          int
          process (const PointCloud<PointTC>::ConstPtr &cloud);

          int
          processProb (const PointCloud<PointTC>::ConstPtr &cloud);

          int
          process (const Depth& depth,
                   const Image& rgba);

          /** \brief Set the tolerance for the delta on the Hue in Seeded Hue Segmentation step */
          inline void
          setDeltaHueTolerance (unsigned int delta_hue_tolerance)
          {
            delta_hue_tolerance_ = delta_hue_tolerance;
          }

          /** \brief Get the tolerance for the delta on the Hue in Seeded Hue Segmentation step, defaults to 5 */
          inline unsigned int
          getDeltaHueTolerance () const
          {
            return (delta_hue_tolerance_);
          }

          /** \brief Class getName method. */
          inline const std::string
          getClassName () const
          {
            return "PeopleDetector";
          }

          typedef DeviceArray2D<unsigned char> Labels;
          typedef DeviceArray2D<unsigned char> Mask;
          typedef DeviceArray2D<float> Hue;

          /** \brief indicates first time callback (allows for tracking features to start from second frame) **/
          bool first_iteration_;
          float fx_, fy_, cx_, cy_;
          unsigned int delta_hue_tolerance_;

          DeviceArray<unsigned char> kernelRect5x5_;

          PointCloud<PointT> cloud_host_;
          PointCloud<PointTC> cloud_host_color_;
          PointCloud<float> hue_host_;
          PointCloud<unsigned short> depth_host_;
          PointCloud<unsigned char> flowermat_host_;

          DeviceArray2D<PointT> cloud_device_;

          Hue hue_device_;

          Depth depth_device1_;
          Depth depth_device2_;

          Mask fg_mask_;
          Mask fg_mask_grown_;

          //skeleton
          Eigen::Vector4f skeleton_joints_[num_parts_all];
          //Used for tracking
          Eigen::Vector4f joints_velocity_[num_parts_all];
          Eigen::Vector4f skeleton_joints_prev_[num_parts_all];
          Eigen::Vector4f mean_vals_;
          bool active_tracking_;
          float dt_;
          float alpha_tracking_;
          float beta_tracking_;

          Blob2 skeleton_blobs_[num_parts_all];

          void
          alphaBetaTracking ();

          /**
           *Goes recursively through all the children of the tree and calculate the positions of the joints
           *@param[in] tree the blob tree which we search to estimate the optimal blob for every joint
           *@param[in] part_label label of the part we a building the tree from (e.g. Neck)
           *@param[in] id of the blob from which we build the tree (usually 0. There might be several blobs with the same label, they are sorted by the size.)
           */
          int
          estimateJoints (const std::vector<std::vector<Blob2, Eigen::aligned_allocator<Blob2> > >& sorted,
                          Tree2& tree,
                          int part_label,
                          int part_lid);

          /**
           * Calculates/corrects the position of following joints:
           * Shoulders, ELbows(if the Elbow-Blob is missing) and hips
           */
          void
          calculateAddtionalJoints ();


          void
          getMaxDist (const pcl::PointCloud<PointXYZ> &cloud,
                      Eigen::Vector4f &pivot_pt,
                      Eigen::Vector4f &max_pt);

          int
          process ();

          /**
           * Calculates 2D coordinates of the joint
           * @param[in] point_3d 3d position of the joint
           * @return 2D coordinates
           */
          Eigen::Vector3f
          project3dTo2d (Eigen::Vector4f point_3d);

          /**
           * \brief Process the depth based on probabilities supporting tracking, person specific files used
           **/
          int
          processProb ();

          void
          allocate_buffers (int rows = 480,
                            int cols = 640);

          void
          shs5 (const pcl::PointCloud<PointT> &cloud,
                const std::vector<int>& indices,
                unsigned char *mask);

          //!!! only for debug purposes TODO: remove this. 
          //friend class PeoplePCDApp;
      };
    }
  }
}
#endif // PCL_GPU_PEOPLE_PERSON_H_
