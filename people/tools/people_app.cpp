/**
 *  Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
 *
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
 * $Id: $
 * @brief This file is the execution node of the Human Tracking
 * @copyright Copyright (2011) Willow Garage
 * @authors Koen Buys, Anatoly Baksheev
 **/
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/time.h>
#include <pcl/exceptions.h>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/gpu/containers/initialization.h>
#include <pcl/gpu/people/people_detector.h>
#include <pcl/gpu/people/colormap.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <boost/filesystem.hpp>

//new includes
#include <pcl/io/lzf_image_io.h>
#include <iostream>
#include <fstream>
#include <pcl/io/image_grabber.h>
#include <pcl/gpu/people/label_common.h>
#include <ctime>
#include <iostream>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>

//for resampling 

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/mls.h>

#include <math.h>
#include <string.h>

namespace pc = pcl::console;
using namespace pcl::visualization;
using namespace pcl::gpu;
using namespace pcl::gpu::people;
using namespace pcl;
using namespace std;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

enum
{
  COLS = 640, ROWS = 480
};

//IO variables
io::LZFDepth16ImageWriter ld;
io::LZFRGB24ImageWriter lrgb;
int lzf_fps = 30;
string lzf_dir_global = "";
string pcd_file_global = "";
string pcd_dir_global = "";
bool islzf = false;

//declarations for people detector on ground plane

bool new_cloud_available_flag_ground_plane = false;
pcl::visualization::PCLVisualizer viewer_ground_plane ("PCL Viewer");

pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector_ground_plane;    // people detection object
// Mutex: //
boost::mutex cloud_mutex_ground_plane;
pcl::PointIndices::Ptr inds (new pcl::PointIndices);
bool valid_inds[COLS * ROWS];
bool detected = false;

bool fill_holes = true;
bool correct_calibration = false;
PointCloudT::Ptr cloud_ground_plane (new PointCloudT);

//flags:
bool segment_people = true;//false;    //flag
bool resample = false;    //flag
//args
std::string svm_filename = "Data/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
Eigen::VectorXf ground_coeffs;

vector<string>
getPcdFilesInDir (const string& directory)
{
  namespace fs = boost::filesystem;
  fs::path dir (directory);

  if (!fs::exists (dir) || !fs::is_directory (dir))
    PCL_THROW_EXCEPTION(pcl::IOException, "Wrong PCD directory");

  vector<string> result;
  fs::directory_iterator pos (dir);
  fs::directory_iterator end;

  for (; pos != end; ++pos)
    if (fs::is_regular_file (pos->status ()))
      if (fs::extension (*pos) == ".pcd")
        result.push_back (pos->path ().string ());

  return result;
}

//not used yet, resampling directly on the depth image in source_cb1_resample
void
resample_point_cloud (PointCloud<PointXYZRGBA>::Ptr cloud_src,
                      PointCloud<PointXYZRGBA>::Ptr cloud_target)
{

  //pcl::PointCloud<PointXYZRGBA>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

  // Fill in the cloud data

  pcl::copyPointCloud (*cloud_src, *cloud_target);

  cloud_target->width = COLS;
  cloud_target->height = ROWS;
  cloud_target->points.resize (cloud_target->width * cloud_target->height);

  // Create a KD-Tree
  // pcl::search::KdTree<PointXYZRGBA>::Ptr tree (new pcl::search::KdTree<PointXYZRGBA>);
  // Init object (second point type is for the normals, even if unused)
  // pcl::MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA> mls;

  //mls.setComputeNormals (true);
  /*
   // Set parameters
   mls.setInputCloud (cloud_src);
   mls.setPolynomialFit (true);
   mls.setSearchMethod (tree);
   mls.setSearchRadius (0.03);
   mls.setUpsamplingMethod (MovingLeastSquares<pcl::PointXYZRGBA, pcl::PointXYZRGBA>::SAMPLE_LOCAL_PLANE);
   mls.setUpsamplingRadius (0.025);
   mls.setUpsamplingStepSize (0.01);

   // Reconstruct
   mls.process ( *cloud_target);
   */

  int factor_h = ROWS / cloud_src->height;
  int factor_w = COLS / cloud_src->width;

  for (size_t x = 0; x < COLS; ++x)
  {
    for (size_t y = 0; y < ROWS; ++y)
    {
      int y_old = y / factor_h;
      int x_old = x / factor_w;
      int position_old = y_old * cloud_src->width + x_old;
      //unsigned short depth_val=static_cast<unsigned short>(cloud_host_.points[position_old].z )*1000.0; //m -> mm
      //depth_host_.points[r*COLS+c]=depth_val;
      cloud_target->points[y * COLS + x].z = cloud_src->points[position_old].z;
    }
  }

}


/**
 * Cloud callback for Ground Plane People Detector
 */
void
cloud_cb_2 (const PointCloudT::ConstPtr &callback_cloud,
            PointCloudT::Ptr& cloud,
            bool* new_cloud_available_flag)
{

  cloud_mutex_ground_plane.lock ();    // for not overwriting the point cloud from another thread
  //*cloud = *callback_cloud;
  pcl::copyPointCloud (*callback_cloud, *cloud);
//resampling
  /*
   pcl::PointCloud<PointXYZRGBA>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZRGBA>);
   pcl::copyPointCloud(*callback_cloud, *cloud_src);
   pcl::PointCloud<PointXYZRGBA>::Ptr cloud_resampled (new pcl::PointCloud<pcl::PointXYZRGBA>);
   resample_point_cloud(cloud_src,cloud_resampled);

   */
  // pcl::copyPointCloud(*cloud_resampled, *cloud);
  *new_cloud_available_flag = true;
  cloud_mutex_ground_plane.unlock ();
}

struct callback_args
{
    // structure used to pass arguments to the callback function
    PointCloudT::Ptr clicked_points_3d;
    pcl::visualization::PCLVisualizer::Ptr viewerPtr;

};

void
pp_callback (const pcl::visualization::PointPickingEvent& event,
             void* args)
{
  struct callback_args* data = (struct callback_args *) args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint (current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back (current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud ("clicked_points");
  data->viewerPtr->addPointCloud (data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct SampledScopeTime : public StopWatch
{
    enum
    {
      EACH = 33
    };
    SampledScopeTime (int& time_ms) :
        time_ms_ (time_ms)
    {
    }
    ~SampledScopeTime ()
    {
      static int i_ = 0;
      time_ms_ += getTime ();
      if (i_ % EACH == 0 && i_)
      {
        cout << "[~SampledScopeTime] : Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << endl;
        time_ms_ = 0;
      }
      ++i_;
    }
  private:
    int& time_ms_;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

string
make_name (int counter,
           const char* suffix)
{
  char buf[4096];
  sprintf (buf, "./people%04d_%s.png", counter, suffix);
  return buf;
}

template <typename T> void
savePNGFile (const std::string& filename,
             const pcl::gpu::DeviceArray2D<T>& arr)
{
  int c;
  pcl::PointCloud<T> cloud (arr.cols (), arr.rows ());
  arr.download (cloud.points, c);
  pcl::io::savePNGFile (filename, cloud);
}

template <typename T> void
savePNGFile (const std::string& filename,
             const pcl::PointCloud<T>& cloud)
{
  pcl::io::savePNGFile (filename, cloud);
}

/////////////////////////////TRACKER///////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PeoplePCDApp
{
  public:
    typedef pcl::gpu::people::PeopleDetector PeopleDetector;
    ofstream skeleton_file;
	ofstream Larm_file, Lforearm_file, Lfoot_file, Lleg_file, Lknee_file, Lthigh_file, Lhips_file, Lelbow_file, Lhand_file;
	ofstream Rarm_file, Rforearm_file, Rfoot_file, Rleg_file, Rknee_file, Rthigh_file, Rhips_file, Relbow_file, Rhand_file;
	ofstream Neck_file, FaceLB_file, FaceRB_file, FaceLT_file, FaceRT_file, Rchest_file, Lchest_file;
	
    enum
    {
      COLS = 640, ROWS = 480
    };

    PeoplePCDApp (pcl::Grabber& capture,
                  bool write) :
        capture_ (capture),
        write_ (write),
        exit_ (false),
        time_ms_ (0),
        cloud_cb_ (true),
        counter_ (0),
        final_view_ ("Final labeling"),
        depth_view_ ("Depth")

    {
      final_view_.setSize (COLS, ROWS);
      depth_view_.setSize (COLS, ROWS);

      final_view_.setPosition (0, 400);
      depth_view_.setPosition (650, 400);

      cmap_device_.create (ROWS, COLS);
      cmap_host_.points.resize (COLS * ROWS);
      depth_device_.create (ROWS, COLS);
      image_device_.create (ROWS, COLS);

      depth_host_.points.resize (COLS * ROWS);

      rgba_host_.points.resize (COLS * ROWS);
      rgb_host_.resize (COLS * ROWS * 3);

      pcl::gpu::people::uploadColorMap (color_map_);

    }

    /**
     * Adds a line to Depth and RGB views between the joints parent and child
     */
    int
    drawLimb (int parent,
              int child)
    {

      Eigen::Vector4f j1 = people_detector_.skeleton_joints_[parent];
      Eigen::Vector4f j2 = people_detector_.skeleton_joints_[child];
      if (j1[0] != -1 && j2[0] != -1 && abs (double (j1[2])) < 100.0 && abs (double (j2[2])) < 100.0 && abs (double (j1[1])) < 100.0
          && abs (double (j2[1])) < 100.0)
      {
        Eigen::Vector3f j_projected_parent = people_detector_.project3dTo2d (j1);
        Eigen::Vector3f j_projected_child = people_detector_.project3dTo2d (j2);

        depth_view_.addLine ((int) j_projected_parent[0], (int) j_projected_parent[1], (int) j_projected_child[0], (int) j_projected_child[1], 200, 0, 0,
                             "limbs", 5000);
        final_view_.addLine ((int) j_projected_parent[0], (int) j_projected_parent[1], (int) j_projected_child[0], (int) j_projected_child[1], 200, 0, 0,
                             "limbs", 5000);

        depth_view_.addCircle ((int) j_projected_parent[0], (int) j_projected_parent[1], 5.0, "circle", 5000);
        final_view_.addCircle ((int) j_projected_parent[0], (int) j_projected_parent[1], 5.0, "circle", 5000);
        depth_view_.addCircle ((int) j_projected_child[0], (int) j_projected_child[1], 5.0, "circle", 5000);
        final_view_.addCircle ((int) j_projected_child[0], (int) j_projected_child[1], 5.0, "circle", 5000);

        return 1;
      }
      return -1;

    }

    /*
     * Draws the whole skeleton
     */
    void
    drawAllLimbs ()
    {
      int i = 0;

      final_view_.removeLayer ("limbs");
      depth_view_.removeLayer ("limbs");
      // Iterate over all parts

      drawLimb (Neck, FaceRB);
      drawLimb (Neck, FaceLB);
      drawLimb (Neck, Rshoulder);
      drawLimb (Neck, Lshoulder);
      // drawLimb( Rshoulder, Rarm);
      // drawLimb( Lshoulder, Larm);
      drawLimb (Rshoulder, Rchest);
      drawLimb (Lshoulder, Lchest);
      drawLimb (Rchest, Rhips);
      drawLimb (Lchest, Lhips);
      drawLimb (Lhips, Lknee);
      drawLimb (Rhips, Rknee);

      drawLimb (Lknee, Lfoot);
      drawLimb (Rknee, Rfoot);

      drawLimb (Rshoulder, Relbow);
      drawLimb (Relbow, Rhand);
      drawLimb (Lshoulder, Lelbow);
      drawLimb (Lelbow, Lhand);
      drawLimb (FaceLB, FaceLT);

      drawLimb (FaceRB, FaceRT);

    }

	// Calculate the angle between limbs
	double 
	angleBetweenTwoVectors(int joint1, int joint2, int joint3)
    {
		Eigen::Vector4f j1 = people_detector_.skeleton_joints_[joint1];
		Eigen::Vector4f j2 = people_detector_.skeleton_joints_[joint2];
		Eigen::Vector4f j3 = people_detector_.skeleton_joints_[joint3];

		Eigen::Vector4f j1_ = j1 - j2;
		Eigen::Vector4f j2_ = j1 - j3;

		if ((j1_[0] != -1) && (j2_[0] !=-1))
        {
			double len_j1 = j1_.norm();
			double len_j2 = j2_.norm();
		
			Eigen::Vector4f norm_j1 = j1_/len_j1;
			Eigen::Vector4f norm_j2 = j2_/len_j2;
		
			double dotProduct = norm_j1.dot(norm_j2);
			double res = (double)acos(dotProduct)/M_PI*180;
			
			return res;
		}

		return (double) -1;
    }

	void cloud2disk(const pcl::PointCloud<PointXYZ> &ptr, std::ofstream& pointcloud_file, std::string bodypart)
	{
		std::stringstream cnt;
		cnt << counter_;
		std::string count = cnt.str();

		std::string filename = count + "_" + bodypart + ".txt";
		pointcloud_file.open (filename, std::ios::app);
		for (int c = 0 ; c < ptr.points.size() ; c++)
			pointcloud_file << ptr.points[c].x  << "," << ptr.points[c].y  << "," << ptr.points[c].z << "," << 1 << "\n";
		pointcloud_file.close();
	}

	void storeFilesToDisk(const pcl::PointCloud<PointXYZ> &LarmPtr, const pcl::PointCloud<PointXYZ> &LforearmPtr, const pcl::PointCloud<PointXYZ> &RarmPtr, const pcl::PointCloud<PointXYZ> &RforearmPtr)
	{
		ofstream pointcloud_file;
		std::stringstream cnt;
		cnt << counter_;
		std::string count = cnt.str();

		std::string filename_Larm = count + "_Larm.txt";
		pointcloud_file.open (filename_Larm, std::ios::app);
		for (int c = 0 ; c < LarmPtr.points.size() ; c++)
			pointcloud_file << LarmPtr.points[c].x  << "," << LarmPtr.points[c].y  << "," << LarmPtr.points[c].z << "," << 1 << "\n";
		pointcloud_file.close();

		std::string filename_Lforearm = count + "_Lforearm.txt";
		pointcloud_file.open (filename_Lforearm, std::ios::app);
		for (int c = 0 ; c < LforearmPtr.points.size() ; c++)
			pointcloud_file << LforearmPtr.points[c].x  << "," << LforearmPtr.points[c].y  << "," << LforearmPtr.points[c].z << "," << 1 << "\n";
		pointcloud_file.close();

		std::string filename_Rarm = count + "_Rarm.txt";
		pointcloud_file.open (filename_Rarm, std::ios::app);
		for (int c = 0 ; c < RarmPtr.points.size() ; c++)
			pointcloud_file << RarmPtr.points[c].x  << "," << RarmPtr.points[c].y  << "," << RarmPtr.points[c].z << "," << 1 << "\n";
		pointcloud_file.close();

		std::string filename_Rforearm = count + "_Rforearm.txt";
		pointcloud_file.open (filename_Rforearm, std::ios::app);
		for (int c = 0 ; c < RforearmPtr.points.size() ; c++)
			pointcloud_file << RforearmPtr.points[c].x  << "," << RforearmPtr.points[c].y  << "," << RforearmPtr.points[c].z << "," << 1 << "\n";
		pointcloud_file.close();
	}

	void storeFiles()
	{
		const pcl::PointCloud<pcl::PointXYZ>::Ptr LarmPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Larm].indices.indices));
		cloud2disk(*LarmPtr, Larm_file, "LeftArm");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr LforearmPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lforearm].indices.indices));
		cloud2disk(*LforearmPtr, Lforearm_file, "LeftForearm");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RarmPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rarm].indices.indices));
		cloud2disk(*RarmPtr, Rarm_file, "RightArm");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RforearmPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rforearm].indices.indices));
		cloud2disk(*RforearmPtr, Rforearm_file, "RightForeArm");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LfootPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lfoot].indices.indices));
		cloud2disk(*LfootPtr, Lfoot_file, "LeftFoot");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr LlegPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lleg].indices.indices));
		cloud2disk(*LlegPtr, Lleg_file, "LeftLeg");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr RfootPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rfoot].indices.indices));
		cloud2disk(*RfootPtr, Rfoot_file, "RightFoot");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RlegPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rleg].indices.indices));
		cloud2disk(*RlegPtr, Rleg_file, "RightLeg");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LkneePtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lknee].indices.indices));
		cloud2disk(*LkneePtr, Lknee_file, "LeftKnee");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RkneePtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rknee].indices.indices));
		cloud2disk(*RkneePtr, Rknee_file, "RightKnee");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LthighPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lthigh].indices.indices));
		cloud2disk(*LthighPtr, Lthigh_file, "LeftThigh");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RthighPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rthigh].indices.indices));
		cloud2disk(*RthighPtr, Rthigh_file, "RightThigh");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LhipsPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lhips].indices.indices));
		cloud2disk(*LhipsPtr, Lhips_file, "LeftHip");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RhipsPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rhips].indices.indices));
		cloud2disk(*RhipsPtr, Rhips_file, "RightHip");

//		const pcl::PointCloud<pcl::PointXYZ>::Ptr LelbowPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lelbow].indices.indices));
//		cloud2disk(*LelbowPtr, Lelbow_file, "LeftElbow");
//		const pcl::PointCloud<pcl::PointXYZ>::Ptr RelbowPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Relbow].indices.indices));
//		cloud2disk(*RelbowPtr, Relbow_file, "RightElbow");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LhandPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lhand].indices.indices));
		cloud2disk(*LhandPtr, Lhand_file, "LeftHand");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RhandPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rhand].indices.indices));
		cloud2disk(*RhandPtr, Rhand_file, "RightHand");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr NeckPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Neck].indices.indices));
		cloud2disk(*NeckPtr, Neck_file, "Neck");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr FaceLBPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[FaceLB].indices.indices));
		cloud2disk(*FaceLBPtr, FaceLB_file, "FaceLB");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr FaceRBPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[FaceRB].indices.indices));
		cloud2disk(*FaceRBPtr, FaceRB_file, "FaceRB");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr FaceLTPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[FaceLT].indices.indices));
		cloud2disk(*FaceLTPtr, FaceLT_file, "FaceLT");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr FaceRTPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[FaceRT].indices.indices));
		cloud2disk(*FaceRTPtr, FaceRT_file, "FaceRT");

		const pcl::PointCloud<pcl::PointXYZ>::Ptr LchestPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Lchest].indices.indices));
		cloud2disk(*LchestPtr, Lchest_file, "Lchest");
		const pcl::PointCloud<pcl::PointXYZ>::Ptr RchestPtr (new pcl::PointCloud<pcl::PointXYZ> (people_detector_.cloud_host_, people_detector_.skeleton_blobs_[Rchest].indices.indices));
		cloud2disk(*RchestPtr, Rchest_file, "Rchest");
	}

	void
    visualizeAndWrite ()
    {
      const PeopleDetector::Labels& labels = people_detector_.rdf_detector_->getLabels ();
	  const pcl::gpu::people::RDFBodyPartsDetector::BlobMatrix& blobs = people_detector_.rdf_detector_->getBlobMatrix();
	  //getLineEq("Lelbow");
	  
      pcl::gpu::people::colorizeLabels (color_map_, labels, cmap_device_);
      //people::colorizeMixedLabels(

      int c;
      cmap_host_.width = cmap_device_.cols ();
      cmap_host_.height = cmap_device_.rows ();
      cmap_host_.points.resize (cmap_host_.width * cmap_host_.height);
      cmap_device_.download (cmap_host_.points, c);
      final_view_.removeLayer ("circle");
      depth_view_.removeLayer ("circle");
      final_view_.addRGBImage<pcl::RGB> (cmap_host_);
      part_t wanted_joints[] = { Rhand, Lhand, Relbow, Lelbow };
      int num_joints = sizeof (wanted_joints) / sizeof (wanted_joints)[0];

      for (int i = 0; i < 27; i++)
      {

        Eigen::Vector4f j = people_detector_.skeleton_joints_[i];
        if (j[0] != -1)
        {
          Eigen::Vector3f j_projected = people_detector_.project3dTo2d (j);
          // depth_view_.addCircle((int)j_projected[0],(int)j_projected[1],10.0,"circle",5000);
          // final_view_.addCircle((int)j_projected[0],(int)j_projected[1],10.0,"circle",5000);

        }
      }

      drawAllLimbs ();
	  //double leftElbowAngle = angleBetweenTwoVectors(Lelbow, Lshoulder, Lhand);
	  //double rightElbowAngle = angleBetweenTwoVectors(Relbow, Rshoulder, Rhand);
	  
	  /*Show the angle on screen*/
	  //char leftElbowString[40];
	  //char rightElbowString[40];
	  //sprintf(leftElbowString, "Left elbow degrees: %.2f", leftElbowAngle);
  	  //sprintf(rightElbowString, "Right elbow degrees: %.2f", rightElbowAngle);
	  //final_view_.removeLayer("text");

	 // final_view_.addText ((int) 10, (int) 450, rightElbowString, 200, 0, 0, "text", 10);
	  //final_view_.addText ((int) 460, (int) 450, leftElbowString, 200, 0, 0, "text", 10);

      final_view_.spinOnce (1, true);
	  savePNGFile(make_name(counter_, "labels_NEW_DATA"), labels);  
	  storeFiles();

      if (cloud_cb_)
      {
        depth_host_.width = people_detector_.depth_device1_.cols ();
        depth_host_.height = people_detector_.depth_device1_.rows ();
        depth_host_.points.resize (depth_host_.width * depth_host_.height);
        people_detector_.depth_device1_.download (depth_host_.points, c);
      }

      //depth_view_.addShortImage (&depth_host_.points[0], depth_host_.width, depth_host_.height, 0, 4000, true);

      //depth_view_.spinOnce (1, true);

	  write_ = true;
      if (write_)
      {
        PCL_VERBOSE("PeoplePCDApp::visualizeAndWrite : (I) : Writing to disk");
        //We are not saving the PNG files right now
        /*
         if (cloud_cb_)
         savePNGFile(make_name(counter_, "ii"), cloud_host_);
         else
         savePNGFile(make_name(counter_, "ii"), rgba_host_);
         savePNGFile(make_name(counter_, "c2"), cmap_host_);
         savePNGFile(make_name(counter_, "s2"), labels);
         savePNGFile(make_name(counter_, "d1"), people_detector_.depth_device1_);
         savePNGFile(make_name(counter_, "d2"), people_detector_.depth_device2_);*/
		savePNGFile(make_name(counter_, "c2"), cmap_host_);

		std::stringstream cnt;
		cnt << counter_;
		std::string count = cnt.str();
		std::string skel_fname = count + "_skeleton.txt";

        skeleton_file.open (skel_fname, std::ios::app);
        skeleton_file << "\n" << counter_ << " " << time_ms_;
        skeleton_file << "\n";
        for (int i = 0; i < 27; i++)
        {

          Eigen::Vector4f j = people_detector_.skeleton_joints_[i];

          skeleton_file << j[0] << "," << j[1] << "," << j[2] << "\n";
          //skeleton_file << "\n";

        }
        skeleton_file << " ";
        skeleton_file.close ();
      }
    }

    /*Segments the body in depth_host_
     *Sets all depth values that do not belong to the body to 10000
     *In the end the segmented depth image is uploaded to depth_device
     */
    void
    segment_body_depth ()
    {

      unsigned short depth_invalid_value = 10000;
      int h = ROWS;
      int w = COLS;

      //if a person was tracked: resetting the valid indicies array
      if (detected)
        for (int i = 0; i < depth_host_.points.size (); i++)
          valid_inds[i] = false;

      //minimum and maximum values
      Eigen::Vector4f min;
      Eigen::Vector4f max;
      pcl::getMinMax3D (*people_detector_ground_plane.getNoGroundCloud (), inds->indices, min, max);
      Eigen::Vector3f min_2d = people_detector_.project3dTo2d (min);
      Eigen::Vector3f max_2d = people_detector_.project3dTo2d (max);
      float mean_x = (min_2d[0] + max_2d[0]) / 2.0;
      float mean_y = (min_2d[1] + max_2d[1]) / 2.0;

      //variables for depth filtering
      float mean_depth = 0.0;
      int sum_depth = 0;
      int max_y = 0;

      //going through all valid indicies and filling valid_inds if a person was detected
      for (int i = 0; i < inds->indices.size () && detected; i++)
      {

        pcl::PointXYZRGBA point = people_detector_ground_plane.getNoGroundCloud ()->points[inds->indices[i]];
        //projecting to 2D
        float x = point.x;
        float y = point.y;
        float z = point.z;
        Eigen::Vector4f temp;
        temp[0] = x;
        temp[1] = y;
        temp[2] = z;
        temp[3] = 1;
        //resulting 2D point
        Eigen::Vector3f projected = people_detector_.project3dTo2d (temp);

        //estimating the highest Y-value to add points to the legs
        if (projected[1] > max_y)
          max_y = (int) projected[1];

        //setting the corresponding values
        if ( ((int) projected[1]) < ROWS && ((int) projected[0]) < COLS && ((int) projected[1]) > 0 && ((int) projected[0]) > 0)
        {
          valid_inds[ ((int) projected[1]) * w + ((int) projected[0])] = true;
        }

        //hole filling part 1
        float mean_x = (min_2d[0] + max_2d[0]) / 2.0;
        float mean_y = (min_2d[1] + max_2d[1]) / 2.0;
        int min_j = -20;
        int max_j = 20;
        if ((int) projected[0] < mean_x)
          min_j = 0;
        if ((int) projected[0] > mean_x)
          max_j = 0;
        int min_k = -5;
        int max_k = 5;
        if ((int) projected[1] < mean_y)
          min_k = 0;
        if ((int) projected[1] > mean_y)
          max_k = 0;

        for (int j = -10; j < 10; j++)
        {
          for (int k = -10; k < 10; k++)
          {
            if ( ((int) projected[1] + k) < ROWS && ((int) projected[0]) + j < COLS && ((int) projected[1] + k) > 0 && ((int) projected[0]) + j > 0)
            {
              valid_inds[ ((int) projected[1] + k) * w + ((int) projected[0]) + j] = true;
            }
          }

          //for  depth filtering
          mean_depth += depth_host_.points[ ((int) projected[1]) * w + ((int) projected[0])];
          sum_depth++;

        }         	//end of hole filling
      }         	//end of filling valid_inds

      //closing the gaps: rows

      for (int i = 0; i < h; i++)
      {
        int first = -1;
        int last = 0;
        for (int j = 0; j < w - 1; j++)
        {
          if ( (valid_inds[i * w + j]))
          {
            if (first == -1)
            {
              first = j;
            }
            last = j;
          }

        }
        //filling all the indicies between first and last in the current row
        for (int j = first; j < last && first != -1; j++)
        {

          valid_inds[i * w + j] = true;

        }
      }

      //closing the gaps: cols
      for (int i = 0; i < w; i++)
      {
        int first = 0;
        int last = 0;
        for (int j = 0; j < h; j++)
        {
          if (valid_inds[j * w + i])
          {
            if (first == 0)
            {
              first = j;
            }
            last = j;
          }

        }

        //std::cout <<"max_y"<< max_y<< std::endl;
        //first=last;
        if (max_y != 0 && max_y - last < 40)
          if (last + 25 < h - 1)
            last = last + 25;
          else
            last = h - 2;

        //filling all the indicies between first and last in the current column
        for (int j = first; j < last; j++)
        {

          valid_inds[j * w + i] = true;

        }

      }

//Depth filtering

      float depth_thresh = 250.0f;
      mean_depth = mean_depth / sum_depth;

      for (int i = 0; i < ROWS * COLS; i++)
        if ( (depth_host_.points[i] - mean_depth) > depth_thresh)
          valid_inds[i] = false;

//Filling the depth image
      for (int i = 0; i < ROWS * COLS; i++)
        if (valid_inds[i] == false)
          depth_host_.points[i] = depth_invalid_value;

    }

    void
    source_cb1_resampled (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >& cloud)
    {

      {

        boost::mutex::scoped_lock lock (data_ready_mutex_);
        if (exit_)
          return;
        //cloud->is_dense = false;
        //cloud->resize(COLS*ROWS);
        pcl::PointCloud<PointXYZRGBA>::Ptr cloud_src (new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::copyPointCloud (*cloud, *cloud_src);
        pcl::PointCloud<PointXYZRGBA>::Ptr cloud_resampled (new pcl::PointCloud<pcl::PointXYZRGBA>);
        //resample_point_cloud(cloud_src,cloud_resampled);
        //pcl::copyPointCloud(*cloud_resampled, cloud_host_);
        pcl::copyPointCloud (*cloud, cloud_host_);

        int factor_h = ROWS / cloud->height;
        int factor_w = COLS / cloud->width;

        double depth_mean = 0.0;
        int depth_sum = 0;
        for (size_t c = 0; c < COLS; ++c)
        {
          for (size_t r = 0; r < ROWS; ++r)
          {
            int r_old = r / factor_h;
            int c_old = c / factor_w;
            int position_old = r_old * cloud->width + c_old;
            unsigned short depth_val = static_cast<unsigned short> (cloud_host_.points[position_old].z) * 1000.0;  //m -> mm
            /* If we want some smoothing (did not improve anything)

             if (r_old>0 && r_old<cloud->height-1 &&c_old>0 && c_old<cloud->width-1){
             depth_val+=static_cast<unsigned short>(cloud_host_.points[position_old+1].z )*1000.0;
             depth_val+=static_cast<unsigned short>(cloud_host_.points[position_old-1].z )*1000.0;
             depth_val+=static_cast<unsigned short>(cloud_host_.points[position_old+cloud->width].z )*1000.0;
             depth_val+=static_cast<unsigned short>(cloud_host_.points[position_old-cloud->width].z )*1000.0;
             depth_val=depth_val/6.0;
             }
             */
            depth_host_.points[r * COLS + c] = depth_val;

            if (ROWS - r < 110 || r < 50 || COLS - c < 200 || c < 200)
            {
              depth_host_.points[r * COLS + c] = 10000;
            }
            else
            {
              depth_mean += depth_val;
              depth_sum++;
            }

          }
        }
        depth_mean = depth_mean / depth_sum;

        for (size_t i = 0; i < depth_host_.points.size (); ++i)
        {
          if (depth_host_.points[i] > depth_mean - 200)
          {
            depth_host_.points[i] = 10000;  //m -> mm
          }

        }

        int w = COLS;
        int h = ROWS;
        //uploading the data
        int s = w * PeopleDetector::Depth::elem_size;
        depth_device_.upload (&depth_host_.points[0], s, h, w);
        s = w * PeopleDetector::Image::elem_size;
        image_device_.upload (&rgba_host_.points[0], s, h, w);

      }
      data_ready_cond_.notify_one ();
    }

    void
    source_cb1 (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >& cloud)
    {
      {
        boost::mutex::scoped_lock lock (data_ready_mutex_);
        if (exit_)
          return;

        pcl::copyPointCloud (*cloud, cloud_host_);
      }
      data_ready_cond_.notify_one ();
    }

    void
    source_cb2 (const boost::shared_ptr<openni_wrapper::Image>& image_wrapper,
                const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper,
                float)
    {

      {

        boost::mutex::scoped_try_lock lock (data_ready_mutex_);

        if (exit_ || !lock)
          return;

        //getting depth
        int w = depth_wrapper->getWidth ();
        int h = depth_wrapper->getHeight ();
        int s = w * PeopleDetector::Depth::elem_size;
        const unsigned short *data = depth_wrapper->getDepthMetaData ().Data ();

        depth_host_.points.resize (w * h);
        depth_host_.width = w;
        depth_host_.height = h;
        std::copy (data, data + w * h, &depth_host_.points[0]);
        w = image_wrapper->getWidth ();
        h = image_wrapper->getHeight ();

        if (inds->indices.size () > 0 && segment_people)
          segment_body_depth ();

        depth_device_.upload (&depth_host_.points[0], s, h, w);
        //getting image

        //fill rgb array
        rgb_host_.resize (w * h * 3);
        image_wrapper->fillRGB (w, h, (unsigned char*) &rgb_host_[0]);

        // convert to rgba, TODO image_wrapper should be updated to support rgba directly
        rgba_host_.points.resize (w * h);
        rgba_host_.width = w;
        rgba_host_.height = h;
        s = w * PeopleDetector::Image::elem_size;
        for (int i = 0; i < rgba_host_.size (); ++i)
        {
          const unsigned char *pixel = &rgb_host_[i * 3];
          RGB& rgba = rgba_host_.points[i];
          rgba.r = pixel[0];
          rgba.g = pixel[1];
          rgba.b = pixel[2];
        }
        image_device_.upload (&rgba_host_.points[0], s, h, w);
      }
      data_ready_cond_.notify_one ();

    }

    void
    source_cb3 (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >& cloud)
    {

      {
        boost::mutex::scoped_lock lock (data_ready_mutex_);
        if (exit_)
          return;

        pcl::copyPointCloud (*cloud, cloud_host_);
        pcl::copyPointCloud (*cloud, rgba_host_);

        for (size_t i = 0; i < cloud->points.size (); ++i)
        {

          depth_host_.points[i] = static_cast<unsigned short> (cloud_host_.points[i].z * 1000);  //m -> mm
        }

        if (inds->indices.size () > 0 && segment_people)
          segment_body_depth ();
        int w = COLS;
        int h = ROWS;
        //uploading the data
        int s = w * PeopleDetector::Depth::elem_size;
        depth_device_.upload (&depth_host_.points[0], s, h, w);
        s = w * PeopleDetector::Image::elem_size;
        image_device_.upload (&rgba_host_.points[0], s, h, w);

      }
      data_ready_cond_.notify_one ();
    }

    void
    startMainLoop ()
    {

      //--------People detection------

      // Algorithm parameters:

      float min_confidence = -2.0;
      float min_height = 0.6;
      float max_height = 1.9;

      float voxel_size = 0.06;

      float sampling_factor = 1;
      Eigen::Matrix3f rgb_intrinsics_matrix;
      rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0;  // Kinect RGB camera intrinsics

      // Read Kinect live stream:

      PCDGrabberBase* ispcd = dynamic_cast<pcl::PCDGrabberBase*> (&capture_);

      boost::function<void
      (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f = boost::bind (&cloud_cb_2, _1, cloud_ground_plane, &new_cloud_available_flag_ground_plane);

      capture_.registerCallback (f);
      capture_.start ();
      // Wait for the first frame:
      while (!new_cloud_available_flag_ground_plane)
        boost::this_thread::sleep (boost::posix_time::milliseconds (1));
      new_cloud_available_flag_ground_plane = false;

      cloud_mutex_ground_plane.lock ();    // for not overwriting the point cloud

      // Display pointcloud:
      pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb (cloud_ground_plane);

      viewer_ground_plane.addPointCloud<PointT> (cloud_ground_plane, rgb, "input_cloud");
      viewer_ground_plane.setCameraPosition (0, 0, -2, 0, -1, 0, 0);

      // Add point picking callback to viewer:
      struct callback_args cb_args;
      PointCloudT::Ptr clicked_points_3d (new PointCloudT);
      cb_args.clicked_points_3d = clicked_points_3d;
      cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr (&viewer_ground_plane);
      viewer_ground_plane.registerPointPickingCallback (pp_callback, (void*) &cb_args);


      // Spin until 'Q' is pressed:
      viewer_ground_plane.spin ();

      cloud_mutex_ground_plane.unlock ();

      // Ground plane estimation:
      Eigen::VectorXf ground_coeffs;
      ground_coeffs.resize (4);
      std::vector<int> clicked_points_indices;

        std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;
      for (unsigned int i = 0; i < clicked_points_3d->points.size (); i++)
        clicked_points_indices.push_back (i);
      pcl::SampleConsensusModelPlane<PointT> model_plane (clicked_points_3d);
      model_plane.computeModelCoefficients (clicked_points_indices, ground_coeffs);
      std::cout << "Ground plane: " << ground_coeffs (0) << " " << ground_coeffs (1) << " " << ground_coeffs (2) << " " << ground_coeffs (3) << std::endl;

     // viewer_ground_plane.close ();

      // Initialize new viewer:
     // pcl::visualization::PCLVisualizer viewer_ground_plane ("PCL Viewer");          // viewer initialization
      //viewer_ground_plane.setCameraPosition (0, 0, -2, 0, -1, 0, 0);

      // Create classifier for people detection:
      pcl::people::PersonClassifier<pcl::RGB> person_classifier;

      person_classifier.loadSVMFromFile (svm_filename);   // load trained SVM

      // People detection app initialization:

      people_detector_ground_plane.setVoxelSize (voxel_size);                        // set the voxel size
      people_detector_ground_plane.setIntrinsics (rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
      people_detector_ground_plane.setClassifier (person_classifier);                // set person classifier
      people_detector_ground_plane.setHeightLimits (min_height, max_height);         // set person classifier
      people_detector_ground_plane.setSamplingFactor (sampling_factor);              // set a downsampling factor to the point cloud (for increasing speed)

      people_detector_ground_plane.setMinimumDistanceBetweenHeads (1.0);
      //-----------end of people detection-----------

      cloud_cb_ = false;

      if (ispcd || islzf)
        cloud_cb_ = true;

      typedef boost::shared_ptr<openni_wrapper::DepthImage> DepthImagePtr;
      typedef boost::shared_ptr<openni_wrapper::Image> ImagePtr;

      boost::function<void
      (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >&)> func1 = boost::bind (&PeoplePCDApp::source_cb1, this, _1);
      boost::function<void
      (const ImagePtr&,
       const DepthImagePtr&,
       float constant)> func2 = boost::bind (&PeoplePCDApp::source_cb2, this, _1, _2, _3);

      boost::function<void
      (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >&)> func3 = boost::bind (&PeoplePCDApp::source_cb3, this, _1);

      boost::function<void
      (const boost::shared_ptr<const PointCloud<PointXYZRGBA> >&)> func1_resampled = boost::bind (&PeoplePCDApp::source_cb1_resampled, this, _1);

      boost::signals2::connection c;
      if (cloud_cb_ && segment_people)
      {
        c = capture_.registerCallback (func3);
      }
      else if (cloud_cb_ && resample)
      {
        c = capture_.registerCallback (func1_resampled);
      }
      else if (cloud_cb_)
      {
        c = capture_.registerCallback (func1);
      }
      else
      {

        c = capture_.registerCallback (func2);
      }

      {
        boost::unique_lock<boost::mutex> lock (data_ready_mutex_);

        try
        {
          // capture_.start ();
          while (!exit_ && !final_view_.wasStopped ())
          {

            //PEOPLE DETECTION

            if (segment_people && !resample && new_cloud_available_flag_ground_plane)    // if a new cloud is available
            {
              cloud_mutex_ground_plane.try_lock ();
              new_cloud_available_flag_ground_plane = false;

              // Perform people detection on the new cloud:
              std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters

              int shift = 8;
              for (int x = 0; x < COLS - shift; x++)
              {
                for (int y = 0; y < ROWS; y++)
                {
                  cloud_ground_plane->points[y * COLS + x].r = cloud_ground_plane->points[y * COLS + x + shift].r;
                  cloud_ground_plane->points[y * COLS + x].g = cloud_ground_plane->points[y * COLS + x + shift].g;
                  cloud_ground_plane->points[y * COLS + x].b = cloud_ground_plane->points[y * COLS + x + shift].b;

                }
              }

              people_detector_ground_plane.setInputCloud (cloud_ground_plane);
              people_detector_ground_plane.setGround (ground_coeffs);                    // set floor coefficients
              people_detector_ground_plane.compute (clusters);                           // perform people detection

              ground_coeffs = people_detector_ground_plane.getGround ();                 // get updated floor coefficients
              viewer_ground_plane.removeAllPointClouds ();
              viewer_ground_plane.removeAllShapes ();
              pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb (cloud_ground_plane);
              viewer_ground_plane.addPointCloud<PointT> (cloud_ground_plane, rgb, "input_cloud");

              unsigned int k = 0;

              float conf = -1000.0f;

              std::vector<pcl::people::PersonCluster<PointT> >::iterator it_maxconf = clusters.begin ();
              int k_maxconf = -1;
              inds->indices.clear ();
              for (std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin (); it != clusters.end (); ++it)
              {

                //selecting the cluster with maximum confidence

                //
                if (it->getPersonConfidence () >= conf)
                {
                  conf = it->getPersonConfidence ();
                  it_maxconf = it;
                  k_maxconf = k;
                }

                if (it->getPersonConfidence () > min_confidence)
                  k++;

              }
              if (conf > min_confidence)
              {

                it_maxconf->drawTBoundingBox (viewer_ground_plane, k_maxconf);
                *inds = it_maxconf->getIndices ();
              }
              detected = (k > 0);

              viewer_ground_plane.spinOnce ();

              //viewer.spinOnce();

              cloud_mutex_ground_plane.unlock ();

            }

            //NORMAl people labeling

            bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec (100));
            if (has_data)
            {
              SampledScopeTime fps (time_ms_);

              if (cloud_cb_ && !segment_people && !resample)
                process_return_ = people_detector_.process (cloud_host_.makeShared ());
              else
              {

                process_return_ = people_detector_.process (depth_device_, image_device_);

              }
              ++counter_;
            }

            if (has_data)
            {
              visualizeAndWrite ();
            }

          }

          // final_view_.spinOnce ();
        }
        catch (const std::bad_alloc& /*e*/)
        {
          cout << "Bad alloc" << endl;
        }
        catch (const std::exception& /*e*/)
        {
          cout << "Exception" << endl;
        }
		
        capture_.stop ();
      }
	  
      c.disconnect ();
    }

    boost::mutex data_ready_mutex_;
    boost::condition_variable data_ready_cond_;

    pcl::Grabber& capture_;

    bool cloud_cb_;
    bool write_;
    bool exit_;
    int time_ms_;
    int counter_;
    int process_return_;
    PeopleDetector people_detector_;
    PeopleDetector::Image cmap_device_;
    pcl::PointCloud<pcl::RGB> cmap_host_;

    PeopleDetector::Depth depth_device_;
    PeopleDetector::Image image_device_;

    pcl::PointCloud<unsigned short> depth_host_;
    pcl::PointCloud<pcl::RGB> rgba_host_;
    std::vector<unsigned char> rgb_host_;

    PointCloud<PointXYZRGBA> cloud_host_;

    ImageViewer final_view_;
    ImageViewer depth_view_;

    DeviceArray<pcl::RGB> color_map_;
};

void
print_help ()
{
  cout << "\nPeople tracking app options (help):" << endl;
  cout << "\t -numTrees    \t<int> \tnumber of trees to load" << endl;
  cout << "\t -tree0       \t<path_to_tree_file>" << endl;
  cout << "\t -tree1       \t<path_to_tree_file>" << endl;
  cout << "\t -tree2       \t<path_to_tree_file>" << endl;
  cout << "\t -tree3       \t<path_to_tree_file>" << endl;
  cout << "\t -gpu         \t<GPU_device_id>" << endl;
  cout << "\t -w           \t<bool> \tWrite the results of skeleton tracking in skeleton.txt" << endl;
  cout << "\t -h           \tPrint this help" << endl;
  cout << "\t -dev         \t<Kinect_device_id>" << endl;
  cout << "\t -pcd         \t<path_to_pcd_file>" << endl;
  cout << "\t -oni         \t<path_to_oni_file>" << endl;
  cout << "\t -pcd_folder  \t<path_to_folder_with_pcd_files>" << endl;
  cout << "\t -tracking           \t<bool> \ activate the skeleton tracking" << endl;
  cout << "\t -alpha    \t<float> \tset tracking parameter" << endl;
  cout << "\t -beta    \t<float> \tset tracking parameter" << endl;
  cout << "\t -lzf        \t<path_to_folder_with_pclzf_files>" << endl;
  cout << "\t -lzf_fps    \t<int> \tfps for replaying the lzf-files (default: 10)" << endl;
  cout << "\t -segment_people       \t<path_to_svm_file> \t activates body segmentation" << endl;

}

int
main (int argc,
      char** argv)
{
  // answering for help
  PCL_INFO("People tracking App version 0.2\n");
  if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
    return print_help (), 0;

  // selecting GPU and prining info
  int device = 0;
  pc::parse_argument (argc, argv, "-gpu", device);
  pcl::gpu::setDevice (device);
  pcl::gpu::printShortCudaDeviceInfo (device);

  bool write = 0;
  pc::parse_argument (argc, argv, "-w", write);

  // selecting data source
  boost::shared_ptr<pcl::Grabber> capture;
  string openni_device, oni_file, pcd_file, pcd_folder, lzf_dir;

  try
  {
    pc::parse_argument (argc, argv, "-lzf_fps", lzf_fps);

    if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
    {

      capture.reset (new pcl::OpenNIGrabber (openni_device));
    }
    else if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
    {
      capture.reset (new pcl::ONIGrabber (oni_file, true, true));
    }
    else if (pc::parse_argument (argc, argv, "-lzf", lzf_dir) > 0)
    {

      capture.reset (new pcl::ImageGrabber<PointXYZRGBA> (lzf_dir, lzf_fps, false, true));
      lzf_dir_global = lzf_dir;
      islzf = true;
    }
    else if (pc::parse_argument (argc, argv, "-pcd", pcd_file) > 0)
    {
      pcd_file_global = pcd_file;
      capture.reset (new pcl::PCDGrabber<PointXYZRGBA> (vector<string> (31, pcd_file), 1, true));
    }
    else if (pc::parse_argument (argc, argv, "-pcd_folder", pcd_folder) > 0)
    {
      pcd_dir_global = pcd_folder;
      vector<string> pcd_files = getPcdFilesInDir (pcd_folder);
      capture.reset (new pcl::PCDGrabber<PointXYZRGBA> (pcd_files, 1, true));
    }
    else
    {

      capture.reset (new pcl::OpenNIGrabber ());
      //capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224932.oni", true, true) );

      //vector<string> pcd_files(31, "d:/3/0008.pcd");
      //vector<string> pcd_files(31, "d:/git/pcl/gpu/people/tools/test.pcd");
      //vector<string> pcd_files = getPcdFilesInDir("d:/3/");
      //capture.reset( new pcl::PCDGrabber<PointXYZRGBA>(pcd_files, 30, true) );
    }
  }
  catch (const pcl::PCLException& /*e*/)
  {
    return cout << "Can't open depth source" << endl, -1;
  }

  //selecting tree files
  vector<string> tree_files;
  tree_files.push_back ("Data/forest1/tree_20.txt");
  tree_files.push_back ("Data/forest2/tree_20.txt");
  tree_files.push_back ("Data/forest3/tree_20.txt");
  //tree_files.push_back ("Data/forest4/tree_20.txt");

  pc::parse_argument (argc, argv, "-tree0", tree_files[0]);
  pc::parse_argument (argc, argv, "-tree1", tree_files[1]);
  pc::parse_argument (argc, argv, "-tree2", tree_files[2]);
  //pc::parse_argument (argc, argv, "-tree3", tree_files[3]);

  if (pc::parse_argument (argc, argv, "-segment_people", svm_filename) > 0)
  {
    segment_people = true;
  }

  pc::parse_argument (argc, argv, "-resample", resample);

  int num_trees = (int) tree_files.size ();
  pc::parse_argument (argc, argv, "-numTrees", num_trees);

  tree_files.resize (num_trees);
  if (num_trees == 0 || num_trees > 4)
  {
    PCL_ERROR("[Main] : Invalid number of trees");
    print_help ();
    return -1;
  }

  try
  {
    // loading trees
    typedef pcl::gpu::people::RDFBodyPartsDetector RDFBodyPartsDetector;
    RDFBodyPartsDetector::Ptr rdf (new RDFBodyPartsDetector (tree_files));
    PCL_VERBOSE("[Main] : Loaded files into rdf");

    // Create the app
    PeoplePCDApp app (*capture, write);
    app.people_detector_.rdf_detector_ = rdf;

    float alpha = 0.0;
    float beta = 0.0;
    bool tracking = true;//false;
    if (pc::parse_argument (argc, argv, "-alpha", alpha) > 0)
    {
      app.people_detector_.setAlphaTracking (alpha);
    }
    if (pc::parse_argument (argc, argv, "-beta", beta) > 0)
    {
      app.people_detector_.setBetaTracking (beta);
    }
    if (pc::parse_argument (argc, argv, "-tracking", tracking) > 0)
    {
      app.people_detector_.setActiveTracking (tracking);
    }

    // executing
    app.startMainLoop ();
  }
  catch (const pcl::PCLException& e)
  {
    cout << "PCLException: " << e.detailedMessage () << endl;
    print_help ();
  }
  catch (const std::runtime_error& e)
  {
    cout << e.what () << endl;
    print_help ();
  }
  catch (const std::bad_alloc& /*e*/)
  {
    cout << "Bad alloc" << endl;
    print_help ();
  }
  catch (const std::exception& /*e*/)
  {
    cout << "Exception" << endl;
    print_help ();
  }

  return 0;
}

