/*
 * Software License Agreement (BSD License)
 *
 * Point Cloud Library (PCL) - www.pointclouds.org
 * Copyright (c) 2013-, Open Perception, Inc.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following
 * disclaimer in the documentation and/or other materials provided
 * with the distribution.
 * * Neither the name of the copyright holder(s) nor the names of its
 * contributors may be used to endorse or promote products derived
 * from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * activity_recognition_app.cpp
 * Created on: Aug 10, 2014
 * Author: Alina Roitberg
 *
 * Activity classification framework, which applies K-Nearest-Neighbours algorithm on the sequences of skeleton data.
 *
 */

#include <iostream>
#include <string>
#include <fstream>
#include <cstring>
#include <pcl/common/eigen.h>
#include <math.h>  
#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/gpu/people/label_common.h>
using std::ifstream;
using std::cout;
using std::endl;
using namespace pcl::gpu;
using namespace pcl::gpu::people;
using namespace pcl;
using namespace std;
namespace pc = pcl::console;

#define PI 3.14159265
#define MIN3(a, b, c) ((a) < (b) ? ((a) < (c) ? (a) : (c)) : ((b) < (c) ? (b) : (c)))

//constants
const int MAX_CHARS_PER_LINE = 1024;
const int MAX_TOKENS_PER_LINE = 29;
const char* const DELIMITER = " ";
const int NUM_PARTS_LABELED = 27;
const int NR_SEGMENTS = 8;
const int MAX_FRAMES = 700;
const int MAX_RECORDINGS = 500;
const int MAX_DIMENSIONS = 1000;
const int NR_TOP = 10;
const int NR_ACTIVITIES = 14;
const int KNN = 1;  //maximum is 10!!!

bool mirror_data = 1;
bool use_variances = 0;
bool normalize_all_data = 1;
bool normalize_height = 1;
bool use_position_change = 0;
double height_mean = 0.0;
double height_var = 0.0;
double* vars = (double *) malloc (sizeof(double[MAX_DIMENSIONS]));
double* means = (double *) malloc (sizeof(double[MAX_DIMENSIONS]));
int nr_read_files = 0;
int dimensions_true = 500;

char *ACTIVITIES[NR_ACTIVITIES] = { "check_watch", "kick", "punch", "stand", "wave", "cross_arms", "pick_up", "scratch_head", "throw_from_bottom_up",
    "turn_around", "get_up", "point", "sit_down", "throw_over_head" };

typedef Eigen::Vector3f joints_frame_eigen[NUM_PARTS_LABELED];

//Struct for a data sequence (for both training and test data)
struct data_sequence_raw
{
    char annotation[20];
    joints_frame_eigen data_eigen[MAX_FRAMES];
    joints_frame_eigen data_segmented_eigen[NR_SEGMENTS];
    joints_frame_eigen data_segmented_eigen_var[NR_SEGMENTS];
    double feature_vector_mirrored[MAX_DIMENSIONS];
    double feature_vector[MAX_DIMENSIONS];
    char* uid;
    int nrFrames;
    double height;
};

//Training data
struct data_sequence_raw *all_data = (struct data_sequence_raw *) malloc (sizeof(struct data_sequence_raw) * MAX_RECORDINGS);
;

/**
 *Estimates the array index of the corresponding action name
 *(Actions defined in ACTIVITIES)
 *@return ID of the action
 */
int
findIdOfAction (char* action)
{

  for (int i = 0; i < NR_ACTIVITIES; i++)
  {
    if (strcmp (action, ACTIVITIES[i]) == 0)
      return i;
  }
  return -1;
}

/*
 *Calculates the mean value
 */
double
getMean (double * data,
         int numElements)
{
  double sum = 0.0;
  int numElementsFinal = numElements;
  for (int i = 0; i < numElements; i++)
  {
    //getting rid of invalid values
    if (data[i] < 10000 && data[i] > -10000)
    {
      sum += data[i];
    }
    else
      numElementsFinal--;
  }
  return sum / numElementsFinal;
}

/*
 *Calculates the variance
 */
double
getVariance (double * data,
             int numElements)
{

  double mean = getMean (data, numElements);
  double temp = 0;
  int numElementsFinal = numElements;
  for (int i = 0; i < numElements; i++)
  {
    //getting rid of invalid values
    if (data[i] < 100000 && data[i] > -100000)
    {
      temp += (mean - data[i]) * (mean - data[i]);
    }
    else
      numElementsFinal--;
  }

  return temp / numElementsFinal;
}

/**
 * Not used yet!
 * Calculates levenshtein between two integer arrays
 * (String comparison on the joint rankings)
 * Implemented initially for the "sequence of most informative joints" approach
 */
int
levenshtein (double *s1,
             double *s2,
             int size)
{
  unsigned int x, y;

  unsigned int matrix[size + 1][size + 1];
  matrix[0][0] = 0;
  for (x = 1; x <= size; x++)
    matrix[x][0] = matrix[x - 1][0] + 1;
  for (y = 1; y <= size; y++)
    matrix[0][y] = matrix[0][y - 1] + 1;
  for (x = 1; x <= size; x++)
    for (y = 1; y <= size; y++)
      matrix[x][y] = MIN3(matrix[x - 1][y] + 1, matrix[x][y - 1] + 1, matrix[x - 1][y - 1] + (s1[y - 1] == s2[x - 1] ? 0 : 1));

  return (matrix[size][size]);
}

/*
 *
 * Not used yet!
 * Implemented initially for the "sequence of most informative joints" approach
 * Estimated the NR_TOP features (joints) with highest variance
 * @return array with indicies of top features (sorted)
 */
void
rankJoints (double variance_vector[NUM_PARTS_LABELED],
            int target[NR_TOP])
{

  double values[NR_TOP] = { -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0, -100000.0 };

  for (int i = 0; i < NUM_PARTS_LABELED; i++)
  {
    double curr_val = variance_vector[i];
    for (int k = NR_TOP - 1; k >= 0; k--)
    {

      if (curr_val < values[k])
      {
        break;
      }
      else
      {

        if (k < NR_TOP - 1)
        {  //if not the last one, we swap values
          values[k + 1] = values[k];
          values[k] = curr_val;
          target[k + 1] = target[k];
          target[k] = i;
        }
        else
        {
          values[k] = curr_val;

          target[k] = i;
        }
      }

    }
  }

}

/*
 * Calculates the Euclidean distance
 */

double
euclidean_distance (double *p1,
                    double *p2,
                    int dimension)
{

  double result = 0.0;
  for (int i = 0; i < dimension; i++)
  {
    double temp = p1[i] - p2[i];

    temp *= temp;
    if (temp > 0)  //getting rid of invalid values
      result += temp;

  }

  result = sqrt (result);

  return result;

}

/*
 * Calculates the angle between two vectors
 */
double
angleBetween (Eigen::Vector3f j1,
              Eigen::Vector3f j2)
{
  double result = 0.0;

  double cosval = j1.dot (j2) / (j1.norm () * j2.norm ());
  result = acos (cosval) * 180.0 / PI;

  if (result > 360 || result < -360)
    return 0.0;
  return result;
}

/*
 * Parses a file containing the skeleton data (one!) sample
 * Stores the data in  struct data_sequence_raw *data
 */
int
parseFile (std::string filename,
           struct data_sequence_raw *data)
{

  ifstream fin;
  fin.open (filename.c_str ());  // open a file
  if (!fin.good ())
    return -1;  // exit if file not found
  int currFrame = 0;
  int currJoint = 0;

  // read each line of the file
  while (!fin.eof ())
  {

    char buf[MAX_CHARS_PER_LINE];
    fin.getline (buf, MAX_CHARS_PER_LINE);

    int n = 0;  // a for-loop index

    char* token[MAX_TOKENS_PER_LINE] = { };  // initialize to 0

    // parse the line
    token[0] = strtok (buf, DELIMITER);  // first token
    if (token[0])  // zero if line is blank
    {
      for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
      {
        token[n] = strtok (0, DELIMITER);  // subsequent tokens

        if (!token[n])
        {

          break;
        }
      }
    }

    Eigen::Vector3f pos;

    for (int i = 0; i < n; i++)
    {  // n = #of tokens
      if (n == 3)  //A line with joint information
      {
        double f = std::atof (token[i]);

        pos[i] = f;

        data->data_eigen[currFrame][currJoint][i] = f * 100.0;

        if (i == 2)
          currJoint++;
      }
      if (n == 2 && currJoint > 0 && i == 1)  //end of the frame
      {
        currFrame++;
        currJoint = 0;

      }

    }

  }
  data->nrFrames = currFrame;

  return 1;
}

/*
 * Parses a file containing the training data (multiple recordings)
 * Stores the data in the array of struct data_sequence_raw
 */
int
parseFileTraining (std::string filename,
                   struct data_sequence_raw data_all[MAX_RECORDINGS])
{

  ifstream fin;
  fin.open (filename.c_str ());
  if (!fin.good ())
    return -1;
  int currFrame = 0;
  int currJoint = 0;
  char* currUid = "-1";
  char* currAction = "check_watch";

  int curr_seq = 0;
  while (!fin.eof ())
  {

    char buf[MAX_CHARS_PER_LINE];
    fin.getline (buf, MAX_CHARS_PER_LINE);

    int n = 0;

    char* token[MAX_TOKENS_PER_LINE] = { };

    token[0] = strtok (buf, DELIMITER);
    if (token[0])  //blank line
    {
      for (n = 1; n < MAX_TOKENS_PER_LINE; n++)
      {
        token[n] = strtok (0, DELIMITER);

        if (!token[n])
        {

          break;  // no more tokens
        }
      }
    }

    Eigen::Vector3f pos;

    if (n == 4)  //new recording
    {
      if (currFrame > NR_SEGMENTS)
      {

        strcpy (data_all[curr_seq].annotation, token[3]);
        data_all[curr_seq].uid = token[1];
        data_all[curr_seq].nrFrames = currFrame;
        curr_seq++;

      }

      currJoint = 0;

      currFrame = 0;
    }
    for (int i = 0; i < n; i++)
    {
      if (n == 3)  //line with joint positions
      {
        double f = std::atof (token[i]);

        pos[i] = f;

        //cout << "pos[i]" <<pos[i] << endl;
        data_all[curr_seq].data_eigen[currFrame][currJoint][i] = f * 100.0;

        if (i == 2)
          currJoint++;
      }
      if (n == 2 && currJoint > 0 && i == 1)  //new frame
      {
        currFrame++;
        currJoint = 0;

      }

    }

  }
  return curr_seq;

}

/*
 *Segments the skeleton data from joints_array[MAX_FRAMES]
 *Output:
 *sequence_eigen[NR_SEGMENTS]: segmented skeleton position data (mean values over the whole segment are used)
 *sequence_var[NR_SEGMENTS]: variances (over each segment)
 *@return: the height of the person in the recording
 */
double
processOne (joints_frame_eigen joints_array[MAX_FRAMES],
            joints_frame_eigen sequence_eigen[NR_SEGMENTS],
            joints_frame_eigen sequence_var[NR_SEGMENTS],
            int nrFrames)
{

  int segmentLength = nrFrames / NR_SEGMENTS;

  for (int j = 0; j < NUM_PARTS_LABELED; j++)
  {  //joints

    double x_vals = 0.0;
    double y_vals = 0.0;
    double z_vals = 0.0;
    double x_vals_vel = 0.0;
    double y_vals_vel = 0.0;
    double z_vals_vel = 0.0;
    double x_count = 0.0;
    double y_count = 0.0;
    double z_count = 0.0;
    int segment_nr = 0;
    double x_arr[nrFrames];
    double y_arr[nrFrames];
    double z_arr[nrFrames];

    double height = 1.0;

    for (int i = 1; i < nrFrames; i++)
    {  //frames

      if (joints_array[i][j][0] != -1.0)
      {  //if not an invalid value

        //positions
        x_arr[(int) x_count] = joints_array[i][j][0];
        y_arr[(int) y_count] = joints_array[i][j][1];
        z_arr[(int) z_count] = joints_array[i][j][2];

        x_vals += joints_array[i][j][0];
        y_vals += joints_array[i][j][1];
        z_vals += joints_array[i][j][2];

        x_count++;
        y_count++;
        z_count++;
      }
      if (i % segmentLength == 0 && i > 0)
      {  //if it's the end of a segment

        sequence_eigen[segment_nr][j][0] = x_vals / x_count;
        sequence_eigen[segment_nr][j][1] = y_vals / y_count;
        sequence_eigen[segment_nr][j][2] = z_vals / z_count;
        /*
         sequence[segment_nr][j][0]=joints_array[nrFrames/2][j][0];
         sequence[segment_nr][j][1]=joints_array[nrFrames/2][j][1];
         sequence[segment_nr][j][2]=joints_array[nrFrames/2][j][2];
         
         */

        //sequence[segment_nr][j][0]=x_vals_vel/x_count;
        //sequence[segment_nr][j][1]=y_vals_vel/y_count;
        //sequence[segment_nr][j][2]=z_vals_vel/z_count;
        sequence_var[segment_nr][j][0] = getVariance (x_arr, (int) x_count);
        sequence_var[segment_nr][j][1] = getVariance (y_arr, (int) y_count);
        sequence_var[segment_nr][j][2] = getVariance (z_arr, (int) z_count);
        //cout << "7" << nrFrames << endl;
        segment_nr++;
        x_count = 0.0;
        y_count = 0.0;
        z_count = 0.0;
        x_vals = 0.0;
        y_vals = 0.0;
        z_vals = 0.0;
      }
    }

  }
  //cout << "8" << nrFrames << endl;
  double height = abs (
      ( (sequence_eigen[0][FaceLT] + sequence_eigen[0][FaceRT]) / 2.0)[1] - ( (sequence_eigen[0][Lfoot] + sequence_eigen[0][Rfoot]) / 2.0)[1]);

  double height_last = abs (
      ( (sequence_eigen[NR_SEGMENTS - 1][FaceLT] + sequence_eigen[NR_SEGMENTS - 1][FaceRT]) / 2.0)[1]
          - ( (sequence_eigen[NR_SEGMENTS - 1][Lfoot] + sequence_eigen[NR_SEGMENTS - 1][Rfoot]) / 2.0)[1]);

  if (height_last > height)
    height = height_last;

  return height;

}

/**
 *Computes the feature vector used as input to the machine learning framework
 *This is the part with feature selection
 *Input
 * *target: contains the segmented skeleton data
 *Output
 *result: computed feature vector
 *result_mirrored: mirrored feature vector (CAUTION, only accurate with currently used joints(Hands,Elbows,Feet and the Neck)))
 *If you want to use other joints, don't mirror the data
 */
int
computeFeatureVector (struct data_sequence_raw *target,
                      double result[MAX_DIMENSIONS],
                      double result_mirrored[MAX_DIMENSIONS])
{

  int count = 0;

  double height_ratio = 1.0;
  if (normalize_height)
  {
//estimating the height of the person
    height_ratio = abs (
        ( (target->data_segmented_eigen[0][FaceLT] + target->data_segmented_eigen[0][FaceRT]) / 2.0)[1]
            - ( (target->data_segmented_eigen[0][Lfoot] + target->data_segmented_eigen[0][Rfoot]) / 2.0)[1]);

    double height_last = abs (
        ( (target->data_segmented_eigen[NR_SEGMENTS - 1][FaceLT] + target->data_segmented_eigen[NR_SEGMENTS - 1][FaceRT]) / 2.0)[1]
            - ( (target->data_segmented_eigen[NR_SEGMENTS - 1][Lfoot] + target->data_segmented_eigen[NR_SEGMENTS - 1][Rfoot]) / 2.0)[1]);
    if (height_last > height_ratio)
      height_ratio = height_last;

    height_ratio /= height_mean;
  }

  //going through all segments
  for (int i = 0; i < NR_SEGMENTS; i++)
  {
    //Getting the angles

    //elbows
    Eigen::Vector3f j1 = target->data_segmented_eigen[i][Lhand] - target->data_segmented_eigen[i][Lelbow];
    Eigen::Vector3f j2 = target->data_segmented_eigen[i][Lelbow] - target->data_segmented_eigen[i][Lshoulder];
    double angleElbowLeft = angleBetween (j1, j2);

    j1 = target->data_segmented_eigen[i][Rhand] - target->data_segmented_eigen[i][Relbow];
    j2 = target->data_segmented_eigen[i][Relbow] - target->data_segmented_eigen[i][Rshoulder];
    double angleElbowRight = angleBetween (j1, j2);

    //arms
    j1 = target->data_segmented_eigen[i][Rshoulder] - target->data_segmented_eigen[i][Relbow];
    j2 = target->data_segmented_eigen[i][Rshoulder] - target->data_segmented_eigen[i][Neck];
    double angleArmRight = angleBetween (j1, j2);

    j1 = target->data_segmented_eigen[i][Lshoulder] - target->data_segmented_eigen[i][Lelbow];
    j2 = target->data_segmented_eigen[i][Lshoulder] - target->data_segmented_eigen[i][Neck];
    double angleArmLeft = angleBetween (j1, j2);

    //head orientation (3D)
    j1 = (target->data_segmented_eigen[i][FaceLT] + target->data_segmented_eigen[i][FaceRT]) / 2.0;
    j2 = (target->data_segmented_eigen[i][FaceLB] + target->data_segmented_eigen[i][FaceRB]) / 2.0;
    j1 = j1 - j2;  //Face vector in j1

    j2 = Eigen::Vector3f (1, 0, 0);

    double angleHeadX = angleBetween (j1, j2);
    j2 = Eigen::Vector3f (0, 1, 0);
    double angleHeadY = angleBetween (j1, j2);
    j2 = Eigen::Vector3f (0, 0, 1);
    double angleHeadZ = angleBetween (j1, j2);

    //torso

    j1 = (target->data_segmented_eigen[i][Lchest] + target->data_segmented_eigen[i][Rchest]) / 2.0
        - (target->data_segmented_eigen[i][Lhips] + target->data_segmented_eigen[i][Rhips]) / 2.0;
    j2 = (target->data_segmented_eigen[i][Lhips] - target->data_segmented_eigen[i][Rhips]);

    double angle_torso = angleBetween (j1, j2);
    //legs
    j1 = target->data_segmented_eigen[i][Lleg] - target->data_segmented_eigen[i][Lknee];
    j2 = (target->data_segmented_eigen[i][Lhips] - target->data_segmented_eigen[i][Rhips]);

    double angle_Lleg = angleBetween (j1, j2);

    j1 = target->data_segmented_eigen[i][Rleg] - target->data_segmented_eigen[i][Rknee];
    j2 = target->data_segmented_eigen[i][Lhips] - target->data_segmented_eigen[i][Rhips];

    double angle_Rleg = angleBetween (j1, j2);

    //knees

    j1 = target->data_segmented_eigen[i][Lfoot] - target->data_segmented_eigen[i][Lknee];
    j2 = (target->data_segmented_eigen[i][Lhips] - target->data_segmented_eigen[i][Lknee]);

    double angle_Lknee = angleBetween (j1, j2);

    j1 = target->data_segmented_eigen[i][Rfoot] - target->data_segmented_eigen[i][Rknee];
    j2 = (target->data_segmented_eigen[i][Rhips] - target->data_segmented_eigen[i][Rknee]);

    double angle_Rknee = angleBetween (j1, j2);

    //Adding the selected angles to the feature vector

    result[count++] = angleElbowLeft;
    result[count++] = angleElbowRight;
    result_mirrored[count - 1] = result[count - 2];
    result_mirrored[count - 2] = result[count - 1];

    //result[count++]=angleArmLeft;
    //result[count++]=angleArmRight;
    //result_mirrored[count-1]=result[count-2];
    //result_mirrored[count-2]=result[count-1];

    //result[count++]=angleHeadX;
    //result_mirrored[count-1]=result[count-1];

    //result[count++]=angleHeadY;
    //result_mirrored[count-1]=result[count-1];

    //result[count++]=angleHeadZ;
    //result_mirrored[count-1]=result[count-1];

    //result[count++]=angle_torso;
    //result_mirrored[count-1]=result[count-1];

    //result[count++]=angle_Lleg;
    //result[count++]=angle_Rleg;
    //result_mirrored[count-1]=result[count-2];
    //result_mirrored[count-2]=result[count-1];

    result[count++] = angle_Lknee;
    result[count++] = angle_Rknee;
    result_mirrored[count - 1] = result[count - 2];
    result_mirrored[count - 2] = result[count - 1];

    //Normalizing the height:
    if (normalize_height)
      for (int j = 0; j < NUM_PARTS_LABELED; j++)
        target->data_segmented_eigen[i][j][1] = target->data_segmented_eigen[i][j][1] / (height_ratio);

    //we use the Neck position in reference to the feet position
    Eigen::Vector3f foot_mean = (target->data_segmented_eigen[i][Lfoot] + target->data_segmented_eigen[i][Rfoot]) / 2.0;
    //Eigen::Vector3f foot_mean(0.0,0.0,0.0);
    for (int j = 0; j < NUM_PARTS_LABELED; j++)    //going through all joints
    {

      //for variances

      if (use_variances && (j == Rhand || j == Lhand || j == Rfoot || j == Lfoot))
      {

        result[count++] = target->data_segmented_eigen_var[i][j][0];
        result_mirrored[count - 1] = result[count - 1];
        result[count++] = target->data_segmented_eigen_var[i][j][1];
        result_mirrored[count - 1] = result[count - 1];
        result[count++] = target->data_segmented_eigen_var[i][j][2];
        result_mirrored[count - 1] = result[count - 1];
      }

      if (use_position_change)
      {
        if (i > 0)
          target->data_segmented_eigen[i][j] = target->data_segmented_eigen[i][j] - target->data_segmented_eigen[i - 1][j];
        else
          target->data_segmented_eigen[i][j] = target->data_segmented_eigen[i][j] - target->data_segmented_eigen[i][j];
      }

      //If we mirror the data
      if (mirror_data)
      {
        //Do not add other joints here! otherwise mirroring does not work!
        //Deactivate mirroring and add the joints below (see else-block)
        if ( (j == Relbow || j == Rhand || j == Rfoot || j == Neck))
        {
          if (j != Neck)
          {
            result[count++] = target->data_segmented_eigen[i][j][0] - target->data_segmented_eigen[i][Neck][0];
            result_mirrored[count - 1] = - (target->data_segmented_eigen[i][j + 4][0] - target->data_segmented_eigen[i][Neck][0]);
            result[count++] = target->data_segmented_eigen[i][j][1] - target->data_segmented_eigen[i][Neck][1];
            result_mirrored[count - 1] = target->data_segmented_eigen[i][j + 4][1] - target->data_segmented_eigen[i][Neck][1];
            result[count++] = target->data_segmented_eigen[i][j][2] - target->data_segmented_eigen[i][Neck][2];
            result_mirrored[count - 1] = target->data_segmented_eigen[i][j + 4][2] - target->data_segmented_eigen[i][Neck][2];
          }
          else
          {

            result[count++] = target->data_segmented_eigen[i][j][0] - foot_mean[0];
            result_mirrored[count - 1] = result[count - 1];
            result[count++] = target->data_segmented_eigen[i][j][1] - foot_mean[1];
            result_mirrored[count - 1] = result[count - 1];
            result[count++] = target->data_segmented_eigen[i][j][2] - foot_mean[2];
            result_mirrored[count - 1] = result[count - 1];
          }

        }

        if ( (j == Lelbow || j == Lhand || j == Lfoot))
        {

          result[count++] = target->data_segmented_eigen[i][j][0] - target->data_segmented_eigen[i][Neck][0];
          result_mirrored[count - 1] = - (target->data_segmented_eigen[i][j - 4][0] - target->data_segmented_eigen[i][Neck][0]);
          result[count++] = target->data_segmented_eigen[i][j][1] - target->data_segmented_eigen[i][Neck][1];
          result_mirrored[count - 1] = target->data_segmented_eigen[i][j - 4][1] - target->data_segmented_eigen[i][Neck][1];
          result[count++] = target->data_segmented_eigen[i][j][2] - target->data_segmented_eigen[i][Neck][2];
          result_mirrored[count - 1] = target->data_segmented_eigen[i][j + 4][2] - target->data_segmented_eigen[i][Neck][2];

        }
      }
      else
      {  //if !mirror_data
        //Other joints can be added here
        if ( (j == Relbow || j == Rhand || j == Rfoot || j == Neck))
        {
          if (j != Neck)
          {
            result[count++] = target->data_segmented_eigen[i][j][0] - target->data_segmented_eigen[i][Neck][0];

            result[count++] = target->data_segmented_eigen[i][j][1] - target->data_segmented_eigen[i][Neck][1];

            result[count++] = target->data_segmented_eigen[i][j][2] - target->data_segmented_eigen[i][Neck][2];

          }
          else
          {

            result[count++] = target->data_segmented_eigen[i][j][0] - foot_mean[0];

            result[count++] = target->data_segmented_eigen[i][j][1] - foot_mean[1];

            result[count++] = target->data_segmented_eigen[i][j][2] - foot_mean[2];

          }

        }

      }

    }

  }

  return count;
}

/*
 * Calculates the mean and variance of each joint position and angle
 * Strores means and variances in vars and means (global variables)
 * Normalizes the training dataset
 */
void
preprocessForKnn (struct data_sequence_raw training_data[MAX_RECORDINGS],
                  int nrOfSamples)
{

  int size = nrOfSamples * NR_SEGMENTS;
  if (mirror_data)
    size *= 2;
  int dim_in_segment = dimensions_true / NR_SEGMENTS;

  double** res = (double **) malloc (sizeof(double*) * dim_in_segment);

  for (int j = 0; j < dim_in_segment; j++)
  {
    res[j] = (double *) malloc (sizeof(double) * size);
  }
  int counter = 0;

  for (int dim = 0; dim < dim_in_segment; dim++)
  {

    for (int sample = 0; sample < nrOfSamples; sample++)
    {

      for (int segment = 0; segment < NR_SEGMENTS; segment++)
      {
        res[dim][sample * NR_SEGMENTS + segment] = training_data[sample].feature_vector[segment * dim_in_segment + dim];
      }

    }

  }
  if (mirror_data)
  {
    for (int dim = 0; dim < dim_in_segment; dim++)
    {

      for (int sample = 0; sample < nrOfSamples; sample++)
      {

        for (int segment = 0; segment < NR_SEGMENTS; segment++)
        {
          res[dim][NR_SEGMENTS * nrOfSamples - 1 + sample * NR_SEGMENTS + segment] = training_data[sample].feature_vector_mirrored[segment * dim_in_segment
              + dim];
        }

      }

    }
  }

  for (int i = 0; i < dim_in_segment; i++)
  {

    vars[i] = getVariance (res[i], size);
    means[i] = getMean (res[i], size);
  }

  for (int dim = 0; dim < dim_in_segment; dim++)
  {

    for (int sample = 0; sample < nrOfSamples; sample++)
    {

      for (int segment = 0; segment < NR_SEGMENTS; segment++)
      {

        if (false && segment == 0)
          training_data[sample].feature_vector[segment * dim_in_segment + dim] = 0.0;
        else
        {
          training_data[sample].feature_vector[segment * dim_in_segment + dim] = (training_data[sample].feature_vector[segment * dim_in_segment + dim]
              - means[dim]) / sqrt (vars[dim]);
          training_data[sample].feature_vector_mirrored[segment * dim_in_segment + dim] = (training_data[sample].feature_vector_mirrored[segment
              * dim_in_segment + dim] - means[dim]) / sqrt (vars[dim]);
        }
      }

    }

  }

}

/**
 * Normalizes the test sample
 * The mean and variance values of the training set are used
 */
void
preprocessForKnnOne (struct data_sequence_raw* data)
{

  int dim_in_segment = dimensions_true / NR_SEGMENTS;

  for (int dim = 0; dim < dim_in_segment; dim++)
  {

    for (int segment = 0; segment < NR_SEGMENTS; segment++)
    {

      data->feature_vector[segment * dim_in_segment + dim] = (data->feature_vector[segment * dim_in_segment + dim] - means[dim]) / sqrt (vars[dim]);
      data->feature_vector_mirrored[segment * dim_in_segment + dim] = (data->feature_vector_mirrored[segment * dim_in_segment + dim] - means[dim])
          / sqrt (vars[dim]);

    }

  }

}

/**
 * Applies KNN classification
 * Input
 * data: training data
 * point: feature vector that needs to be classified
 * nrOfSamples: size of the training data set
 * kn: number of neighbors (Caution: MAX 10!!!)
 * Output: name of the estimated activity
 */
char*
computeNearestNeighbour (struct data_sequence_raw *data,
                         double point[MAX_DIMENSIONS],
                         int nrOfSamples,
                         int kn)
{

  char* result = "";
  int res_id = 0;

  double values[NR_TOP] = { 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0 };
  int target[NR_TOP] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int buckets[NR_ACTIVITIES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  for (int i = 0; i < nrOfSamples; i++)
  {

    double curr_val = euclidean_distance (point, & (all_data[i].feature_vector[0]), MAX_DIMENSIONS);

    for (int k = NR_TOP - 1; k >= 0; k--)
    {

      if (curr_val > values[k])
      {
        break;
      }
      else
      {

        if (k < NR_TOP - 1)
        {  //if not the last one, we swap values
          values[k + 1] = values[k];
          values[k] = curr_val;
          target[k + 1] = target[k];
          target[k] = i;
        }
        else
        {
          values[k] = curr_val;

          target[k] = i;
        }
      }

    }

    if (mirror_data)
    {
      curr_val = euclidean_distance (point, & (all_data[i].feature_vector_mirrored[0]), MAX_DIMENSIONS);

      for (int k = NR_TOP - 1; k >= 0; k--)
      {

        if (curr_val > values[k])
        {
          break;
        }
        else
        {

          if (k < NR_TOP - 1)
          {  //if not the last one, we swap values
            values[k + 1] = values[k];
            values[k] = curr_val;
            target[k + 1] = target[k];
            target[k] = i;
          }
          else
          {
            values[k] = curr_val;

            target[k] = i;
          }
        }

      }
    }

  }

  for (int i = 0; i < kn; i++)
  {
    buckets[findIdOfAction (all_data[target[i]].annotation)]++;
  }
  int winner = 0;
  int winner_val = 0;
  for (int i = 0; i < NR_ACTIVITIES; i++)
  {
    if (buckets[i] > winner_val)
    {
      winner = i;
      winner_val = buckets[i];
    }

  }

  result = ACTIVITIES[winner];
  return result;

}

/**
 * Applies KNN classification for cross validation
 * Sample to classify: sample with testID
 * Training data:  all samples from training_data EXCEPT for the one with testID
 * kn: number of neighbors (Caution: MAX 10!!!)
 *
 * Output: name of the estimated activity
 */
char*
computeNearestNeighbourTest (struct data_sequence_raw training_data[MAX_RECORDINGS],
                             int testId,
                             int nrOfSamples,
                             int kn)
{

  char* result = "";
  int res_id = 0;

  double values[NR_TOP] = { 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0 };
  int target[NR_TOP] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  int buckets[NR_ACTIVITIES] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

  for (int i = 0; i < nrOfSamples; i++)
  {
    if (i != testId)
    {

      double curr_val = euclidean_distance (& (training_data[testId].feature_vector[0]), & (training_data[i].feature_vector[0]), MAX_DIMENSIONS);

      for (int k = NR_TOP - 1; k >= 0; k--)
      {

        if (curr_val > values[k])
        {
          break;
        }
        else
        {

          if (k < NR_TOP - 1)
          {  //if not the last one, we swap values
            values[k + 1] = values[k];
            values[k] = curr_val;
            target[k + 1] = target[k];
            target[k] = i;
          }
          else
          {
            values[k] = curr_val;

            target[k] = i;
          }
        }

      }

      if (mirror_data)
      {
        curr_val = euclidean_distance (& (training_data[testId].feature_vector[0]), & (training_data[i].feature_vector_mirrored[0]), MAX_DIMENSIONS);

        for (int k = NR_TOP - 1; k >= 0; k--)
        {

          if (curr_val > values[k])
          {
            break;
          }
          else
          {

            if (k < NR_TOP - 1)
            {  //if not the last one, we swap values
              values[k + 1] = values[k];
              values[k] = curr_val;
              target[k + 1] = target[k];
              target[k] = i;
            }
            else
            {
              values[k] = curr_val;

              target[k] = i;
            }
          }

        }
      }

    }
  }

  for (int i = 0; i < kn; i++)
  {
    buckets[findIdOfAction (training_data[target[i]].annotation)]++;
  }
  int winner = 0;
  int winner_val = 0;
  for (int i = 0; i < NR_ACTIVITIES; i++)
  {
    if (buckets[i] > winner_val)
    {
      winner = i;
      winner_val = buckets[i];
    }

  }

  result = ACTIVITIES[winner];
  return result;

}

/**
 *Parses and classifies skeleton file
 *Input:
 *filename: path to the skeleton file
 *Output:
 *name of the estimated activity
 */
char*
classifySequence (std::string filename)
{

  char* action = "";

  struct data_sequence_raw *data = (struct data_sequence_raw *) malloc (sizeof(struct data_sequence_raw));

  if (parseFile (filename, data) != -1)
  {

    processOne (data->data_eigen, data->data_segmented_eigen, data->data_segmented_eigen_var, data->nrFrames);

    computeFeatureVector (data, & (data->feature_vector[0]), & (data->feature_vector_mirrored[0]));

    if (normalize_all_data)
      preprocessForKnnOne (data);

    action = computeNearestNeighbour (all_data, & (data->feature_vector[0]), nr_read_files, KNN);
  }
  std::cout << "Estimated activity:  " << action << endl;
  return action;
}

/**
 *Parses and processes the training data file
 *Input:
 *filename_all: path to the file containing all the training data
 *The training data is stored in the global array all_data (struct data_sequence_raw *)
 */
void
loadTrainingData (std::string filename_all)
{

  nr_read_files = parseFileTraining (filename_all, all_data);

  cout << "Read " << nr_read_files << " recordings" << endl;
  joints_frame_eigen sequence[NR_SEGMENTS];
  height_mean = 0.0;

  //segmenting all recordings

  for (int i = 0; i < nr_read_files; i++)
  {

    double h = processOne (all_data[i].data_eigen, all_data[i].data_segmented_eigen, all_data[i].data_segmented_eigen_var, all_data[i].nrFrames);
    all_data[i].height = h;
    height_mean += h;
  }
  height_mean /= nr_read_files;
  //computing feature vectors

  for (int i = 0; i < nr_read_files; i++)
  {
    dimensions_true = computeFeatureVector (&all_data[i], & (all_data[i].feature_vector[0]), & (all_data[i].feature_vector_mirrored[0]));

  }
  if (normalize_all_data)
    preprocessForKnn (all_data, nr_read_files);

}

/*
 * Preforms leave one ot cross validation on the training data set and outputs the confusion matrix
 */
void
testWithCrossValidation ()
{

  cout << "---Preforming Leave-One-Out Cross Validation on the data set---" << endl;
  cout << endl << "---Confusion matrix: ---" << endl << endl;

  double count_corr_all = 0;
  int count_samples_all = 0;
  double count_corr = 0;
  int count_samples = 0;

  int confusion_matrix[NR_ACTIVITIES][NR_ACTIVITIES];
  for (int i = 0; i < NR_ACTIVITIES; i++)
  {
    for (int j = 0; j < NR_ACTIVITIES; j++)
    {
      confusion_matrix[i][j] = 0;
    }
  }

  //findIdOfAction
  for (int a = 0; a < NR_ACTIVITIES; a++)
  {
    int id1 = findIdOfAction (ACTIVITIES[a]);

    for (int i = 0; i < nr_read_files; i++)
    {
      if (strcmp (all_data[i].annotation, ACTIVITIES[a]) == 0)
      {

        count_samples++;
        int id2 = findIdOfAction (computeNearestNeighbourTest (all_data, i, nr_read_files, KNN));
        confusion_matrix[id1][id2]++;

      }

    }

    cout << endl;
    std::string s (ACTIVITIES[a]);
    cout << ACTIVITIES[a];

    for (int i = 0; i < 25 - s.size (); i++)
      cout << " ";

    for (int i = 0; i < NR_ACTIVITIES; i++)
    {

      cout << confusion_matrix[a][i] << "  ";

    }
    cout << "correct: " << confusion_matrix[a][a] / (double) count_samples << "  ";

    if (confusion_matrix[a][a] / (double) count_samples > -1)
      count_corr_all += confusion_matrix[a][a] / (double) count_samples;

    count_samples = 0;

  }

  cout << endl << "Average (over all actions): " << count_corr_all / (float) NR_ACTIVITIES << endl;

}

int
print_help ()
{
  cout << "*******************************************************" << std::endl;
  cout << "Activity detection from skeleton information options:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --training_data    <path_to_traning_data_file>" << std::endl;
  cout << "   --cross_validation    <bool value , perform leave-one-out cross validation on the training data>" << std::endl;

  cout
      << "Example: pcl_activity_recognition_app -training_data ../gpu/people/data/skeleton_final_data.txt -test_file ../gpu/people/data/skeleton_sample_kick.txt"
      << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}

int
main (int argc,
      char** argv)
{

  std::string filename_all = "../gpu/people/data/skeleton_final_data.txt";
  std::string filename_test = "../gpu/people/data/skeleton_sample_kick.txt";

  bool cross_validation = 0;
  bool test_file = 0;

  if (pc::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h"))
    return print_help ();

  pc::parse_argument (argc, argv, "-training_data", filename_all);
  test_file = pc::parse_argument (argc, argv, "-test_file", filename_test);
  pc::parse_argument (argc, argv, "-cross_validation", cross_validation);

  loadTrainingData (filename_all);
  if (cross_validation)
    testWithCrossValidation ();

  if (test_file > 0)
    classifySequence (filename_test);

  return 0;
}

