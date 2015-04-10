/** ****************************************************************************
 *  @file    face_utils.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/01
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_UTILS_HPP
#define FACE_UTILS_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <SplitGen.hpp>
#include <Forest.hpp>
#include <ImageSample.hpp>
#include <HeadPoseSample.hpp>
#include <MPSample.hpp>
#include <FaceForest.hpp>

/** ****************************************************************************
 * @brief Parse annotations file
 ******************************************************************************/
struct FaceAnnotation
{
  std::vector<cv::Point> parts; // number of facial feature points
  std::string url;              // path to original image
  cv::Rect bbox;                // bounding box
  int pose;                     // head pose
};

/** ****************************************************************************
 * @brief Patch vote for each facial feature point
 ******************************************************************************/
struct Vote
{
  Vote() :
    weight(0.0f), check(false) {};

  cv::Point pos; // estimated feature point position
  float weight;  // patch distance
  bool check;    // meaningful vote
};

// Load image from annotations file
cv::Mat
loadImage
  (
  std::string path,
  std::string name
  );

// Scale image and annotations for training
cv::Mat
scale
  (
  cv::Mat img,
  int face_size,
  FaceAnnotation &annotation
  );

// Enlarge image and annotations by 20% for training
cv::Mat
enlarge
  (
  cv::Mat img,
  FaceAnnotation &annotation
  );

// Loads and parse a configuration file
bool
loadConfigFile
  (
  std::string path,
  ForestParam &param
  );

// Loads and parse annotations
bool
loadAnnotations
  (
  std::string path,
  std::vector<FaceAnnotation> &annotations
  );

// Called from "estimateHeadPose"
void
getHeadPoseVotesMT
  (
  const ImageSample &sample,
  const Forest<HeadPoseSample> &forest,
  cv::Rect face_bbox,
  float *headpose,
  float *variance,
  HeadPoseEstimatorOption options = HeadPoseEstimatorOption()
  );

// Called from "estimateFacialFeatures"
void
getFacialFeaturesVotesMT
  (
  const ImageSample &sample,
  const Forest<MPSample> &forest,
  cv::Rect face_bbox,
  std::vector< std::vector<Vote> > &votes,
  MultiPartEstimatorOption options = MultiPartEstimatorOption()
  );

// Computes the area under curve called from "analizeFace"
float
areaUnderCurve
  (
  float x1,
  float x2,
  double mean,
  double std
  );

// Returns the intersection called from "detectFace" and "extractFace"
cv::Rect
intersect
  (
  const cv::Rect r1,
  const cv::Rect r2
  );

// Plots all the votes for each part from "estimateFacialFeatures"
void
plotVotes
  (
  const cv::Mat &img_gray,
  std::vector< std::vector<Vote> > &votes,
  std::vector<cv::Point> ffd_cordinates
  );

#endif /* FACE_UTILS_HPP */
