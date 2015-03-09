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

struct FaceAnnotation
{
    std::vector<cv::Point> parts; // number of facial feature points
    std::string url; // url to original image
    cv::Rect bbox; // bounding box
    int pose; // head pose
};

struct Vote
{
	Vote() :
	    check(false) {};

	cv::Point2i pos;
	float weight;
	bool check;
};

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
  std::vector<HeadPoseLeaf*> &leafs,
  int step_size = 5
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

// extract a region of interest
/*void
extract_face
  (
  const cv::Mat& img,
  FaceAnnotation& ann,
  cv::Mat& face ,
  int offset_x,
  int offset_y
  );

// rescale image
void
rescale_img
  (
  const cv::Mat& src,
  cv::Mat& dest,
  float scale,
  FaceAnnotation& ann
  );

// displays the annotations
void
plot_face
  (
  const cv::Mat &img,
  FaceAnnotation ann
  );

// Plots all the votes for each part from "estimateFacialFeatures"
void
plot_ffd_votes
  (
  const cv::Mat& face,
  std::vector<std::vector<Vote> >& votes,
  std::vector<cv::Point> results,
  std::vector<cv::Point> gt
  );*/

#endif /* FACE_UTILS_HPP */
