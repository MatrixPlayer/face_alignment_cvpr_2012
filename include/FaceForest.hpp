/** ****************************************************************************
 *  @file    FaceForest.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/06
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_FOREST_HPP
#define FACE_FOREST_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Viewer.hpp>
#include <Forest.hpp>
#include <SplitGen.hpp>
#include <ImageSample.hpp>
#include <HeadPoseSample.hpp>
#include <MPSample.hpp>
#include <opencv2/objdetect/objdetect.hpp>

struct FaceDetectionOption
{
  FaceDetectionOption
    () :
    min_feature_size(30), min_neighbors(1), search_scale_factor(1.3f) {};

  // Options for face detection
  int min_feature_size;
  int min_neighbors;
  float search_scale_factor;
  std::string path_face_cascade;
};

struct HeadPoseEstimatorOption
{
  HeadPoseEstimatorOption
    () :
    num_head_pose_labels(5), step_size(4), min_forground_probability(0.5) {};

  int num_head_pose_labels;
  int step_size;
  float min_forground_probability;
};

struct MultiPartEstimatorOption
{
  MultiPartEstimatorOption
    () :
    num_parts(10), step_size(3), min_samples(2), min_forground(0.5),
    min_pf(0.25), max_variance(25) {};

  int num_parts;
  int step_size;
  int min_samples;
  float min_forground;
  float min_pf;
  float max_variance;
};

struct FaceForestOptions
{
  ForestParam hp_forest_param;
  ForestParam mp_forest_param;
  FaceDetectionOption fd_option;
  HeadPoseEstimatorOption hp_option;
  MultiPartEstimatorOption mp_option;
  std::vector<std::string> mp_tree_paths;
};

struct Face
{
  float headpose;
  cv::Rect bbox;
  std::vector<cv::Point> ffd_cordinates;
};

/** ****************************************************************************
 * @class FaceForest
 * @brief Estimate head-pose and detect facial feature points
 ******************************************************************************/
class FaceForest
{
public:
  FaceForest
    () :
    m_trees(0), num_trees(0), is_inizialized(false) {};

  FaceForest
    (
    FaceForestOptions option
    );

  virtual
  ~FaceForest
    () {};

  static void
  estimateHeadPose
    (
    const ImageSample &sample,
    const cv::Rect &face_bbox,
    const Forest<HeadPoseSample> &forest,
    HeadPoseEstimatorOption options,
    float *headpose,
    float *variance
    );

  static void
  estimateFacialFeatures
    (
    const ImageSample &sample,
    const cv::Rect face_bbox,
    const Forest<MPSample> &forest,
    MultiPartEstimatorOption options,
    std::vector<cv::Point> &ffd_cordinates
    );

  static void
  showResults
    (
    const cv::Mat img,
    std::vector<Face> &faces,
    upm::Viewer &viewer
    );

  static void
  detectFace
    (
    const cv::Mat &img,
    cv::CascadeClassifier &face_cascade,
    FaceDetectionOption fd_option,
    std::vector<cv::Rect> &faces_bboxes
    );

  void
  analyzeImage
    (
    cv::Mat img,
    std::vector<Face> &faces
    );

  void
  analyzeFace
    (
    const cv::Mat img,
    cv::Rect face_bbox,
    Face &face,
    bool normalize = true
    );

private:
  void
  getPathsToTrees
    (
    std::string path,
    std::vector<std::string> &urls
    );

  void
  loadingAllTrees
    (
     std::vector<std::string> urls
     );

  FaceForestOptions m_ff_options;
  cv::CascadeClassifier m_face_cascade;
  Forest<HeadPoseSample> m_hp_forest;
  Forest<MPSample> m_mp_forest;
  std::vector< std::vector<Tree<MPSample>*> > m_trees;
  int num_trees;
  FeatureChannelFactory fcf;
  bool is_inizialized;
};

#endif /* FACE_FOREST_HPP */
