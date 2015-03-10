/** ****************************************************************************
 *  @file    HeadPoseSample.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef HEAD_POSE_SAMPLE_HPP
#define HEAD_POSE_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ThresholdSplit.hpp>
#include <ImageSample.hpp>
#include <SplitGen.hpp>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <opencv2/highgui/highgui.hpp>

class HeadPoseLeaf;

/** ****************************************************************************
 * @class HeadPoseSample
 * @brief Head pose sample
 ******************************************************************************/
class HeadPoseSample
{
public:
  typedef ThresholdSplit<SimplePatchFeature> Split;
  typedef HeadPoseLeaf Leaf;

  HeadPoseSample
    () {};

  HeadPoseSample
    (
    const ImageSample *image_,
    const cv::Rect roi_,
    cv::Rect rect_,
    int label_
    ) :
      image(image_), roi(roi_), rect(rect_), label(label_)
  {
    // If the label is smaller then 0 then negative example.
    isPos = (label >= 0);
  };

  HeadPoseSample
    (
    const ImageSample *image_,
    cv::Rect rect_
    ) :
      image(image_), rect(rect_), label(-1) {};

  virtual
  ~HeadPoseSample
    () {};

  void
  show
    ();

  // Used from SplitGen "generateMT"
  int
  evalTest
    (
    const Split &test
    ) const;

  bool
  eval
    (
    const Split &test
    ) const;

  // Called from SplitGen "generateMT"
  static bool
  generateSplit
    (
    const std::vector<HeadPoseSample*> &samples,
    boost::mt19937 *rng,
    ForestParam fp,
    Split &split
    );

  // Called from SplitGen "findThreshold"
  static double
  evalSplit
    (
    const std::vector<HeadPoseSample*> &setA,
    const std::vector<HeadPoseSample*> &setB,
    float split_mode,
    int depth
    );

  static void
  makeLeaf
    (
    HeadPoseLeaf &leaf,
    const std::vector<HeadPoseSample*> &set,
    const std::vector<float> &poppClasses,
    int leaf_id = 0
    );

  // Called from "Tree" initialization
  static void
  calcWeightClasses
    (
    std::vector<float> &class_weights,
    const std::vector<HeadPoseSample*> &samples
    );

  const ImageSample *image;
  bool isPos;
  cv::Rect roi;
  cv::Rect rect;
  int label;

private:
  static double
  entropie
    (
    const std::vector<HeadPoseSample*> &set
    );

  /*static double
  entropie_pose
    (
    const std::vector<HeadPoseSample*> &set
    );*/

  /*static double
  gain
    (
    const std::vector<HeadPoseSample*> &set,
    int *num_pos_elements
    );*/

  static double
  gain2
    (
    const std::vector<HeadPoseSample*> &set,
    int *num_pos_elements
    );
};

/** ****************************************************************************
 * @class HeadPoseLeaf
 * @brief Head pose leaf sample
 ******************************************************************************/
class HeadPoseLeaf
{
public:
  HeadPoseLeaf
    ()
  {
    depth = -1;
  };

  int nSamples; //number of patches reached the leaf
  std::vector<int> hist_labels;
  float forgound;
  int depth;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & nSamples;
    ar & forgound;
    ar & depth;
    ar & hist_labels;
  }
};

#endif /* HEAD_POSE_SAMPLE_HPP */
