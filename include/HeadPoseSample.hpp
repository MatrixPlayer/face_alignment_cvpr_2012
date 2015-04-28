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

  // Training
  HeadPoseSample
    (
    const ImageSample *sample,
    const cv::Rect face_bbox,
    cv::Rect patch_bbox,
    int label
    ) :
      m_image(sample), m_face_bbox(face_bbox), m_patch_bbox(patch_bbox), m_label(label)
  {
    // If the label is smaller than 0 then negative example
    m_is_positive = (label >= 0);
  };

  // Testing
  HeadPoseSample
    (
    const ImageSample *sample,
    cv::Rect patch_bbox
    ) :
      m_image(sample), m_patch_bbox(patch_bbox), m_label(-1) {};

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

  // Called from "evaluateMT" on TreeNode
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
    int patch_size,
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

  // Called from TreeNode "createLeaf"
  static void
  makeLeaf
    (
    HeadPoseLeaf &leaf,
    const std::vector<HeadPoseSample*> &set
    );

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

  const ImageSample *m_image;
  bool m_is_positive;
  cv::Rect m_face_bbox;
  cv::Rect m_patch_bbox;
  int m_label;
};

/** ****************************************************************************
 * @class HeadPoseLeaf
 * @brief Head pose leaf sample
 ******************************************************************************/
class HeadPoseLeaf
{
public:
  HeadPoseLeaf
    () {};

  int hp_nsamples;            // number of patches reached the leaf
  float hp_foreground;        // positive patches percentage
  std::vector<int> hp_labels; // number of patches for each class

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & hp_nsamples;
    ar & hp_foreground;
    ar & hp_labels;
  }
};

#endif /* HEAD_POSE_SAMPLE_HPP */
