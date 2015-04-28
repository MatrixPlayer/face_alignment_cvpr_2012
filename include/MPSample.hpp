/** ****************************************************************************
 *  @file    MPSample.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/09
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef MP_SAMPLE_HPP
#define MP_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ThresholdSplit.hpp>
#include <ImageSample.hpp>
#include <SplitGen.hpp>
#include <opencv_serialization.hpp>
#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <opencv2/highgui/highgui.hpp>

class MPLeaf;

/** ****************************************************************************
 * @class MPSample
 * @brief Face feature detection sample
 ******************************************************************************/
class MPSample
{
public:
  typedef ThresholdSplit<SimplePatchFeature> Split;
  typedef MPLeaf Leaf;

  // Training
  MPSample
    (
    const ImageSample *sample,
    cv::Rect patch_bbox,
    const std::vector<cv::Point> annotation_parts,
    float face_size,
    bool label,
    float lamda = 0.125
    );

  // Testing
  MPSample
    (
    const ImageSample *sample,
    cv::Rect patch_bbox
    ) :
      m_image(sample), m_patch_bbox(patch_bbox) {};

  virtual
  ~MPSample
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
    const std::vector<MPSample*> &samples,
    boost::mt19937 *rng,
    int patch_size,
    Split &split
    );

  // Called from SplitGen "findThreshold"
  static double
  evalSplit
    (
    const std::vector<MPSample*> &setA,
    const std::vector<MPSample*> &setB,
    float split_mode,
    int depth
    );

  // Called from TreeNode "createLeaf"
  static void
  makeLeaf
    (
    MPLeaf &leaf,
    const std::vector<MPSample*> &samples
    );

  cv::Rect
  getPatch
    ()
  {
    return m_patch_bbox;
  };

private:
  inline static double
  entropie
    (
    const std::vector<MPSample*> &set
    );

  inline static double
  entropie_parts
    (
    const std::vector<MPSample*> &set
    );

  const ImageSample *m_image;
  bool m_is_positive;
  cv::Rect m_patch_bbox;
  cv::Point_<int> m_patch_offset;
  int m_nparts;
  std::vector< cv::Point_<int> > m_part_offsets;
  cv::Mat m_prob;
};

/** ****************************************************************************
 * @class MPLeaf
 * @brief Multiple parts leaf sample
 ******************************************************************************/
class MPLeaf
{
public:
  MPLeaf
    () {};

  int mp_samples;                                 // number of patches reached the leaf
  std::vector< cv::Point_<int> > mp_parts_offset; // vector of facial points
  std::vector<float> mp_parts_variance;           // variance of the votes
  std::vector<float> mp_prob_foreground;          // probability of foreground per each point
  float mp_foreground;                            // probability of face
  cv::Point_<int> mp_patch_offset;
  bool mp_save_all;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & mp_samples;
    ar & mp_parts_offset;
    ar & mp_parts_variance;
    ar & mp_prob_foreground;
    ar & mp_foreground;
    ar & mp_patch_offset;
    ar & mp_save_all;
  }
};

#endif /* MP_SAMPLE_HPP */
