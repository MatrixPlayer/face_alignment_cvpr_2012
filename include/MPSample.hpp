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

  MPSample
    () :
      m_nparts(0) {};

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
    const ImageSample *patch,
    cv::Rect rect
    );

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
  float distToCenter;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & m_is_positive;
    if (m_is_positive)
    {
      ar & m_patch_offset;
      ar & m_part_offsets;
      ar & m_prob;
      ar & m_patch_bbox;
      ar & distToCenter;
    }
  }
};

/** ****************************************************************************
 * @class MPLeaf
 * @brief Multiple parts leaf sample
 ******************************************************************************/
class MPLeaf
{
public:
  MPLeaf
    ()
  {
    depth = -1;
    save_all = false;
  };

  std::vector<float> maxDists;
  std::vector<float> lamda;
  int nSamples; // number of patches reached the leaf
  std::vector<cv::Point_<int> > parts_offset; // vector of the means
  std::vector<float> variance; // variance of the votes
  std::vector<float> pF; // probability of foreground per each point
  cv::Point_<int> patch_offset;
  float forgound; //probability of face
  int depth;
  bool save_all;
  std::vector<cv::Point_<int> > offset_sum;
  std::vector<cv::Point_<int> > offset_sum_sq;
  std::vector<float> sum_pf;
  int sum_pos;
  int sum_all;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & nSamples;
    ar & parts_offset;
    ar & variance;
    ar & pF;
    ar & forgound;
    ar & patch_offset;
    ar & save_all;
  }
};

#endif /* MP_SAMPLE_HPP */
