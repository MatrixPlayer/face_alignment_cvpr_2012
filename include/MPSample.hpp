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
#include <trace.hpp>
#include <ThresholdSplit.hpp>
#include <ImageSample.hpp>
#include <SplitGen.hpp>
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
      num_parts(0) {};

  MPSample
    (
    const ImageSample *patch,
    cv::Rect rect,
    const cv::Rect roi,
    const std::vector<cv::Point> parts,
    float size,
    bool label,
    float lamda = 0.125
    );

  MPSample
    (
    const ImageSample *patch,
    cv::Rect rect,
    int n_points,
    float size
    );

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
    ()
  {
    cv::imshow("MPSample X", image->m_feature_channels[0](rect));
    cv::Mat face = image->m_feature_channels[0].clone();
    cv::rectangle(face, rect, cv::Scalar(255, 255, 255, 0));
    cv::rectangle(face, roi, cv::Scalar(255, 255, 255, 0));
    if (isPos)
    {
      int patch_size = (rect.height) / 2.0;
      for (int i = 0; i < (int) part_offsets.size(); i++)
      {
        int x = rect.x + patch_size + part_offsets[i].x;
        int y = rect.y + patch_size + part_offsets[i].y;
        cv::circle(face, cv::Point_<int>(x, y), 3, cv::Scalar(255, 255, 255, 0));
        PRINT(i << " " << dist.at<float>(0, i));
      }
      int x = rect.x + patch_size + patch_offset.x;
      int y = rect.y + patch_size + patch_offset.y;
      cv::circle(face, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
    }
    cv::imshow("MPSample Y", face);
    cv::waitKey(0);
  };

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

  static bool
  generateSplit
    (
    const std::vector<MPSample*> &data,
    boost::mt19937 *rng,
    ForestParam fp,
    Split &split,
    float split_mode,
    int depth
    );

  static double
  evalSplit
    (
    const std::vector<MPSample*> &setA,
    const std::vector<MPSample*>& setB,
    const std::vector<float> &poppClasses,
    float splitMode,
    int depth
    );

  static void
  optimize
    (
    boost::mt19937 *rng,
    const std::vector<MPSample*> &set,
    Split &split,
    float split_mode,
    int depth
    );

  inline static double
  entropie
    (
    const std::vector<MPSample*> &set
    );

  inline static double
  entropie_pose
    (
    const std::vector<MPSample*> &set
    );

  inline static double
  entropie_parts
    (
    const std::vector<MPSample*> &set
    );

  inline static double
  infoGain
    (
    const std::vector<MPSample*> &set
    );

  static void
  makeLeaf
    (
    MPLeaf &leaf,
    const std::vector<MPSample*> &set,
    const std::vector<float> &poppClasses,
    int leaf_id = 0
    );

  static void
  calcWeightClasses
    (
    std::vector<float> &poppClasses,
    const std::vector<MPSample*> &set
    );

  static double
  entropieFaceOrNot
    (
    const std::vector<MPSample*> &set
    );

  static double
  eval_oob
    (
    const std::vector<MPSample*> &data, Split &test
    )
  {
    return 0;
  };

  float distToCenter;
  const ImageSample *image;
  std::vector< cv::Point_<int> > part_offsets;
  cv::Rect rect;
  cv::Rect roi;
  cv::Mat dist;
  float size;
  int num_parts;
  cv::Point_<int> patch_offset;
  bool isPos;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & isPos;
    if (isPos)
    {
      ar & patch_offset;
      ar & part_offsets;
      ar & dist;
      ar & rect;
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
