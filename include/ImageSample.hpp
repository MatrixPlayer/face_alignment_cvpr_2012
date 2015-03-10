/** ****************************************************************************
 *  @file    ImageSample.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef IMAGE_SAMPLE_HPP
#define IMAGE_SAMPLE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <FeatureChannelFactory.hpp>
#include <opencv_serialization.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/serialization/access.hpp>
#include <opencv2/highgui/highgui.hpp>

struct SimplePatchFeature
{
  void
  print
    ()
  {
    PRINT("feature_channel: " << feature_channel);
    PRINT("rect1: " << rect1);
    PRINT("rect2: " << rect2);
  };

  void
  generate
    (
    int patch_size,
    boost::mt19937 *rng,
    int num_feature_channels = 0,
    float max_subpatch_ratio = 1.0
    )
  {
    // Selected appearance channel randomly
    if (num_feature_channels > 1)
    {
      boost::uniform_int<> dist_feat(0, num_feature_channels - 1);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      feature_channel = rand_feat();
    }
    else
      feature_channel = 0;

    // R1 and R2 describe two rectangles within the patch boundaries
    int subpatch_size = static_cast<int>(patch_size * max_subpatch_ratio); // 31 x 1
    boost::uniform_int<> dist_size(1, (subpatch_size-1) * 0.75);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);
    rect1.width  = rand_size();
    rect1.height = rand_size();
    rect2.width  = rand_size();
    rect2.height = rand_size();

    boost::uniform_int<> dist_x_a(0, subpatch_size-rect1.width-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x_a(*rng, dist_x_a);
    rect1.x = rand_x_a();
    boost::uniform_int<> dist_y_a(0, subpatch_size-rect1.height-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y_a(*rng, dist_y_a);
    rect1.y = rand_y_a();
    boost::uniform_int<> dist_x_b(0, subpatch_size-rect2.width-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x_b(*rng, dist_x_b);
    rect2.x = rand_x_b();
    boost::uniform_int<> dist_y_b(0, subpatch_size-rect2.height-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y_b(*rng, dist_y_b);
    rect2.y = rand_y_b();

    CV_Assert(rect1.x >= 0 && rect2.x >= 0 && rect1.y >= 0 && rect2.y >= 0);
    CV_Assert(rect1.x+rect1.width < patch_size && rect1.y+rect1.height < patch_size);
    CV_Assert(rect2.x+rect2.width < patch_size && rect2.y+rect2.height < patch_size);
  };

  int feature_channel;
  cv::Rect_<int> rect1;
  cv::Rect_<int> rect2;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & feature_channel;
    ar & rect1;
    ar & rect2;
  }
};

struct SimplePixelFeature
{
  void
  print
    ()
  {
    PRINT("FC: " << featureChannel);
    PRINT("Point A " << pointA.x << ", " << pointA.y);
    PRINT("Point B " << pointB.x << ", " << pointB.y);
  }

  void
  generate
    (
    int patch_size,
    boost::mt19937 *rng,
    int num_feature_channels = 0,
    float max_sub_patch_ratio = 1.0
    )
  {
    if (num_feature_channels > 1)
    {
      boost::uniform_int<> dist_feat(0, num_feature_channels - 1);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      featureChannel = rand_feat();
    }
    else
      featureChannel = 0;

    boost::uniform_int<> dist_size(1, patch_size);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);

    pointA.x = rand_size();
    pointA.y = rand_size();
    pointB.x = rand_size();
    pointB.y = rand_size();
  };

  int featureChannel;
  cv::Point_<int> pointA;
  cv::Point_<int> pointB;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & featureChannel;
    ar & pointA;
    ar & pointB;
  }
};

/** ****************************************************************************
 * @class ImageSample
 * @brief Patch sample from an image
 ******************************************************************************/
class ImageSample
{
public:
  ImageSample
    () {};

  ImageSample
    (
    const cv::Mat img,
    std::vector<int> features,
    bool use_integral = false
    );

  ImageSample
    (
    const cv::Mat img,
    std::vector<int> features,
    FeatureChannelFactory &fcf,
    bool use_integral = false
    );

  virtual
  ~ImageSample
    ();

  // Used from HeadPoseSample and MPSample "evalTest"
  int
  evalTest
    (
    const SimplePatchFeature &test,
    const cv::Rect rect
    ) const;

  int
  evalTest
    (
    const SimplePixelFeature &test,
    const cv::Rect rect
    ) const;

  void
  extractFeatureChannels
    (
    const cv::Mat &img,
    std::vector<cv::Mat> &feature_channels,
    std::vector<int> features,
    bool use_integral,
    FeatureChannelFactory &fcf
    ) const;

  void
  getSubPatches
    (
    cv::Rect rect,
    std::vector<cv::Mat> &tmpPatches
    );

  int
  width
    () const
  {
    return m_feature_channels[0].cols;
  };

  int
  height
    () const
  {
    return m_feature_channels[0].rows;
  };

  void
  show
    () const
  {
    cv::imshow("Image Sample", m_feature_channels[0]);
    cv::waitKey(0);
  };

  std::vector<cv::Mat> m_feature_channels;

private:
  bool m_use_integral;
};

#endif /* IMAGE_SAMPLE_HPP */
