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
    PRINT("FC: " << featureChannel);
    PRINT("Rect A " << rectA.x << ", " << rectA.y << ", " << rectA.width << " " << rectA.height);
    PRINT("Rect B " << rectB.x << ", " << rectB.y << ", " << rectB.width << " " << rectB.height);
  };

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
      boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_feat(*rng, dist_feat);
      featureChannel = rand_feat();
    }
    else
      featureChannel = 0;

    int size = static_cast<int>(patch_size * max_sub_patch_ratio);

    boost::uniform_int<> dist_size(1, (size - 1) * 0.75);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_size(*rng, dist_size);
    rectA.width = rand_size();
    rectA.height = rand_size();
    rectB.width = rand_size();
    rectB.height = rand_size();

    boost::uniform_int<> dist_x(0, size - rectA.width - 1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x(*rng, dist_x);
    rectA.x = rand_x();

    boost::uniform_int<> dist_y(0, size - rectA.height - 1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y(*rng, dist_y);
    rectA.y = rand_y();

    boost::uniform_int<> dist_x_b(0, size - rectB.width - 1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x_b(*rng, dist_x_b);
    rectB.x = rand_x_b();

    boost::uniform_int<> dist_y_b(0, size - rectB.height - 1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y_b(*rng, dist_y_b);
    rectB.y = rand_y_b();

    CV_Assert(rectA.x >= 0 and rectB.x>=0 and rectA.y >= 0 and rectB.y>=0);
    CV_Assert(rectA.x+rectA.width < patch_size and rectA.y+rectA.height < patch_size);
    CV_Assert(rectB.x+rectB.width < patch_size and rectB.y+rectB.height < patch_size);
  };

  int featureChannel;
  cv::Rect_<int> rectA;
  cv::Rect_<int> rectB;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & featureChannel;
    ar & rectA;
    ar & rectB;
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
