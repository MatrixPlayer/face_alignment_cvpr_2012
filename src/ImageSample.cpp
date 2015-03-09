/** ****************************************************************************
 *  @file    ImageSample.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <ImageSample.hpp>

ImageSample::ImageSample
  (
  const cv::Mat img,
  std::vector<int> features,
  bool use_integral
  ) :
  m_use_integral(use_integral)
{
  FeatureChannelFactory fcf = FeatureChannelFactory();
  extractFeatureChannels(img, m_feature_channels, features, m_use_integral, fcf);
};

ImageSample::ImageSample
  (
  const cv::Mat img,
  std::vector<int> features,
  FeatureChannelFactory &fcf,
  bool use_integral
  ) :
  m_use_integral(use_integral)
{
  extractFeatureChannels(img, m_feature_channels, features, m_use_integral, fcf);
};

ImageSample::~ImageSample
  ()
{
  for (unsigned int i=0; i < m_feature_channels.size(); i++)
    m_feature_channels[i].release();
  m_feature_channels.clear();
};

int
ImageSample::evalTest
  (
  const SimplePatchFeature &test,
  const cv::Rect rect
  ) const
{
  int p1 = 0;
  int p2 = 0;
  const cv::Mat ptC = m_feature_channels[test.featureChannel];
  if (!m_use_integral)
  {
    cv::Mat tmp = ptC(cv::Rect(test.rectA.x + rect.x, test.rectA.y + rect.y, test.rectA.width, test.rectA.height));
    p1 = (cv::sum(tmp))[0] / static_cast<float>(test.rectA.width * test.rectA.height);

    cv::Mat tmp2 = ptC(cv::Rect(test.rectB.x + rect.x, test.rectB.y + rect.y, test.rectB.width, test.rectB.height));
    p2 = (cv::sum(tmp2))[0] / static_cast<float>(test.rectB.width * test.rectB.height);
  }
  else
  {
    int a = ptC.at<float>(rect.y + test.rectA.y, rect.x + test.rectA.x);
    int b = ptC.at<float>(rect.y + test.rectA.y, rect.x + test.rectA.x + test.rectA.width);
    int c = ptC.at<float>(rect.y + test.rectA.y + test.rectA.height, rect.x + test.rectA.x);
    int d = ptC.at<float>(rect.y + test.rectA.y + test.rectA.height, rect.x + test.rectA.x + test.rectA.width);
    p1 = (d - b - c + a) / static_cast<float>(test.rectA.width * test.rectA.height);

    a = ptC.at<float>(rect.y + test.rectB.y, rect.x + test.rectB.x);
    b = ptC.at<float>(rect.y + test.rectB.y, rect.x + test.rectB.x + test.rectB.width);
    c = ptC.at<float>(rect.y + test.rectB.y + test.rectB.height, rect.x + test.rectB.x);
    d = ptC.at<float>(rect.y + test.rectB.y + test.rectB.height, rect.x + test.rectB.x + test.rectB.width);
    p2 = (d - b - c + a) / static_cast<float>(test.rectB.width * test.rectB.height);
  }
  return p1 - p2;
};

int
ImageSample::evalTest
  (
  const SimplePixelFeature &test,
  const cv::Rect rect
  ) const
{
  return m_feature_channels[test.featureChannel].at<unsigned char>(rect.y + test.pointA.y, rect.x + test.pointA.x)
      - m_feature_channels[test.featureChannel].at<unsigned char>(rect.y + test.pointB.y, rect.x + test.pointB.x);
};

void
ImageSample::extractFeatureChannels
  (
  const cv::Mat &img,
  std::vector<cv::Mat> &feature_channels,
  std::vector<int> features,
  bool use_integral,
  FeatureChannelFactory &fcf
  ) const
{
  sort(features.begin(), features.end());
  for (unsigned int i=0; i < features.size(); i++)
    fcf.extractChannel(features[i], use_integral, img, feature_channels);
};

void
ImageSample::getSubPatches
  (
  cv::Rect rect,
  std::vector<cv::Mat> &tmpPatches
  )
{
  for (unsigned int i=0; i < m_feature_channels.size(); i++)
    tmpPatches.push_back(m_feature_channels[i](rect));
};
