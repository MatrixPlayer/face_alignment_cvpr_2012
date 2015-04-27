/** ****************************************************************************
 *  @file    HeadPoseSample.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <HeadPoseSample.hpp>
#include <boost/numeric/conversion/bounds.hpp>

void
HeadPoseSample::show
  ()
{
  cv::Scalar white_color = cv::Scalar(255, 255, 255);
  cv::Scalar black_color = cv::Scalar(0, 0, 0);
  cv::Mat img = m_image->m_feature_channels[0].clone();
  cv::imshow("Gray patch", img(m_patch_bbox));
  cv::rectangle(img, m_face_bbox, white_color);
  if (m_label >= 0)
    cv::rectangle(img, m_patch_bbox, white_color);
  else
    cv::rectangle(img, m_patch_bbox, black_color);
  cv::imshow("Face", img);
  cv::waitKey(0);
};

int
HeadPoseSample::evalTest
  (
  const Split &test
  ) const
{
  return m_image->evalTest(test.feature, m_patch_bbox);
};

bool
HeadPoseSample::eval
  (
  const Split &test
  ) const
{
  return evalTest(test) <= test.threshold;
};

bool
HeadPoseSample::generateSplit
  (
  const std::vector<HeadPoseSample*> &samples,
  boost::mt19937 *rng,
  int patch_size,
  Split &split
  )
{
  int num_feature_channels = samples[0]->m_image->m_feature_channels.size();
  split.feature.generate(patch_size, rng, num_feature_channels);
  split.num_thresholds = 25;
  split.margin = 0;

  return true;
};

double
HeadPoseSample::evalSplit
  (
  const std::vector<HeadPoseSample*> &setA,
  const std::vector<HeadPoseSample*> &setB,
  float split_mode,
  int depth
  )
{
  if (split_mode < 50)
  {
    double ent_a = entropie(setA);
    double ent_b = entropie(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(setA.size() + setB.size());
  }
  else
  {
    int size_a = 0;
    int size_b = 0;
    double ent_a = gain2(setA, &size_a);
    double ent_b = gain2(setB, &size_b);
    return (ent_a * size_a + ent_b * size_b) / static_cast<double>(size_b + size_a);
  }
};

void
HeadPoseSample::makeLeaf
  (
  HeadPoseLeaf &leaf,
  const std::vector<HeadPoseSample*> &set
  )
{
  int size = 0; // number of foreground samples
  std::vector<HeadPoseSample*>::const_iterator it_sample;
  for (it_sample = set.begin(); it_sample < set.end(); ++it_sample)
    if ((*it_sample)->m_is_positive)
      size++;

  leaf.hp_foreground = static_cast<float>(size) / static_cast<float>(set.size());
  leaf.hp_nsamples = set.size();
  leaf.hp_labels.clear();
  leaf.hp_labels.resize(NUM_HEADPOSE_CLASSES, 0);
  if (size > 0)
  {
    for (it_sample = set.begin(); it_sample < set.end(); ++it_sample)
      if ((*it_sample)->m_is_positive)
        leaf.hp_labels[(*it_sample)->m_label]++;

    PRINT("  Histogram: " << cv::Mat(leaf.hp_labels).t());
  }
  else
  {
    PRINT("  Leaf with only background patches: " << set.size());
    for (unsigned int i=0; i < leaf.hp_labels.size(); i++)
      leaf.hp_labels[i] = 0;
  }
};

double
HeadPoseSample::entropie
  (
  const std::vector<HeadPoseSample*> &set
  )
{
  // Count number of foreground patches
  double p = 0;
  std::vector<HeadPoseSample*>::const_iterator it_sample;
  for (it_sample = set.begin(); it_sample < set.end(); ++it_sample)
    if ((*it_sample)->m_is_positive)
      p += 1;

  double n_entropy = 0;
  // Probability to classify as foreground patch (positive)
  double p_pos = p/static_cast<double>(set.size());
  if (p_pos > 0)
    n_entropy += p_pos * log(p_pos);
  // Probability to classify as background patch (negative)
  double p_neg = (set.size()-p)/static_cast<double>(set.size());
  if (p_neg > 0)
    n_entropy += p_neg * log(p_neg);

  return n_entropy;
};

/*double
HeadPoseSample::entropie_pose
  (
  const std::vector<HeadPoseSample*> &set
  )
{
  double n_entropy = 0;
  for (int i=0; i < 5; i++)
  {
    std::vector<HeadPoseSample*>::const_iterator itSample;
    int size = 0;
    int p = 0;
    for (itSample = set.begin(); itSample < set.end(); ++itSample)
    {
      if ((*itSample)->isPos)
      {
        size += 1;
        if ((*itSample)->label == i)
        {
          p += 1;
        }
      }
    }
    double p_pos = float(p) / size;
    if (p_pos > 0)
      n_entropy += p_pos * log(p_pos);
  }
  return n_entropy;
};*/

/*double
HeadPoseSample::gain
  (
  const std::vector<HeadPoseSample*> &set,
  int *num_pos_elements
  )
{
  int size = 0;
  std::vector<HeadPoseSample*>::const_iterator itSample;

  // Fill points for variance calculation
  float mean_pos = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample)
  {
    if ((*itSample)->isPos)
    {
      size++;
      mean_pos += (*itSample)->label;
    }
  }
  mean_pos /= size;
  *num_pos_elements = size;

  if (size > 0)
  {
    float var = 0;
    for (itSample = set.begin(); itSample < set.end(); ++itSample)
    {
      if ((*itSample)->isPos)
      {
        var += (mean_pos - (*itSample)->label) * (mean_pos - (*itSample)->label);
      }
    }
    var /= size;
    return -var;
  }
  else
  {
    return boost::numeric::bounds<double>::lowest();
  }
};*/

double
HeadPoseSample::gain2
  (
  const std::vector<HeadPoseSample*> &set,
  int *num_pos_elements
  )
{
  double n = 0;
  double sum = 0;
  double sq_sum = 0;

  std::vector<HeadPoseSample*>::const_iterator it_sample;
  for (it_sample = set.begin(); it_sample < set.end(); ++it_sample)
  {
    if ((*it_sample)->m_is_positive)
    {
      n++;
      int l = (*it_sample)->m_label; // pose = [0, 1, 2, 3, 4]
      sum += l;
      sq_sum += l * l;
    }
  }
  *num_pos_elements = n;

  double mean = sum / n;
  double variance = (sq_sum / n) - (mean * mean);
  return -variance;
};
