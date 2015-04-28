/** ****************************************************************************
 *  @file    MPSample.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/09
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <MPSample.hpp>
#include <Constants.hpp>
#include <boost/numeric/conversion/bounds.hpp>

MPSample::MPSample
  (
  const ImageSample *sample,
  cv::Rect patch_bbox,
  const std::vector<cv::Point> annotation_parts,
  float face_size,
  bool label,
  float lamda
  ) :
  m_image(sample), m_patch_bbox(patch_bbox)
{
  cv::Point center_patch(m_patch_bbox.x+(m_patch_bbox.width/2), m_patch_bbox.y+(m_patch_bbox.height/2));
  m_nparts = static_cast<int>(annotation_parts.size());
  m_prob = cv::Mat(1, m_nparts, CV_32FC1, cv::Scalar::all(0.0));
  m_part_offsets.resize(m_nparts);
  m_is_positive = false;
  for (int i=0; i < m_nparts; i++)
  {
    m_part_offsets[i] = annotation_parts[i] - center_patch;
    float d = cv::norm(cv::Point_<float>(m_part_offsets[i].x/face_size, m_part_offsets[i].y/face_size));
    // Probability goes to zero for patches that are far away from the feature points
    m_prob.at<float>(0,i) = expf(-d/lamda);
    if (m_prob.at<float>(0,i) > PATCH_CLOSE_TO_FEATURE)
      m_is_positive = true;
  }
  cv::Size bbox = m_image->m_feature_channels[0].size();
  cv::Point center_bbox(bbox.width/2, bbox.height/2);
  m_patch_offset = center_bbox - center_patch;
};

void
MPSample::show
  ()
{
  cv::Scalar white_color = cv::Scalar(255, 255, 255);
  cv::Scalar black_color = cv::Scalar(0, 0, 0);
  cv::Mat img = m_image->m_feature_channels[0].clone();
  cv::imshow("Gray patch", img(m_patch_bbox));
  cv::rectangle(img, m_patch_bbox, white_color);
  if (m_is_positive)
  {
    cv::Point center_patch(m_patch_bbox.x+(m_patch_bbox.width/2), m_patch_bbox.y+(m_patch_bbox.height/2));
    // Annotations points
    for (unsigned i=0; i < m_part_offsets.size(); i++)
    {
      cv::Point pt = center_patch + m_part_offsets[i];
      cv::circle(img, pt, 3, white_color);
    }
    cv::Point center = center_patch + m_patch_offset;
    cv::circle(img, center, 3, black_color);
  }
  cv::imshow("Face", img);
  cv::waitKey(0);
};

int
MPSample::evalTest
  (
  const Split &test
  ) const
{
  return m_image->evalTest(test.feature, m_patch_bbox);
};

bool
MPSample::eval
  (
  const Split &test
  ) const
{
  return evalTest(test) <= test.threshold;
};

bool
MPSample::generateSplit
  (
  const std::vector<MPSample*> &samples,
  boost::mt19937 *rng,
  int patch_size,
  Split &split
  )
{
  int num_feat_channels = samples[0]->m_image->m_feature_channels.size();
  split.feature.generate(patch_size, rng, num_feat_channels);
  split.num_thresholds = 25;
  split.margin = 0;

  return true;
};

double
MPSample::evalSplit
  (
  const std::vector<MPSample*> &setA,
  const std::vector<MPSample*> &setB,
  float split_mode,
  int depth
  )
{
  int mode = int(split_mode) / 50;
  if (split_mode < 50 or depth < 2)
    mode = 0;
  else
    mode = 1;

  mode = 1;
  int size = setA.size() + setB.size();
  if (mode == 0)
  {
    double ent_a = entropie(setA);
    double ent_b = entropie(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(size);
  }
  else
  {
    double ent_a = entropie_parts(setA);
    double ent_b = entropie_parts(setB);
    return (ent_a * setA.size() + ent_b * setB.size()) / static_cast<double>(size);
  }
};

void
MPSample::makeLeaf
  (
  MPLeaf &leaf,
  const std::vector<MPSample*> &samples
  )
{
  int num_parts;
  if (samples.size() > 0)
  {
    num_parts = samples[0]->m_part_offsets.size();
  }
  else
  {
    leaf.mp_foreground = 0;
    num_parts = 0;
    PRINT("something is wrong");
  }

  int nElements = samples.size();

  leaf.mp_parts_offset.clear();
  leaf.mp_parts_offset.resize(num_parts);
  leaf.mp_parts_variance.resize(num_parts);
  leaf.mp_prob_foreground.resize(num_parts);
  leaf.mp_samples = nElements;

  for (int j = 0; j < num_parts; j++)
  {
    leaf.mp_parts_offset[j] = cv::Point(0, 0);
    leaf.mp_parts_variance[j] = boost::numeric::bounds<float>::highest();
    leaf.mp_prob_foreground[j] = 0;
  }
  //leaf.mp_patch_offset = cv::Point(0, 0);
  leaf.mp_foreground = 0;

  // Count number of patches close to the facial points
  std::vector<MPSample*>::const_iterator it_sample;
  int size = 0;
  for (it_sample = samples.begin(); it_sample < samples.end(); ++it_sample)
    if ((*it_sample)->m_is_positive)
      size++;

  if (size > 0)
  {
    for (int j = 0; j < num_parts; j++)
    {
      cv::Point_<int> mean(0.0, 0.0);
      float sumDist = 0;

      for (it_sample = samples.begin(); it_sample < samples.end(); ++it_sample)
      {
        if ((*it_sample)->m_is_positive)
        {
          sumDist += (*it_sample)->m_prob.at<float>(0, j);
          mean += (*it_sample)->m_part_offsets[j];
        }
      }
      mean.x /= static_cast<int>(size);
      mean.y /= static_cast<int>(size);

      leaf.mp_prob_foreground[j] = static_cast<float>(sumDist) / size;
      leaf.mp_parts_offset[j] = mean;

      double var = 0.0;
      for (it_sample = samples.begin(); it_sample < samples.end(); ++it_sample)
      {
        if ((*it_sample)->m_is_positive)
        {
          int x = (*it_sample)->m_part_offsets[j].x;
          int y = (*it_sample)->m_part_offsets[j].y;
          float dist = sqrt((x - mean.x) * (x - mean.x) + (y - mean.y) * (y - mean.y));
          var += dist;
        }
      }
      var /= size;
      leaf.mp_parts_variance[j] = var;

      //std::cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << var<<std::endl;
//            {
//              cv::Point_<int> mean(0.0,0.0);
//              cv::Point_<int> mean_sq(0.0,0.0);
//
//              for ( itSample = set.begin(); itSample < set.end(); ++itSample ){
//                  if( (*itSample)->isPos){
//                      mean += (*itSample)->part_offsets[j];
//                      mean_sq.x += (*itSample)->part_offsets[j].x * (*itSample)->part_offsets[j].x;
//                      mean_sq.y += (*itSample)->part_offsets[j].y * (*itSample)->part_offsets[j].y;
//                  }
//              }
//
//              mean.x /= size;
//              mean.y /= size;
//
//              var = mean_sq.x / size - mean.x * mean.x+
//                    mean_sq.y / size - mean.y * mean.y;
//              cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << sqrt(var) <<endl;
//
//            }

    }
    /*leaf.mp_patch_offset = cv::Point(0, 0);
    for (it_sample = samples.begin(); it_sample < samples.end(); ++it_sample)
    {
      if ((*it_sample)->m_is_positive)
      {
        leaf.mp_patch_offset += (*it_sample)->m_patch_offset;
      }
    }
    leaf.mp_patch_offset.x /= static_cast<int>(size);
    leaf.mp_patch_offset.y /= static_cast<int>(size);*/

    leaf.mp_foreground = size / static_cast<float>(samples.size());
  }
};

double
MPSample::entropie
  (
  const std::vector<MPSample*> &set
  )
{
  double n_entropy = 0;
  std::vector<MPSample*>::const_iterator itSample;
  int p = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample)
    if ((*itSample)->m_is_positive)
      p += 1;

  double p_pos = float(p) / set.size();
  if (p_pos > 0)
    n_entropy += p_pos * log(p_pos);

  double p_neg = float(set.size() - p) / set.size();
  if (p_neg > 0)
    n_entropy += p_neg * log(p_neg);

  return n_entropy;
};

double
MPSample::entropie_parts
  (
  const std::vector<MPSample*> &set
  )
{
  double n_entropy = 0;
  double num_parts = set[0]->m_nparts;
  cv::Mat sum = set[0]->m_prob.clone();
  sum.setTo(cv::Scalar::all(0.0));

  std::vector<MPSample*>::const_iterator itSample;
  for (itSample = set.begin(); itSample < set.end(); itSample++)
    add(sum, (*itSample)->m_prob, sum);

  sum /= static_cast<float>(set.size());
  float p;
  for (int i = 0; i < num_parts; i++) {
    p = sum.at<float>(0, i);
    if (p > 0)
      n_entropy += p * log(p);
  }
  return n_entropy;
};
