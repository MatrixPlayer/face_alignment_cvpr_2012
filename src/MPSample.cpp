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
    if (m_prob.at<float>(0,i) > Constants::PATCH_CLOSE_TO_FEATURE)
      m_is_positive = true;
  }
  cv::Size bbox = m_image->m_feature_channels[0].size();
  cv::Point center_bbox(bbox.width/2, bbox.height/2);
  m_patch_offset = center_bbox - center_patch;

  // Compute distance to face center
  distToCenter = 0;
};

MPSample::MPSample
  (
  const ImageSample *sample,
  cv::Rect patch_bbox
  ) :
  m_image(sample), m_patch_bbox(patch_bbox) {};

void
MPSample::show
  ()
{
  cv::Scalar white_color = cv::Scalar(255, 255, 255);
  cv::Scalar black_color = cv::Scalar(0, 0, 0);
  cv::Mat img = m_image->m_feature_channels[0].clone();
  cv::imshow("Patch", img(m_patch_bbox));
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

double
MPSample::evalSplit
  (
  const std::vector<MPSample*> &setA,
  const std::vector<MPSample*> &setB,
  float splitMode,
  int depth
  )
{
  int mode = int(splitMode) / 50;
  if (splitMode < 50 or depth < 2)
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

  sum /= static_cast<float>(set.size());float
  p;
  for (int i = 0; i < num_parts; i++) {
    p = sum.at<float>(0, i);
    if (p > 0)
      n_entropy += p * log(p);
  }
  return n_entropy;
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

bool
MPSample::generateSplit
  (
  const std::vector<MPSample*> &data,
  boost::mt19937 *rng,
  ForestParam fp,
  Split &split
  )
{
  int patchSize = fp.face_size * fp.patch_size_ratio;
  int num_feat_channels = data[0]->m_image->m_feature_channels.size();
  split.feature.generate(patchSize, rng, num_feat_channels);

  split.num_thresholds = 25;
  split.margin = 0;

  return true;
};

void
MPSample::makeLeaf
  (
  MPLeaf &leaf,
  const std::vector<MPSample*> &set,
  const std::vector<float> &poppClasses,
  int leaf_id
  )
{
  int num_parts;
  if (set.size() > 0)
  {
    num_parts = set[0]->m_part_offsets.size();
  }
  else
  {
    leaf.forgound = 0;
    num_parts = 0;
    PRINT("something is wrong");
  }

  int nElements = set.size();

  leaf.parts_offset.clear();
  leaf.parts_offset.resize(num_parts);
  leaf.variance.resize(num_parts);
  leaf.pF.resize(num_parts);
  leaf.nSamples = nElements;

  for (int j = 0; j < num_parts; j++)
  {
    leaf.parts_offset[j] = cv::Point(0, 0);
    leaf.variance[j] = boost::numeric::bounds<float>::highest();
    leaf.pF[j] = 0;
  }
  leaf.patch_offset = cv::Point(0, 0);
  leaf.forgound = 0;

  std::vector<MPSample*>::const_iterator itSample;
  int size = 0;
  for (itSample = set.begin(); itSample < set.end(); ++itSample)
  {
    if ((*itSample)->m_is_positive)
      size++;
  }

  if (size > 0)
  {
    for (int j = 0; j < num_parts; j++)
    {
      cv::Point_<int> mean(0.0, 0.0);
      float sumDist = 0;

      for (itSample = set.begin(); itSample < set.end(); ++itSample)
      {
        if ((*itSample)->m_is_positive)
        {
          sumDist += (*itSample)->m_prob.at<float>(0, j);
          mean += (*itSample)->m_part_offsets[j];
        }
      }
      mean.x /= static_cast<int>(size);
      mean.y /= static_cast<int>(size);

      leaf.pF[j] = static_cast<float>(sumDist) / size;
      leaf.parts_offset[j] = mean;

      double var = 0.0;
      for (itSample = set.begin(); itSample < set.end(); ++itSample)
      {
        if ((*itSample)->m_is_positive)
        {
          int x = (*itSample)->m_part_offsets[j].x;
          int y = (*itSample)->m_part_offsets[j].y;
          float dist = sqrt((x - mean.x) * (x - mean.x) + (y - mean.y) * (y - mean.y));
          var += dist;
        }
      }
      var /= size;
      leaf.variance[j] = var;

//            cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << var<<endl;
//            {
//				cv::Point_<int> mean(0.0,0.0);
//				cv::Point_<int> mean_sq(0.0,0.0);
//
//				for ( itSample = set.begin(); itSample < set.end(); ++itSample ){
//					if( (*itSample)->isPos){
//						mean += (*itSample)->part_offsets[j];
//						mean_sq.x += (*itSample)->part_offsets[j].x * (*itSample)->part_offsets[j].x;
//						mean_sq.y += (*itSample)->part_offsets[j].y * (*itSample)->part_offsets[j].y;
//					}
//				}
//
//				mean.x /= size;
//				mean.y /= size;
//
//				var = mean_sq.x / size - mean.x * mean.x+
//				      mean_sq.y / size - mean.y * mean.y;
//	            cout <<"leaf: offset["<<mean.x<<","<<mean.y<<"] pf:"<<leaf.pF[j]<< " var:" << sqrt(var) <<endl;
//
//            }

    }
    leaf.patch_offset = cv::Point(0, 0);
    for (itSample = set.begin(); itSample < set.end(); ++itSample)
    {
      if ((*itSample)->m_is_positive)
      {
        leaf.patch_offset += (*itSample)->m_patch_offset;
      }
    }
    leaf.patch_offset.x /= static_cast<int>(size);
    leaf.patch_offset.y /= static_cast<int>(size);

    leaf.forgound = size / static_cast<float>(set.size());
  }
};

// Not needed for this task
void
MPSample::calcWeightClasses
  (
  std::vector<float> &poppClasses,
  const std::vector<MPSample*> &set
  )
{
  poppClasses.resize(1);
  int size = 0;
  std::vector<MPSample*>::const_iterator itSample;

  // Count samples near the feature point
  for (itSample = set.begin(); itSample < set.end(); ++itSample)
  {
    if ((*itSample)->m_is_positive)
    {
      size++;
    }
  }
  poppClasses[0] = size / static_cast<float>(set.size() - size);
};
