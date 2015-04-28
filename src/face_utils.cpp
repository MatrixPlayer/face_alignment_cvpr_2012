/** ****************************************************************************
 *  @file    face_utils.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/01
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <face_utils.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/highgui/highgui.hpp>

cv::Mat
loadImage
  (
  std::string path,
  std::string name
  )
{
  std::size_t pos = path.rfind("/") + 1;
  std::string filename = path.substr(0,pos) + name;
  return cv::imread(filename, cv::IMREAD_COLOR);
};

void
enlargeFace
  (
  cv::Mat img,
  cv::Rect &enlarge_bbox,
  FaceAnnotation &annotation
  )
{
  cv::Point2i offset(annotation.bbox.width*0.1, annotation.bbox.height*0.1);
  cv::Rect aux = cv::Rect(annotation.bbox.x-offset.x, annotation.bbox.y-offset.y,
                          annotation.bbox.width+(offset.x*2), annotation.bbox.height+(offset.y*2));
  enlarge_bbox = intersect(aux, cv::Rect(0,0,img.cols,img.rows));
  annotation.bbox.x = 0;
  annotation.bbox.y = 0;
  annotation.bbox.width = enlarge_bbox.width;
  annotation.bbox.height = enlarge_bbox.height;
  for (unsigned int i=0; i < annotation.parts.size(); i++)
    annotation.parts[i] += offset;
};

bool
loadConfigFile
  (
  std::string path,
  ForestParam &param
  )
{
  if (boost::filesystem::exists(path.c_str()))
  {
    boost::iostreams::stream <boost::iostreams::file_source> file(path.c_str());
    std::string line;
    if (file.is_open())
    {
      PRINT("Open configuration file: " << path);

      // Path to image annotations
      std::getline(file, line);
      std::getline(file, line);
      param.image_path = line;
      PRINT("> Path to image annotations: " << line);

      // Path load or save trees
      std::getline(file, line);
      std::getline(file, line);
      param.tree_path = line;
      PRINT("> Path to trees: " << line);

      // Number of trees per forest
      std::getline(file, line);
      std::getline(file, line);
      param.ntrees = boost::lexical_cast<int>(line);

      // Number of tests to find the optimal split
      std::getline(file, line);
      std::getline(file, line);
      param.ntests = boost::lexical_cast<int>(line);

      // Maximum depth
      std::getline(file, line);
      std::getline(file, line);
      param.max_depth = boost::lexical_cast<int>(line);

      // Minimum patches per node
      std::getline(file, line);
      std::getline(file, line);
      param.min_patches = boost::lexical_cast<int>(line);

      // Samples per tree
      std::getline(file, line);
      std::getline(file, line);
      param.nimages = boost::lexical_cast<int>(line);

      // Patches per image
      std::getline(file, line);
      std::getline(file, line);
      param.npatches = boost::lexical_cast<int>(line);

      // Face size in pixels
      std::getline(file, line);
      std::getline(file, line);
      param.face_size = boost::lexical_cast<int>(line);

      // Patch size ratio
      std::getline(file, line);
      std::getline(file, line);
      param.patch_size_ratio = boost::lexical_cast<float>(line);

      // Feature channels
      std::getline(file, line);
      std::getline(file, line);
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of(" "));
      for (unsigned int i=0; i < strs.size(); i++)
        param.features.push_back(boost::lexical_cast<float>(strs[i]));

      return true;
    }
  }

  // Default values for ForestParam
  PRINT("Default ForestParam initialization ...");
  param.max_depth = 15;
  param.min_patches = 20;
  param.ntests = 250;
  param.ntrees = 10;
  param.nimages = 500;
  param.npatches = 200;
  param.face_size = 100;
  param.patch_size_ratio = 0.25;
  return false;
};

bool
loadAnnotations
  (
  std::string path,
  std::vector<FaceAnnotation> &annotations
  )
{
  if (boost::filesystem::exists(path.c_str()))
  {
    boost::iostreams::stream <boost::iostreams::file_source> file(path.c_str());
    std::string line;
    PRINT("Open annotations file: " << path);
    while (std::getline(file, line))
    {
      std::vector<std::string> strs;
      boost::split(strs, line, boost::is_any_of(" "));

      // Avoid comments
      if (strs[0] == "#") continue;

      FaceAnnotation ann;
      ann.url = strs[0];
      ann.bbox.x = boost::lexical_cast<int>(strs[1]);
      ann.bbox.y = boost::lexical_cast<int>(strs[2]);
      ann.bbox.width = boost::lexical_cast<int>(strs[3]);
      ann.bbox.height = boost::lexical_cast<int>(strs[4]);
      ann.pose = boost::lexical_cast<int>(strs[5]);
      int num_points = boost::lexical_cast<int>(strs[6]);
      ann.parts.resize(num_points);
      for (int i=0; i < num_points; i++)
      {
        ann.parts[i].x = boost::lexical_cast<int>(strs[7 + (2*i)]);
        ann.parts[i].y = boost::lexical_cast<int>(strs[8 + (2*i)]);
      }
      annotations.push_back(ann);
    }
    return true;
  }
  return false;
};

void
getHeadPoseVotesMT
  (
  const ImageSample &sample,
  const Forest<HeadPoseSample> &forest,
  cv::Rect face_bbox,
  float *headpose,
  float *variance,
  HeadPoseEstimatorOption options
  )
{
  int patch_size = forest.getParam().getPatchSize();
  int num_trees = forest.numberOfTrees();

  // Reserve patches like a dense extraction for options.step_size == 1
  std::vector<HeadPoseSample> samples;
  samples.reserve((face_bbox.width-patch_size) * (face_bbox.height-patch_size));
  for (int x=face_bbox.x; x < face_bbox.x+face_bbox.width-patch_size; x += options.step_size)
  {
    for (int y=face_bbox.y; y < face_bbox.y+face_bbox.height-patch_size; y += options.step_size)
    {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(HeadPoseSample(&sample, patch_box));
    }
  }

  // Process each patch using all the trees
  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::ThreadPool e(num_treads);
  std::vector<HeadPoseLeaf*> leafs;
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i=0; i < samples.size(); i++)
    e.submit(boost::bind(&Forest<HeadPoseSample>::evaluateMT, forest, &samples[i], &leafs[i*num_trees]));
  e.join_all();

  // Parse collected leafs
  float n = 0, sum = 0, sum_sq = 0;
  for (unsigned int i=0; i < leafs.size(); ++i)
  {
    // Only use leafs that contains a minimum of positive patches
    if (leafs[i]->hp_foreground > options.min_foreground_probability)
    {
      float m = 0; // predicted label [0..4]
      for (int j=0; j < options.num_head_pose_labels; j++)
        m += leafs[i]->hp_labels[j] * j;
      m /= (leafs[i]->hp_nsamples * leafs[i]->hp_foreground);
      sum += m;
      sum_sq += m * m;
      n++;
    }
  }
  float mean = sum / n;
  float var  = (sum_sq / n) - (mean * mean);

  mean -= 2;
  var  *= NORM_HEADPOSE_VARIANCE_FACTOR;

  *headpose = mean;
  *variance = var;
};

void
getFacialFeaturesVotesMT
  (
  const ImageSample &sample,
  const Forest<MPSample> &forest,
  cv::Rect face_bbox,
  std::vector< std::vector<Vote> > &votes,
  MultiPartEstimatorOption options
  )
{
  int patch_size = forest.getParam().getPatchSize();
  // Reserve patches like a dense extraction for options.step_size == 1
  std::vector<MPSample> samples;
  samples.reserve((face_bbox.width-patch_size) * (face_bbox.height-patch_size));
  for (int x=face_bbox.x; x < face_bbox.x+face_bbox.width-patch_size; x += options.step_size)
  {
    for (int y=face_bbox.y; y < face_bbox.y+face_bbox.height-patch_size; y += options.step_size)
    {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(MPSample(&sample, patch_box));
    }
  }

  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::ThreadPool e(num_treads);
  std::vector<MPLeaf*> leafs;
  int num_trees = forest.numberOfTrees();
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i=0; i < samples.size(); i++)
    e.submit(boost::bind(&Forest<MPSample>::evaluateMT, forest, &samples[i], &leafs[i*num_trees]));
  e.join_all();

  // Parse collected leafs
  int i_sample = 0;
  for (std::vector<MPLeaf*>::iterator it_leaf = leafs.begin(); it_leaf < leafs.end(); it_leaf++)
  {
    CV_Assert(static_cast<int>(samples.size()) > (i_sample/num_trees));
    int offset_x = samples[i_sample/num_trees].getPatch().x + patch_size/2;
    int offset_y = samples[i_sample/num_trees].getPatch().y + patch_size/2;
    for (unsigned int i=0; i < votes.size(); i++)
    {
      float min_pf = options.min_pf;
      if (i == 0 || i == 7) // probability in outer eye parts
        min_pf *= 1.5;

      if ((*it_leaf)->mp_foreground > options.min_forground && (*it_leaf)->mp_prob_foreground[i] > min_pf &&
          (*it_leaf)->mp_parts_variance[i] < options.max_variance && (*it_leaf)->mp_samples > options.min_samples)
      {
        Vote v;
        v.pos.x  = (*it_leaf)->mp_parts_offset[i].x + offset_x;
        v.pos.y  = (*it_leaf)->mp_parts_offset[i].y + offset_y;
        v.weight = (*it_leaf)->mp_foreground;
        v.check  = true;
        votes[i].push_back(v);
      }
    }
    i_sample++;
  }
};

float
areaUnderCurve
  (
  float x1,
  float x2,
  double mean,
  double std
  )
{
  double sum = 0;
  double step = 0.01;
  double t;
  for (double x = x1; x < x2; x += step)
  {
    t = (x - mean) / std;
    sum += exp(-0.5 * (t * t)) * step;
  }

  return sum * 1.0 / (std * sqrt(2 * M_PI));
};

cv::Rect
intersect
  (
  const cv::Rect r1,
  const cv::Rect r2
  )
{
  // Find overlapping region
  cv::Rect intersection;
  intersection.x = (r1.x < r2.x) ? r2.x : r1.x;
  intersection.y = (r1.y < r2.y) ? r2.y : r1.y;
  intersection.width = (r1.x + r1.width < r2.x + r2.width) ? r1.x + r1.width : r2.x + r2.width;
  intersection.width -= intersection.x;
  intersection.height = (r1.y + r1.height < r2.y + r2.height) ? r1.y + r1.height : r2.y + r2.height;
  intersection.height -= intersection.y;

  // Check for non-overlapping regions
  if ((intersection.width <= 0) || (intersection.height <= 0))
    intersection = cv::Rect(0, 0, 0, 0);

  return intersection;
};

void
plotVotes
  (
  const cv::Mat &img_gray,
  std::vector< std::vector<Vote> > &votes,
  std::vector<cv::Point> ffd_cordinates
  )
{
  cv::Scalar black_color(0, 0, 0);
  cv::Scalar white_color(255, 255, 255);
  cv::Mat face = img_gray.clone();
  for (int i=0; i < static_cast<int>(votes.size()); i++)
  {
    cv::Mat plot = cv::Mat(img_gray.cols, img_gray.rows, CV_32FC1);
    plot.setTo(black_color);
    for (unsigned int j=0; j < votes[i].size(); j++)
    {
      Vote &v = votes[i][j];
      if ((v.pos.x > 0) && (v.pos.x < img_gray.cols) && (v.pos.y > 0) && (v.pos.y < img_gray.rows))
        plot.at<float>(v.pos.y, v.pos.x) += v.weight;
    }
    if (i < static_cast<int>(ffd_cordinates.size()))
      cv::circle(face, ffd_cordinates[i], 3, white_color);

    cv::imshow("Votes", plot);
    cv::imshow("Facial point", face);
    cv::waitKey(0);
  }
};
