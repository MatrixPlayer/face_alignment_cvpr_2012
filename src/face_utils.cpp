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

      // Path to load images
      std::getline(file, line);
      std::getline(file, line);
      param.image_path = line;
      PRINT("> Path to images: " << line);

      // Path load or save trees
      std::getline(file, line);
      std::getline(file, line);
      param.tree_path = line;
      PRINT("> Paths to trees: " << line);

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
  std::vector<HeadPoseLeaf*> &leafs,
  int step_size
  )
{
  ForestParam param = forest.getParam();
  int patch_size = param.face_size * param.patch_size_ratio;
  std::vector<HeadPoseSample> samples;
  samples.reserve((face_bbox.width-patch_size+1) * (face_bbox.height-patch_size+1));
  for (int x = face_bbox.x; x < face_bbox.x+face_bbox.width-patch_size; x += step_size)
  {
    for (int y = face_bbox.y; y < face_bbox.y+face_bbox.height-patch_size; y += step_size)
    {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(HeadPoseSample(&sample, patch_box));
    }
  }

  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::ThreadPool e(num_treads);
  int num_trees = forest.numberOfTrees();
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i=0; i < samples.size(); i++)
    e.submit(boost::bind(&Forest<HeadPoseSample>::evaluateMT, forest, &samples[i], &leafs[i*num_trees]));
  e.join_all();
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
  ForestParam param = forest.getParam();
  int patch_size = param.face_size * param.patch_size_ratio;
  std::vector<MPSample> samples;
  samples.reserve((face_bbox.width-patch_size+1) * (face_bbox.height-patch_size+1));
  for (int x = face_bbox.x; x < face_bbox.x+face_bbox.width-patch_size; x += options.step_size)
  {
    for (int y = face_bbox.y; y < face_bbox.y+face_bbox.height-patch_size; y += options.step_size)
    {
      cv::Rect patch_box(x, y, patch_size, patch_size);
      samples.push_back(MPSample(&sample, patch_box));
    }
  }

  int num_treads = boost::thread::hardware_concurrency();
  boost::thread_pool::ThreadPool e(num_treads);
  int num_trees = forest.numberOfTrees();
  std::vector<MPLeaf*> leafs;
  leafs.resize(samples.size() * num_trees);
  for (unsigned int i=0; i < samples.size(); i++)
    e.submit(boost::bind(&Forest<MPSample>::evaluateMT, forest, &samples[i], &leafs[i*num_trees]));
  e.join_all();

  int num_parts = static_cast<int>(votes.size());
  std::vector<MPLeaf*>::iterator it_leaf;
  int i_sample = 0;
  for (it_leaf = leafs.begin(); it_leaf < leafs.end(); it_leaf++)
  {
    CV_Assert(static_cast<int>(samples.size()) > (i_sample / num_trees));
    int off_set_x = samples[i_sample/num_trees].rect.x + patch_size/2;
    int off_set_y = samples[i_sample/num_trees].rect.y + patch_size/2;
    for (int i=0; i < num_parts; i++)
    {
      float min_pf = options.min_pf;
      if (i == 0 || i == 7)
        min_pf *= 1.5;

      if ((*it_leaf)->forgound > options.min_forground &&
          (*it_leaf)->pF[i] > min_pf &&
          (*it_leaf)->variance[i] < options.max_variance &&
          (*it_leaf)->nSamples > options.min_samples)
      {
        Vote v;
        v.pos.x = (*it_leaf)->parts_offset[i].x + off_set_x;
        v.pos.y = (*it_leaf)->parts_offset[i].y + off_set_y;
        v.weight = (*it_leaf)->forgound;
        v.check = true;
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

/*void
extract_face
  (
  const cv::Mat& img,
  FaceAnnotation& ann,
  cv::Mat& face,
  int offset_x,
  int offset_y
  )
{
  // extract face
  cv::Rect bigbox = cv::Rect(ann.bbox.x - offset_x, ann.bbox.y - offset_y,
      ann.bbox.width + offset_x * 2, ann.bbox.height + offset_y * 2);
  cv::Rect facebbox = intersect(bigbox, cv::Rect(0, 0, img.cols, img.rows));

  face = img(facebbox);
  //update GT
  for (unsigned int j = 0; j < ann.parts.size(); j++)
  {
    ann.parts[j].x -= (facebbox.x - ann.bbox.x);
    ann.parts[j].y -= (facebbox.y - ann.bbox.y);
  }

  ann.bbox = facebbox;
  ann.bbox.x = 0;
  ann.bbox.y = 0;
};

void
rescale_img
  (
  const cv::Mat &src,
  cv::Mat &dest,
  float scale,
  FaceAnnotation &ann
  )
{
  cv::resize(src, dest, cv::Size(src.cols * scale, src.rows * scale), 0, 0);

  for (unsigned int j = 0; j < ann.parts.size(); j++)
  {
    ann.parts[j].x *= scale;
    ann.parts[j].y *= scale;
  }
  ann.bbox.x *= scale;
  ann.bbox.y *= scale;
  ann.bbox.width *= scale;
  ann.bbox.height *= scale;
};

void
plot_face
  (
  const cv::Mat& img,
  FaceAnnotation ann
  )
{
  cv::Mat plot = img.clone();
  for (int i = 0; i < static_cast<int> (ann.parts.size()); i++)
  {
    int x = ann.parts[i].x + ann.bbox.x;
    int y = ann.parts[i].y + ann.bbox.y;
    cv::circle(plot, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
  }
  cv::rectangle(plot, ann.bbox, cv::Scalar(0, 0, 255), 3);
  cv::imshow("Face ", plot);
  cv::waitKey(0);
};

void
plot_ffd_votes
  (
  const cv::Mat& face,
  std::vector<std::vector<Vote> > &votes,
  std::vector<cv::Point> results,
  std::vector<cv::Point> gt)
{
  int num_parts = static_cast<int> (votes.size());
  std::vector < cv::Mat > plots;
  cv::Mat plotFace = face.clone();

  for (int i = 0; i < num_parts; i++)
  {
    cv::Mat plot = cv::Mat(face.cols, face.rows, CV_32FC1);
    plot.setTo(cv::Scalar::all(0.0));

    for (unsigned int j = 0; j < votes[i].size(); j++)
    {
      Vote& v = votes[i][j];
      if (v.pos.x > 0 and v.pos.x < face.cols and v.pos.y > 0 and v.pos.y < face.rows)
      {
        plot.at<float> (v.pos.y, v.pos.x) += v.weight;
      }
    }
    plots.push_back(plot);
    if (i < static_cast<int> (results.size()))
    {
      int x = results[i].x;
      int y = results[i].y;
      cv::circle(plotFace, cv::Point_<int>(x, y), 3, cv::Scalar(255, 255, 255, 0));
    }
    if (i < static_cast<int> (gt.size()))
    {
      int x = gt[i].x;
      int y = gt[i].y;
      cv::circle(plotFace, cv::Point_<int>(x, y), 3, cv::Scalar(0, 0, 0, 0));
    }
  }
  for (int i = 0; i < num_parts; i++)
  {
    cv::imshow("Plot Votes", plots[i]);
    cv::imshow("Face results", plotFace);
  }
  cv::waitKey(0);
};*/
