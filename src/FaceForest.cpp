/** ****************************************************************************
 *  @file    FaceForest.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/06
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <FaceForest.hpp>
#include <boost/filesystem.hpp>
#include <face_utils.hpp>
#include <MeanShift.hpp>

FaceForest::FaceForest
  (
  FaceForestOptions option
  ) :
    m_ff_options(option)
{
  // Loading face cascade classifier on m_face_cascade
  PRINT("Loading face cascade classifier");
  if (!m_face_cascade.load(option.fd_option.path_face_cascade))
  {
    ERROR("(!) Error loading cascade classifier");
    return;
  }

  // Loading head-pose forest on m_hp_forest
  PRINT("Loading head-pose forest");
  if (!m_hp_forest.load(option.hp_forest_param.tree_path, option.hp_forest_param))
  {
    ERROR("(!) Error loading head-pose forest");
    return;
  }

  // Loading facial-feature-detector trees on m_mp_forest
  /*PRINT("Loading facial-feature-detector forest");
  num_trees = option.mp_forest_param.ntrees;
  m_mp_forest.setParam(option.mp_forest_param);
  getPathsToTrees(option.mp_forest_param.tree_path, m_ff_options.mp_tree_paths);
  loadingAllTrees(m_ff_options.mp_tree_paths);*/

  is_inizialized = true;
};

void
FaceForest::estimateHeadPose
  (
  const ImageSample &sample,
  const cv::Rect &face_bbox,
  const Forest<HeadPoseSample> &forest,
  HeadPoseEstimatorOption options,
  float *headpose,
  float *variance
  )
{
  // Collect leafs
  std::vector<HeadPoseLeaf*> leafs;
  getHeadPoseVotesMT(sample, forest, face_bbox, leafs, options.step_size);

  // Parse leaf
  int n = 0;
  float sum = 0,  sum_sq = 0;
  for (unsigned int i=0; i < leafs.size(); ++i)
  {
    if (leafs[i]->forgound > options.min_forground_probability)
    {
      float m = 0;
      for (int j=0; j < options.num_head_pose_labels; j++)
        m += leafs[i]->hist_labels[j] * j;
      m /= (leafs[i]->nSamples * leafs[i]->forgound);
      sum += m;
      sum_sq += m * m;
      n++;
    }
  }
  float mean = sum / n;
  float var  = (sum_sq / n) - (mean * mean);

  const float norm_factor = 0.05;
  var  *= norm_factor;
  mean -= 2;

  *headpose = mean;
  *variance = var;
};

void
FaceForest::estimateFacialFeatures
  (
  const ImageSample &sample,
  const cv::Rect face_bbox,
  const Forest<MPSample> &forest,
  MultiPartEstimatorOption options,
  std::vector<cv::Point> &ffd_cordinates
  )
{
  int num_parts = options.num_parts;
  std::vector < std::vector<Vote> > votes(num_parts);
  getFacialFeaturesVotesMT(sample, forest, face_bbox, votes, options);

  ffd_cordinates.clear();
  ffd_cordinates.resize(num_parts);
  MeanShiftOption ms_option;
  for (int j = 0; j < num_parts; j++)
    MeanShift::shift(votes[j], ffd_cordinates[j], ms_option);

  //boost::thread_pool::ThreadPool e;
  //for (int j=0; j < num_parts; j++)
  //  e.submit(boost::bind(&MeanShift::shift, votes[j], ffd_cordinates[j], 10, 3));
  //e.join_all();

  //std::vector<cv::Point> gt;
  //plot_ffd_votes(sample.featureChannels[0], votes, ffd_cordinates, gt);
};

void
FaceForest::showResults
  (
  std::vector<Face> &faces,
  upm::Viewer &viewer
  )
{
  cv::Scalar red_color(0, 0, 255);
  cv::Scalar green_color(0, 255, 0);
  for (unsigned int i=0; i < faces.size(); i++)
  {
    for (unsigned int j=0; j < faces[i].ffd_cordinates.size(); j++)
    {
      int x = faces[i].ffd_cordinates[j].x + faces[i].bbox.x;
      int y = faces[i].ffd_cordinates[j].y + faces[i].bbox.y;
      viewer.circle(x, y, 3, -1, green_color);
    }
    cv::Rect bbox = faces[i].bbox;
    cv::Point_<int> a(bbox.x, bbox.y);
    cv::Point_<int> b(bbox.x + bbox.width, bbox.y);
    cv::Point_<int> c(bbox.x, bbox.y + bbox.height);
    cv::Point_<int> d(bbox.x + bbox.width, bbox.y + bbox.height);
    cv::Point_<int> y1(bbox.x, bbox.y + (bbox.height / 2));
    cv::Point_<int> y2(bbox.x + bbox.width, bbox.y + (bbox.height / 2));
    float yaw = faces[i].headpose * 80;
    if (yaw < 0)
      y2.x -= yaw;
    else
      y1.x -= yaw;

    viewer.line(a.x, a.y, b.x, b.y, 2, red_color);
    viewer.line(c.x, c.y, d.x, d.y, 2, red_color);
    viewer.line(a.x, a.y, y1.x, y1.y, 2, red_color);
    viewer.line(y1.x, y1.y, c.x, c.y, 2, red_color);
    viewer.line(b.x, b.y, y2.x, y2.y, 2, red_color);
    viewer.line(y2.x, y2.y, d.x, d.y, 2, red_color);
  }
};

void
FaceForest::detectFace
  (
  const cv::Mat &img,
  cv::CascadeClassifier &face_cascade,
  FaceDetectionOption fd_option,
  std::vector<cv::Rect> &faces_bboxes
  )
{
  cv::Size min_feature_size = cv::Size(fd_option.min_feature_size, fd_option.min_feature_size);
  face_cascade.detectMultiScale(img, faces_bboxes, fd_option.search_scale_factor, fd_option.min_neighbors, 0, min_feature_size);

  // The face detection boxes are too tight for us
  for (unsigned int i=0; i < faces_bboxes.size(); i++)
  {
    // Intersect with the original image to not exceed the limits
    int offset_x = faces_bboxes[i].width * 0.05;
    int offset_y = faces_bboxes[i].width * 0.15;
    cv::Rect r1 = cv::Rect(faces_bboxes[i].x-offset_x, faces_bboxes[i].y, faces_bboxes[i].width+(offset_x*2), faces_bboxes[i].height+(offset_y*2));
    cv::Rect r2 = cv::Rect(0, 0, img.cols, img.rows);
    cv::Rect roi = intersect(r1, r2);
    faces_bboxes[i] = roi;
  }
};

void FaceForest::analyzeImage
  (
  cv::Mat img,
  std::vector<Face> &faces
  )
{
  CV_Assert(is_inizialized);

  // Detect the face
  std::vector<cv::Rect> faces_bboxes;
  detectFace(img, m_face_cascade, m_ff_options.fd_option, faces_bboxes);
  TRACE("Number of detected faces: " << faces_bboxes.size());

  // Analyze each detected face
  for (unsigned int i=0; i < faces_bboxes.size(); i++)
  {
    Face f;
    analyzeFace(img, faces_bboxes[i], f);
    faces.push_back(f);
  }
};

void
FaceForest::analyzeFace
  (
  const cv::Mat img,
  cv::Rect face_bbox,
  Face &face,
  bool normalize
  )
{
  CV_Assert(is_inizialized);
  CV_Assert(img.type() == CV_8UC1);
  face.bbox = face_bbox;

  // Scale and extract face
  cv::Mat roi;
  float scale = static_cast<float>(m_ff_options.hp_forest_param.face_size)/static_cast<float>(face_bbox.width);
  cv::resize(img(face_bbox), roi, cv::Size(face_bbox.width*scale, face_bbox.height*scale), 0, 0);

  // Normalize image
  if (normalize)
    cv::equalizeHist(roi, roi);

  // Create image sample
  Timing timer;
  timer.start();
  ImageSample sample(roi, m_ff_options.hp_forest_param.features, fcf, true);
  TRACE("Creating image sample: " << timer.elapsed());
  timer.restart();

  /// Estimate head-pose
  float headpose = 0, variance = 0;
  estimateHeadPose(sample, cv::Rect(0,0,roi.cols,roi.rows), m_hp_forest, m_ff_options.hp_option, &headpose, &variance);
  face.headpose = headpose;

  /*// Compute area under curve
  int hist_size = 5;
  std::vector<float> poseT(hist_size + 1);
  poseT[0] = -2.5;
  poseT[1] = -0.35;
  poseT[2] = -0.20;
  poseT[3] = -poseT[2];
  poseT[4] = -poseT[1];
  poseT[5] = -poseT[0];

  std::vector<float> pose_freq(hist_size);
  float max_area = 0;
  int dominant_headpose = 0;
  for (int j=0; j < hist_size; j++)
  {
    float area = areaUnderCurve(poseT[j], poseT[j+1], headpose, sqrt(variance));
    pose_freq[j] = area;
    if (max_area < area)
    {
      max_area = area;
      dominant_headpose = j;
    }
  }
  timer.restart();

  // Add new trees based on the estimated head-pose
  CV_Assert(m_trees.size() == pose_freq.size());
  CV_Assert(static_cast<int>(m_trees.size()) == hist_size);
  m_mp_forest.cleanForest();

  for (unsigned int i=0; i < m_trees.size(); i++)
  {
    int ntrees = static_cast<int>(floor(pose_freq[i] * num_trees));
    for (int j=0; j < ntrees; j++)
      m_mp_forest.addTree(m_trees[i][j]);
  }

  // Correcting floor rounding errors
  for (int i = m_mp_forest.numberOfTrees(); i < num_trees; i++)
    m_mp_forest.addTree(m_trees[dominant_headpose][i]);

  /// Estimate facial feature points
  estimateFacialFeatures(sample, cv::Rect(0,0,roi.cols,roi.rows), m_mp_forest, m_ff_options.mp_option, face.ffd_cordinates);

  // Scale results
  for (unsigned int i=0; i < face.ffd_cordinates.size(); i++)
  {
    face.ffd_cordinates[i].x /= scale;
    face.ffd_cordinates[i].y /= scale;
  }*/
};

void
FaceForest::getPathsToTrees
  (
  std::string path,
  std::vector<std::string> &urls
  )
{
  boost::filesystem::path dir_path(path);
  boost::filesystem::directory_iterator end_it;
  for (boost::filesystem::directory_iterator it(dir_path); it != end_it; ++it)
  {
    if (is_directory(it->status()))
      urls.push_back(it->path().string());
  }
  sort(urls.begin(), urls.end());
  PRINT("> Number of forest directories found: " << urls.size());
};

void
FaceForest::loadingAllTrees
  (
  std::vector<std::string> urls
  )
{
  for (unsigned int i=0; i < urls.size(); i++)
  {
    std::vector<Tree<MPSample>*> all_trees;
    for (int j=0; j < num_trees; j++)
    {
      char buffer[200];
      sprintf(buffer, "%s/tree_%03d.txt", urls[i].c_str(), j);
      std::string tree_path = buffer;
      PRINT("  Load " << tree_path);
      Forest<MPSample>::load_tree(tree_path, all_trees);
    }
    m_trees.push_back(all_trees);
  }
};
