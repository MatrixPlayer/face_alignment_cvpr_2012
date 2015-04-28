/** ****************************************************************************
 *  @file    demo.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/08
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Viewer.hpp>
#include <Tree.hpp>
#include <ImageSample.hpp>
#include <MPSample.hpp>
#include <FaceForest.hpp>
#include <Constants.hpp>
#include <face_utils.hpp>

#include <vector>
#include <string>
#include <cstdlib>
#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <opencv2/highgui/highgui.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
trainTree
  (
  ForestParam mp_param,
  std::vector<FaceAnnotation> &annotations
  )
{
  int idx_tree = 99;
  srand(idx_tree+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  std::vector<MPSample*> mp_samples;
  mp_samples.reserve(mp_param.nimages*mp_param.npatches);
  PRINT("Total number of images: " << mp_param.nimages);
  PRINT("Reserved patches: " << mp_param.nimages*mp_param.npatches);

  boost::mt19937 rng;
  rng.seed(idx_tree+1);
  boost::progress_display show_progress(mp_param.nimages);
  for (int i=0; i < mp_param.nimages; i++, ++show_progress)
  {
    // Load image
    TRACE("Evaluate image: " << annotations[i].url);
    cv::Mat img = loadImage(mp_param.image_path, annotations[i].url);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert to gray-scale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Scale image and annotations
    float scale = static_cast<float>(mp_param.face_size)/static_cast<float>(annotations[i].bbox.width);
    cv::Mat img_scaled;
    cv::resize(img_gray, img_scaled, cv::Size(img_gray.cols*scale, img_gray.rows*scale), 0, 0);
    annotations[i].bbox.x *= scale;
    annotations[i].bbox.y *= scale;
    annotations[i].bbox.width *= scale;
    annotations[i].bbox.height *= scale;
    for (unsigned int j=0; j < annotations[i].parts.size(); j++)
      annotations[i].parts[j] *= scale;

    // Extract face image and enlarge to make sure that all facial features are enclosed
    cv::Rect enlarge_bbox;
    enlargeFace(img_scaled, enlarge_bbox, annotations[i]);
    cv::Mat img_roi = img_scaled(enlarge_bbox);

    // Create image sample
    ImageSample *sample = new ImageSample(img_roi, mp_param.features, false);

    // Extract positive patches
    int patch_size = mp_param.getPatchSize();
    boost::uniform_int<> dist_x(1, img_roi.cols-patch_size-2);
    boost::uniform_int<> dist_y(1, img_roi.rows-patch_size-2);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
    for (int j=0; j < mp_param.npatches; j++)
    {
      cv::Rect bbox = cv::Rect(rand_x(), rand_y(), patch_size, patch_size);
      MPSample *mps = new MPSample(sample, bbox, annotations[i].parts, mp_param.face_size, true);
      mp_samples.push_back(mps);
    }
  }
  PRINT("Used patches: " << mp_samples.size());

  char tree_path[200];
  sprintf(tree_path, "%s/tree_%03d.txt", mp_param.tree_path.c_str(), idx_tree);
  Tree<MPSample> *tree = new Tree<MPSample>(mp_samples, mp_param, &rng, tree_path);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
evalForest
  (
  FaceForestOptions ff_options,
  std::vector<FaceAnnotation> &annotations
  )
{
  // Initialize face forest
  FaceForest ff(ff_options);

  upm::Viewer viewer;
  viewer.init(0, 0, "demo");

  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); ++i, ++show_progress)
  {
    // Load image
    TRACE("Evaluate image: " << annotations[i].url);
    cv::Mat img = loadImage(ff_options.mp_forest_param.image_path, annotations[i].url);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    std::vector<Face> faces;
    const bool use_predefined_bbox = false;
    if (use_predefined_bbox)
    {
      Face face;
      ff.analyzeFace(img, annotations[i].bbox, face);
      faces.push_back(face);
    }
    else
    {
      ff.analyzeImage(img, faces);
    }

    // Draw results
    viewer.resizeCanvas(img.cols, img.rows);
    viewer.beginDrawing();
    viewer.image(img, 0, 0, img.cols, img.rows);
    ff.showResults(faces, viewer);
    viewer.endDrawing(0);
  }
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
main
  (
  int argc,
  char **argv
  )
{
  // 0 - training a tree
  // 1 - evaluate feature points detector
  std::string ffd_config_file = "data/config_ffd.txt";
  std::string headpose_config_file = "data/config_headpose.txt";
  std::string face_cascade = "data/haarcascade_frontalface_alt.xml";
  int mode = 1;
  if (argc == 2)
    mode = boost::lexical_cast<int>(argv[1]);

  // Parse configuration file
  ForestParam mp_param;
  if (!loadConfigFile(ffd_config_file, mp_param))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(mp_param.image_path, annotations))
    return EXIT_FAILURE;

  switch (mode)
  {
    case 0:
    {
      trainTree(mp_param, annotations);
      break;
    }
    case 1:
    {
      ForestParam hp_param;
      if (!loadConfigFile(headpose_config_file, hp_param))
        return EXIT_FAILURE;

      FaceForestOptions ff_options;
      ff_options.fd_option.path_face_cascade = face_cascade;
      ff_options.hp_forest_param = hp_param;
      ff_options.mp_forest_param = mp_param;

      evalForest(ff_options, annotations);
      break;
    }
    default:
      PRINT("Unknown mode (0=training, 1=evaluate)");
  }

  return EXIT_SUCCESS;
};
