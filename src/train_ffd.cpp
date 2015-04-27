/** ****************************************************************************
 *  @file    train_ffd.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Tree.hpp>
#include <ImageSample.hpp>
#include <MPSample.hpp>
#include <Constants.hpp>
#include <face_utils.hpp>

#include <vector>
#include <string>
#include <cstdlib>
#include <boost/progress.hpp>
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
  std::vector<FaceAnnotation> &annotations,
  int idx_forest,
  int idx_tree
  )
{
  // Try to read the facial feature conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s/forest_%d/tree_%03d.txt", mp_param.tree_path.c_str(), idx_forest, idx_tree);
  PRINT("Read facial feature regression tree: " << tree_path);
  Tree<MPSample> *tree;
  bool is_tree_load = Tree<MPSample>::load(&tree, tree_path);

  srand(idx_tree+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  // Separate annotations by head-pose classes
  std::vector< std::vector<FaceAnnotation> > cluster(NUM_HEADPOSE_CLASSES);
  for (unsigned int i=0; i < annotations.size(); i++)
  {
    int label = annotations[i].pose + 2;
    cluster[label].push_back(annotations[i]);
  }

  // Estimate the number of images available for each class
  int imgs_per_class = mp_param.nimages;
  for (unsigned int i=0; i < cluster.size(); i++)
    imgs_per_class = std::min(imgs_per_class, static_cast<int>(cluster[i].size()));
  PRINT("Number of images per class: " << imgs_per_class);

  // Only annotations filtered by head pose
  annotations.clear();
  for (int i=0; i < imgs_per_class; i++)
    annotations.push_back(cluster[idx_forest][i]);

  std::vector<MPSample*> mp_samples;
  mp_samples.reserve(annotations.size()*mp_param.npatches);
  PRINT("Total number of images: " << annotations.size());
  PRINT("Reserved patches: " << annotations.size()*mp_param.npatches);

  boost::mt19937 rng;
  rng.seed(idx_tree+1);
  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < annotations.size(); i++, ++show_progress)
  {
    // Load image
    TRACE("Evaluate image: " << annotations[i].url);
    cv::Mat img = loadImage(mp_param.image_path, annotations[i].url);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert image to gray scale
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

    // Extract patches from this image sample
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

  if (is_tree_load && !tree->isFinished())
    tree->update(mp_samples, &rng);
  else
    tree = new Tree<MPSample>(mp_samples, mp_param, &rng, tree_path);
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
  // Train feature points detector
  std::string ffd_config_file = "data/config_ffd.txt";

  // Parse configuration file
  ForestParam mp_param;
  if (!loadConfigFile(ffd_config_file, mp_param))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(mp_param.image_path, annotations))
    return EXIT_FAILURE;

  // Train facial-feature tree
  int idx_forest = atoi(argv[1]);
  int idx_tree = atoi(argv[2]);
  trainTree(mp_param, annotations, idx_forest, idx_tree);

  // Train facial-feature forests
  //for (unsigned int i=0; i < NUM_HEADPOSE_CLASSES; i++)
  //  for (int j=0; j < mp_param.ntrees; j++)
  //    trainTree(mp_param, cluster[i], i, j);

  return EXIT_SUCCESS;
};
