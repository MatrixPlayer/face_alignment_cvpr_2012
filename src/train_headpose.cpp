/** ****************************************************************************
 *  @file    train_headpose.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Tree.hpp>
#include <ImageSample.hpp>
#include <HeadPoseSample.hpp>
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

#define NEG_PATCHES_RATIO 0.2f
#define MAX_NEG_ATTEMPTS 100

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
  ForestParam hp_param,
  std::vector< std::vector<FaceAnnotation> > &ann,
  int idx_tree
  )
{
  // Try to read the head pose conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s/tree_%03d.txt", hp_param.tree_path.c_str(), idx_tree);
  PRINT("Read head pose regression tree: " << tree_path);
  Tree<HeadPoseSample> *tree;
  bool is_tree_load = Tree<HeadPoseSample>::load(&tree, tree_path);

  // Estimate the number of images available for each class
  int imgs_per_class = hp_param.nimages;
  for (unsigned int i=0; i < ann.size(); i++)
    imgs_per_class = std::min(imgs_per_class, static_cast<int>(ann[i].size()));
  PRINT("Number of images per class: " << imgs_per_class);

  // Random annotations ordered by head pose
  srand(idx_tree+1);
  std::vector<FaceAnnotation> annotations;
  for (unsigned int i=0; i < ann.size(); i++)
  {
    std::random_shuffle(ann[i].begin(), ann[i].end());
    annotations.insert(annotations.end(), ann[i].begin(), ann[i].begin()+imgs_per_class);
  }

  std::vector<HeadPoseSample*> hp_samples;
  hp_samples.reserve(annotations.size()*hp_param.npatches);
  PRINT("Total number of images: " << annotations.size());
  PRINT("Reserved patches: " << annotations.size()*hp_param.npatches);

  boost::mt19937 rng;
  rng.seed(idx_tree+1);
  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < annotations.size(); i++, ++show_progress)
  {
    // Load image
    TRACE("Evaluate image: " << annotations[i].url);
    cv::Mat img = loadImage(hp_param.image_path, annotations[i].url);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert image to gray scale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Scale image and annotations
    float scale = static_cast<float>(hp_param.face_size)/static_cast<float>(annotations[i].bbox.width);
    cv::Mat img_scaled;
    cv::resize(img_gray, img_scaled, cv::Size(img_gray.cols*scale, img_gray.rows*scale), 0, 0);
    annotations[i].bbox.x *= scale;
    annotations[i].bbox.y *= scale;
    annotations[i].bbox.width *= scale;
    annotations[i].bbox.height *= scale;

    // Extract patches from this image sample
    ImageSample *sample = new ImageSample(img_scaled, hp_param.features, false);

    // Extract positive patches
    int patch_size = hp_param.getPatchSize();
    boost::uniform_int<> dist_pos(0, hp_param.face_size-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_pos(rng, dist_pos);
    for (int j=0; j < hp_param.npatches*(1-NEG_PATCHES_RATIO); j++)
    {
      cv::Rect bbox = cv::Rect(annotations[i].bbox.x+rand_pos(), annotations[i].bbox.y+rand_pos(), patch_size, patch_size);
      int label = annotations[i].pose + 2;
      HeadPoseSample *hps = new HeadPoseSample(sample, annotations[i].bbox, bbox, label);
      hp_samples.push_back(hps);
    }

    // Extract negative patches
    boost::uniform_int<> dist_neg_x(0, img_scaled.cols-patch_size-1);
    boost::uniform_int<> dist_neg_y(0, img_scaled.rows-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_x(rng, dist_neg_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_y(rng, dist_neg_y);
    for (int j=0; j < hp_param.npatches*NEG_PATCHES_RATIO; j++)
    {
      cv::Rect bbox, inter;
      int attempt = 0, max_attempts = MAX_NEG_ATTEMPTS;
      do
      {
        bbox = cv::Rect(rand_neg_x(), rand_neg_y(), patch_size, patch_size);
        inter = intersect(bbox, annotations[i].bbox);
        attempt++;
      } while ((inter.height != 0 || inter.width != 0) && (attempt < max_attempts));
      if (attempt == max_attempts)
        continue;
      HeadPoseSample *hps = new HeadPoseSample(sample, annotations[i].bbox, bbox, -1);
      hp_samples.push_back(hps);
    }
  }
  PRINT("Used patches: " << hp_samples.size());

  if (is_tree_load && !tree->isFinished())
    tree->update(hp_samples, &rng);
  else
    tree = new Tree<HeadPoseSample>(hp_samples, hp_param, &rng, tree_path);
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
  // Train head pose
  std::string headpose_config_file = "data/config_headpose.txt";

  // Parse configuration file
  ForestParam hp_param;
  if (!loadConfigFile(headpose_config_file, hp_param))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(hp_param.image_path, annotations))
    return EXIT_FAILURE;

  // Separate annotations by head-pose classes
  std::vector< std::vector<FaceAnnotation> > ann(NUM_HEADPOSE_CLASSES);
  for (unsigned int i=0; i < annotations.size(); i++)
    ann[annotations[i].pose+2].push_back(annotations[i]);

  // Evaluate performance using 90% train and 10% test
  std::vector< std::vector<FaceAnnotation> > train_ann(NUM_HEADPOSE_CLASSES);
  for (unsigned int i=0; i < ann.size(); i++)
  {
    int num_train_imgs = static_cast<int>(ann[i].size() * TRAIN_IMAGES_PERCENTAGE);
    train_ann[i].insert(train_ann[i].begin(), ann[i].begin(), ann[i].begin()+num_train_imgs);
  }

  // Train head-pose forests
  /*for (int i=0; i < hp_param.ntrees; i++)
    trainTree(hp_param, train_ann, i);*/

  // Train head-pose tree
  int idx_tree = atoi(argv[1]);
  trainTree(hp_param, train_ann, idx_tree);

  return EXIT_SUCCESS;
};
