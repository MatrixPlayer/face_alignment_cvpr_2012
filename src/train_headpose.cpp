/** ****************************************************************************
 *  @file    train_headpose.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Constants.hpp>
#include <face_utils.hpp>
#include <Tree.hpp>
#include <ImageSample.hpp>
#include <HeadPoseSample.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <opencv2/opencv.hpp>

void
trainHeadposeForest
  (
  ForestParam &fp,
  std::vector<FaceAnnotation> &annotations,
  int offset
  )
{
  PRINT("Input offset: " << offset);
  srand(offset+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  // Try to read the head-pose conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s%03d.txt", fp.tree_path.c_str(), offset);
  PRINT("Read head-pose regression tree: " << tree_path);
  Tree<HeadPoseSample> *tree;
  bool is_tree_load = Tree<HeadPoseSample>::load(&tree, tree_path);

  // Separate annotations by head-pose classes
  std::vector< std::vector<FaceAnnotation> > cluster(Constants::NUM_HP_CLASSES);
  for (unsigned int i=0; i < annotations.size(); i++)
  {
    // Range is between -2 and +2 but we shift it to 0 - 5
    int label = annotations[i].pose + 2;
    cluster[label].push_back(annotations[i]);
  }

  // Estimate the number of images for each class
  int imgs_per_class = fp.nimages;
  for (unsigned int i=0; i < cluster.size(); i++)
    imgs_per_class = std::min(imgs_per_class, static_cast<int>(cluster[i].size()));

  annotations.clear();
  for (unsigned int i=0; i < cluster.size(); i++)
    for (int j=0; j < imgs_per_class; j++)
      annotations.push_back(cluster[i][j]);

  // 5 positive classes + 1 negative class
  std::vector<HeadPoseSample*> hp_samples;
  hp_samples.reserve(annotations.size()*fp.npatches);
  PRINT("Number of images per class: " << imgs_per_class);
  PRINT("Total number of images: " << annotations.size());
  PRINT("Reserved patches: " << annotations.size()*fp.npatches);

  boost::mt19937 rng;
  rng.seed(offset+1);
  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); i++, ++show_progress)
  {
    TRACE("Evaluate image: " << annotations[i].url);

    // Load image
    std::size_t pos = fp.image_path.find("lfw_ffd_ann.txt");
    std::string img_path = fp.image_path.substr(0, pos) + annotations[i].url;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Scale image and annotations
    cv::Mat img_scaled;
    float scale = static_cast<float>(fp.face_size)/static_cast<float>(annotations[i].bbox.width);
    cv::resize(img, img_scaled, cv::Size(img.cols*scale, img.rows*scale), 0, 0);
    annotations[i].bbox.x *= scale;
    annotations[i].bbox.y *= scale;
    annotations[i].bbox.width *= scale;
    annotations[i].bbox.height *= scale;
    for (unsigned int j=0; j < annotations[i].parts.size(); j++)
      annotations[i].parts[j] *= scale;

    // Extract features from an image
    ImageSample *sample = new ImageSample(img_scaled, fp.features, false);

    // Extract positive patches
    int patch_size = fp.patchSize();
    boost::uniform_int<> dist_pos(0, fp.face_size-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_pos(rng, dist_pos);
    for (int j=0; j < fp.npatches*(1-Constants::NEG_PATCHES_RATIO); j++)
    {
      cv::Rect bbox = cv::Rect(annotations[i].bbox.x+rand_pos(), annotations[i].bbox.y+rand_pos(), patch_size, patch_size);
      int label = annotations[i].pose + 2;
      HeadPoseSample *hps = new HeadPoseSample(sample, annotations[i].bbox, bbox, label);
      hp_samples.push_back(hps);
      //hps->show();
    }

    // Extract negative patches
    boost::uniform_int<> dist_neg_x(0, img_scaled.cols-patch_size-1);
    boost::uniform_int<> dist_neg_y(0, img_scaled.rows-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_x(rng, dist_neg_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_y(rng, dist_neg_y);
    for (int j=0; j < fp.npatches*Constants::NEG_PATCHES_RATIO; j++)
    {
      cv::Rect bbox, inter;
      int attempt = 0, max_attempts = Constants::MAX_NEG_ATTEMPTS;
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
      //hps->show();
    }
  }
  PRINT("Used patches: " << hp_samples.size());

  if (is_tree_load && !tree->isFinished())
    tree->updateTree(hp_samples, &rng);
  else
    tree = new Tree<HeadPoseSample>(hp_samples, fp, &rng, tree_path);
};

int
main
  (
  int argc,
  char **argv
  )
{
  // Usage: data/config_headpose.txt
  std::string config_file = argv[1];

  // Parse configuration file
  ForestParam fp;
  if (!loadConfigFile(config_file, fp))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(fp.image_path, annotations))
    return EXIT_FAILURE;

  // Train head-pose forest
  for (int i=0; i < fp.ntrees; i++)
    trainHeadposeForest(fp, annotations, i);

  return EXIT_SUCCESS;
};
