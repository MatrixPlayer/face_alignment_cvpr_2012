/** ****************************************************************************
 *  @file    train_headpose.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Constants.hpp>
#include <Tree.hpp>
#include <ImageSample.hpp>
#include <HeadPoseSample.hpp>
#include <face_utils.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

int
main
  (
  int argc,
  char **argv
  )
{
  // Usage: data/config_headpose.txt 0
  std::string config_file = argv[1];
  int offset = boost::lexical_cast<int>(argv[2]);
  PRINT("Input offset: " << offset);

  // Parse configuration file
  ForestParam fp;
  if (!loadConfigFile(config_file, fp))
    return EXIT_FAILURE;

  // Try to read the head-pose conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s%03d.txt", fp.tree_path.c_str(), offset);
  PRINT("Read head-pose regression tree: " << tree_path);
  Tree<HeadPoseSample> *tree;
  Tree<HeadPoseSample>::load(&tree, tree_path);

  // Initialize random generator
  boost::mt19937 rng;
  rng.seed(offset+1);
  srand(offset+1);

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(fp.image_path, annotations))
    return EXIT_FAILURE;

  std::random_shuffle(annotations.begin(), annotations.end());

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
  hp_samples.reserve(imgs_per_class*(cluster.size()+1)*fp.npatches);
  PRINT("Number of images per class: " << imgs_per_class);
  PRINT("Total number of images: " << annotations.size());
  PRINT("Reserved patches: " << imgs_per_class*(cluster.size()+1)*fp.npatches);

  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); i++, ++show_progress)
  {
    PRINT("Evaluate image: " << annotations[i].url);

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
    boost::uniform_int<> dist(0, fp.face_size-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_patch(rng, dist);
    for (int j=0; j < fp.npatches; j++)
    {
      cv::Rect bbox = cv::Rect(annotations[i].bbox.x+rand_patch(), annotations[i].bbox.y+rand_patch(), patch_size, patch_size);
      int label = annotations[i].pose + 2;
      HeadPoseSample *hps = new HeadPoseSample(sample, annotations[i].bbox, bbox, label);
      hp_samples.push_back(hps);
      hps->show();
    }

    /*continue;

    // Extract negative patches
    boost::uniform_int<> dist_neg_x(0, img_rescaled.cols-patch_size-1);
    boost::uniform_int<> dist_neg_y(0, img_rescaled.rows-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_x(rng, dist_neg_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_neg_y(rng, dist_neg_y);
    for (int j=0; j < forest_param.nPatchesPerSample; j++)
    {
      cv::Rect bbox;
      int save = 0;
      while (save < 1000)
      {
        bbox = cv::Rect(rand_neg_x(), rand_neg_y(), patch_size, patch_size);
        cv::Rect inter = intersect(bbox, annotations[i].bbox);

        if (inter.height == 0 and inter.width == 0)
          break;
        save++;
      }
      HeadPoseSample *s = new HeadPoseSample(sample, annotations[i].bbox, bbox, -1);
      hp_samples.push_back(s);
      s->show();
    }*/
  }
  PRINT("Used patches: " << hp_samples.size());

  /*if (unfinished_tree)
    tree->grow(hp_samples, job_timer, &rng);
  else
    tree = new Tree<HeadPoseSample>(hp_samples, fp, &rng, path);*/

  return EXIT_SUCCESS;
};
