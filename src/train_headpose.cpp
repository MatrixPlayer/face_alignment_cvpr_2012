/** ****************************************************************************
 *  @file    train_headpose.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/05
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <SplitGen.hpp>
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
  ForestParam forest_param;
  if (!loadConfigFile(config_file, forest_param))
    return EXIT_FAILURE;

  // Try to read the head-pose conditional regression tree
  char path[200];
  sprintf(path, "%s%03d.txt", forest_param.treePath.c_str(), offset);
  PRINT("Read head-pose regression tree: " << path);
  Tree<HeadPoseSample> *tree;
  Tree<HeadPoseSample>::load(&tree, path);

  // Initialize random generator
  boost::mt19937 rng;
  rng.seed(offset+1);
  srand(offset+1);

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(forest_param.imgPath, annotations))
    return EXIT_FAILURE;

  std::random_shuffle(annotations.begin(), annotations.end());

  // Separate annotations by head-pose classes
  std::vector<std::vector<FaceAnnotation> > cluster(5);
  for (unsigned int i=0; i < annotations.size(); i++)
  {
    // Range is between -2 and +2 but we shift it to 0 - 5
    int label = annotations[i].pose + 2;
    cluster[label].push_back(annotations[i]);
  }

  // Estimate the number of samples for each class
  int sample_per_class = static_cast<int>(cluster[0].size());
  for (unsigned int i=0; i < cluster.size(); i++)
    sample_per_class = std::min(sample_per_class, static_cast<int>(cluster[i].size()));
  sample_per_class = std::min(sample_per_class, forest_param.nSamplesPerTree);


  annotations.clear();
  for (unsigned int i=0; i < cluster.size(); i++)
    for (int j=0; j < sample_per_class; j++)
      annotations.push_back(cluster[i][j]);
  PRINT("Elements per class: " << sample_per_class);
  PRINT("Number of head poses: " << annotations.size());
  PRINT("Reserved patches: " << sample_per_class*(cluster.size()+1)*forest_param.nPatchesPerSample);

  // 5 positive classes + 1 negative class
  std::vector<HeadPoseSample*> hp_samples;
  hp_samples.reserve(sample_per_class*(cluster.size())*forest_param.nPatchesPerSample);

  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); i++, ++show_progress)
  {
    TRACE("Evaluate image: " << annotations[i].url);

    // Load image
    cv::Mat image = cv::imread(annotations[i].url, cv::IMREAD_COLOR);
    if (image.data == NULL)
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Rescale image and annotations
    cv::Mat img_rescaled;
    float scale = static_cast<float>(forest_param.faceSize)/static_cast<float>(annotations[i].bbox.width);
    cv::resize(image, img_rescaled, cv::Size(image.cols*scale, image.rows*scale), 0, 0);
    annotations[i].bbox.x *= scale;
    annotations[i].bbox.y *= scale;
    annotations[i].bbox.width *= scale;
    annotations[i].bbox.height *= scale;
    for (unsigned int j=0; j < annotations[i].parts.size(); j++)
      annotations[i].parts[j] *= scale;

    // Extract patches
    ImageSample *sample = new ImageSample(img_rescaled, forest_param.features, false);
    int patch_size = static_cast<int>(floor(forest_param.faceSize*forest_param.patchSizeRatio));

    // Positive samples
    boost::uniform_int<> dist(0, forest_param.faceSize-patch_size-1);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_patch(rng, dist);
    for (int j=0; j < forest_param.nPatchesPerSample; j++)
    {
      cv::Rect bbox = cv::Rect(annotations[i].bbox.x+rand_patch(), annotations[i].bbox.y+rand_patch(), patch_size, patch_size);
      int headpose  = annotations[i].pose + 2;
      HeadPoseSample *s = new HeadPoseSample(sample, annotations[i].bbox, bbox, headpose);
      hp_samples.push_back(s);
      s->show();
    }

    /*continue;

    // Negative samples
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
  else*/
    tree = new Tree<HeadPoseSample>(hp_samples, forest_param, &rng, path);

  return EXIT_SUCCESS;
};
