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

void
trainTree
  (
  ForestParam mp_param,
  std::vector<FaceAnnotation> &annotations,
  unsigned idx_forest,
  int idx_tree
  )
{
  srand(idx_tree+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  // Try to read the facial feature conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s/forest_%d/tree_%03d.txt", mp_param.tree_path.c_str(), idx_forest, idx_tree);
  PRINT("Read facial feature regression tree: " << tree_path);
  Tree<MPSample> *tree;
  bool is_tree_load = Tree<MPSample>::load(&tree, tree_path);

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

    // Scale image and annotations (125 x 125)
    cv::Mat img_scaled = scale(img_gray, mp_param.face_size, annotations[i]);

    // Enlarge image to make sure that all facial features are enclosed (149 x 149)
    cv::Mat img_enlarged = enlarge(img_scaled, annotations[i]);

    // Normalize histogram
    cv::Mat img_face;
    cv::equalizeHist(img_enlarged, img_face);

    // Create image sample
    ImageSample *sample = new ImageSample(img_face, mp_param.features, false);

    // Extract positive patches
    int patch_size = mp_param.patchSize();
    boost::uniform_int<> dist_x(1, img_face.cols-patch_size-2);
    boost::uniform_int<> dist_y(1, img_face.rows-patch_size-2);
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

  // Separate annotations by head-pose classes
  std::vector< std::vector<FaceAnnotation> > cluster(NUM_HEADPOSE_CLASSES);
  for (unsigned int i=0; i < annotations.size(); i++)
  {
    // Range is between -2 and +2 but we shift it to 0 - 5
    int label = annotations[i].pose + 2;
    cluster[label].push_back(annotations[i]);
  }

  // Train facial-feature tree
  int idx_forest = atoi(argv[1]);
  int idx_tree = atoi(argv[2]);
  trainTree(mp_param, cluster[idx_forest], idx_forest, idx_tree);

  // Train facial-feature forests
  //for (unsigned int i=0; i < NUM_HEADPOSE_CLASSES; i++)
  //  for (int j=0; j < mp_param.ntrees; j++)
  //    trainTree(mp_param, cluster[i], i, j);

  return EXIT_SUCCESS;
};
