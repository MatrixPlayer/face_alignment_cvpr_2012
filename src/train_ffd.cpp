/** ****************************************************************************
 *  @file    train_ffd.cpp
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
#include <cstdlib>
#include <boost/progress.hpp>
#include <boost/random/mersenne_twister.hpp>

void
trainFeaturesForest
  (
  ForestParam &fp,
  std::vector<FaceAnnotation> &annotations,
  unsigned forest,
  int offset
  )
{
  PRINT("Input offset: " << offset);
  srand(offset+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  // Try to read the facial-feature conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s%d/tree%03d.txt", fp.tree_path.c_str(), forest, offset);
  PRINT("Read facial-feature regression tree: " << tree_path);
  Tree<MPSample> *tree;
  bool is_tree_load = Tree<MPSample>::load(&tree, tree_path);

  std::vector<MPSample*> mp_samples;
  mp_samples.reserve(fp.nimages*fp.npatches);
  PRINT("Total number of images: " << fp.nimages);
  PRINT("Reserved patches: " << fp.nimages*fp.npatches);

  boost::mt19937 rng;
  rng.seed(offset+1);
  boost::progress_display show_progress(fp.nimages);
  for (int i=0; i < fp.nimages; i++, ++show_progress)
  {
    TRACE("Evaluate image: " << annotations[i].url);

    // Load image
    std::size_t pos = fp.image_path.rfind("/")+1;
    std::string img_path = fp.image_path.substr(0,pos) + annotations[i].url;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert to gray-scale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // Scale image and annotations
    cv::Mat img_scaled;
    float scale = static_cast<float>(fp.face_size)/static_cast<float>(annotations[i].bbox.width);
    cv::resize(img_gray, img_scaled, cv::Size(img_gray.cols*scale, img_gray.rows*scale), 0, 0);
    annotations[i].bbox.x *= scale;
    annotations[i].bbox.y *= scale;
    annotations[i].bbox.width *= scale;
    annotations[i].bbox.height *= scale;
    for (unsigned int j=0; j < annotations[i].parts.size(); j++)
      annotations[i].parts[j] *= scale;

    // Extract face
    int offset_y = annotations[i].bbox.width * 0.1;
    cv::Mat face;
    extractFace(img_scaled, annotations[i], face, 0, offset_y);

    // Equalize the histogram
    cv::Mat img_equalized;
    cv::equalizeHist(face, img_equalized);

    // Extract features from an image
    ImageSample* sample = new ImageSample(img_equalized, fp.features, false);

    // Extract positive patches
    int patch_size = fp.patchSize();
    boost::uniform_int<> dist_x(1, face.cols-patch_size-2);
    boost::uniform_int<> dist_y(1, face.rows-patch_size-2);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
    for (int j=0; j < fp.npatches; j++)
    {
      cv::Rect bbox = cv::Rect(rand_x(), rand_y(), patch_size, patch_size);
      MPSample *mps = new MPSample(sample, bbox, cv::Rect(0,0,face.cols,face.rows), annotations[i].parts, fp.face_size, true);
      mp_samples.push_back(mps);
      mps->show();
    }
  }
  PRINT("Used patches: " << mp_samples.size());

  /*if (is_tree_load && !tree->isFinished())
    tree->updateTree(mp_samples, &rng);
  else
    tree = new Tree<HeadPoseSample>(mp_samples, fp, &rng, tree_path);*/
};

int
main
  (
  int argc,
  char **argv
  )
{
  // Usage: data/config_ffd.txt
  std::string config_file = argv[1];

  // Parse configuration file
  ForestParam fp;
  if (!loadConfigFile(config_file, fp))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(fp.image_path, annotations))
    return EXIT_FAILURE;

  // Train facial-feature forests
  //for (unsigned i=0; i < Constants::NUM_HP_CLASSES; i++)
  //  for (int j=0; j < fp.ntrees; j++)
      trainFeaturesForest(fp, annotations, 0, 0);

  return EXIT_SUCCESS;
};
