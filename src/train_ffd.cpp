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
#include <MPSample.hpp>

#include <vector>
#include <string>
#include <cstdlib>
#include <boost/progress.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <opencv2/highgui/highgui.hpp>

void
trainFacialFeaturesTree
  (
  ForestParam &fp,
  std::vector<FaceAnnotation> &annotations,
  unsigned idx_forest,
  int idx_tree
  )
{
  srand(idx_tree+1);
  std::random_shuffle(annotations.begin(), annotations.end());

  // Try to read the facial-feature conditional regression tree
  char tree_path[200];
  sprintf(tree_path, "%s/forest_%d/tree_%03d.txt", fp.tree_path.c_str(), idx_forest, idx_tree);
  PRINT("Read facial-feature regression tree: " << tree_path);
  Tree<MPSample> *tree;
  bool is_tree_load = Tree<MPSample>::load(&tree, tree_path);

  std::vector<MPSample*> mp_samples;
  mp_samples.reserve(fp.nimages*fp.npatches);
  PRINT("Total number of images: " << fp.nimages);
  PRINT("Reserved patches: " << fp.nimages*fp.npatches);

  boost::mt19937 rng;
  rng.seed(idx_tree+1);
  boost::progress_display show_progress(fp.nimages);
  for (int i=0; i < fp.nimages; i++, ++show_progress)
  {
    TRACE("Evaluate image: " << annotations[i].url);

    // Load image
    std::string dir = fp.image_path;
    std::size_t pos = dir.rfind("/")+1;
    std::string img_path = dir.substr(0,pos) + annotations[i].url;
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

    // Enlarge by 20% to make sure that all facial features are enclosed
    int offset_x = annotations[i].bbox.width * 0.1;
    int offset_y = annotations[i].bbox.height * 0.1;
    cv::Rect big_bbox = cv::Rect(annotations[i].bbox.x-offset_x, annotations[i].bbox.y-offset_y, annotations[i].bbox.width+(offset_x*2), annotations[i].bbox.height+(offset_y*2));
    cv::Rect face_bbox = intersect(big_bbox, cv::Rect(0,0,img_scaled.cols,img_scaled.rows));
    cv::Mat img_enlarged = img_scaled(face_bbox);
    annotations[i].bbox.x = 0;
    annotations[i].bbox.y = 0;
    annotations[i].bbox.width = face_bbox.width;
    annotations[i].bbox.height = face_bbox.height;
    for (unsigned int j=0; j < annotations[i].parts.size(); j++)
    {
      annotations[i].parts[j].x += offset_x;
      annotations[i].parts[j].y += offset_y;
    }

    // Normalize histogram
    cv::Mat img_face;
    cv::equalizeHist(img_enlarged, img_face);

    // Create image sample
    ImageSample *sample = new ImageSample(img_face, fp.features, false);

    // Extract positive patches
    int patch_size = fp.patchSize();
    boost::uniform_int<> dist_x(1, img_face.cols-patch_size-2);
    boost::uniform_int<> dist_y(1, img_face.rows-patch_size-2);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_x(rng, dist_x);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_y(rng, dist_y);
    for (int j=0; j < fp.npatches; j++)
    {
      cv::Rect bbox = cv::Rect(rand_x(), rand_y(), patch_size, patch_size);
      MPSample *mps = new MPSample(sample, bbox, annotations[i].parts, fp.face_size, true);
      mp_samples.push_back(mps);
      //mps->show();
    }
  }
  PRINT("Used patches: " << mp_samples.size());

  if (is_tree_load && !tree->isFinished())
    tree->updateTree(mp_samples, &rng);
  else
    tree = new Tree<MPSample>(mp_samples, fp, &rng, tree_path);
};

int
main
  (
  int argc,
  char **argv
  )
{
  // Parse configuration file
  std::string config_file = "data/config_ffd.txt";
  ForestParam fp;
  if (!loadConfigFile(config_file, fp))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(fp.image_path, annotations))
    return EXIT_FAILURE;

  // Separate annotations by head-pose classes
  std::vector< std::vector<FaceAnnotation> > ann_cluster(Constants::NUM_HP_CLASSES);
  for (unsigned i=0; i < annotations.size(); i++)
  {
    // Range is between -2 and +2 but we shift it to 0 - 5
    int label = annotations[i].pose + 2;
    ann_cluster[label].push_back(annotations[i]);
  }

  // Train facial-feature tree
  int idx_forest = atoi(argv[1]);
  int idx_tree = atoi(argv[2]);
  trainFacialFeaturesTree(fp, ann_cluster[idx_forest], idx_forest, idx_tree);

  // Train facial-feature forests
  //for (unsigned i=0; i < Constants::NUM_HP_CLASSES; i++)
  //  for (int j=0; j < fp.ntrees; j++)
  //    trainFacialFeaturesTree(fp, ann_cluster[i], i, j);

  return EXIT_SUCCESS;
};
