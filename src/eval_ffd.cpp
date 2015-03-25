/** ****************************************************************************
 *  @file    eval_ffd.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/08
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Viewer.hpp>
#include <FaceForest.hpp>
#include <face_utils.hpp>
#include <fstream>
#include <boost/progress.hpp>
#include <opencv2/opencv.hpp>

// The distance between the eyes
float
getInterOccularDist
  (
  const FaceAnnotation &annotation
  )
{
  cv::Point2f center_left, center_right;
  center_left.x  = (annotation.parts[0].x+annotation.parts[2].x)/2;
  center_left.y  = (annotation.parts[0].y+annotation.parts[2].y)/2;
  center_right.x = (annotation.parts[6].x+annotation.parts[7].x)/2;
  center_right.y = (annotation.parts[6].y+annotation.parts[7].y)/2;

  return cv::norm(center_left-center_right);
};

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
  std::vector< std::vector<float> > errors;
  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); ++i, ++show_progress)
  {
    TRACE("Evaluate image: " << annotations[i].url);

    // Load image
    std::string dir = ff_options.mp_forest_param.image_path;
    std::size_t pos = dir.rfind("/")+1;
    std::string img_path = dir.substr(0,pos) + annotations[i].url;
    cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    std::vector<Face> faces;
    Face face;
    ff.analyzeFace(img, annotations[i].bbox, face);
    faces.push_back(face);

    // Draw results
    viewer.resizeCanvas(img.cols, img.rows);
    viewer.beginDrawing();
    viewer.image(img, 0, 0, img.cols, img.rows);
    ff.showResults(faces, viewer);
    viewer.endDrawing(0);

    std::vector<float> err;
    float inter_occular_dist = getInterOccularDist(annotations[i]);
    for (int j=0; j < face.ffd_cordinates.size(); j++)
    {
      float d = cv::norm(annotations[i].parts[j]-face.ffd_cordinates[j]);
      err.push_back(d / inter_occular_dist);
    }
    errors.push_back(err);
  }

  // Write errors on a file
  std::ofstream ofs;
  std::string path = "output/errors.txt";
  ofs.open(path.c_str(), std::ios::out);
  for (int i=0; i < errors.size(); i++)
  {
    for(int j=0; j < errors[i].size(); j++)
      ofs << errors[i][j] << " ";
    ofs << std::endl;
  }
};

int
main
  (
  int argc,
  char **argv
  )
{
  // Evaluate feature points detector
  std::string ffd_config_file = "data/config_ffd.txt";
  std::string headpose_config_file = "data/config_headpose.txt";
  std::string face_cascade = "data/haarcascade_frontalface_alt.xml";

  // Parse configuration files
  ForestParam hp_param, mp_param;
  if (!loadConfigFile(headpose_config_file, hp_param))
    return EXIT_FAILURE;

  if (!loadConfigFile(ffd_config_file, mp_param))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(mp_param.image_path, annotations))
    return EXIT_FAILURE;

  FaceForestOptions ff_options;
  ff_options.fd_option.path_face_cascade = face_cascade;
  ff_options.hp_forest_param = hp_param;
  ff_options.mp_forest_param = mp_param;

  evalForest(ff_options, annotations);

  return EXIT_SUCCESS;
};
