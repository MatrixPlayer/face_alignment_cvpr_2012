/** ****************************************************************************
 *  @file    eval_headpose.cpp
 *  @brief   Real-time facial pose and feature detection
 *  @author  Roberto Valle Fernandez
 *  @date    2015/02
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Constants.hpp>
#include <face_utils.hpp>
#include <Viewer.hpp>
#include <FaceForest.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <boost/progress.hpp>
#include <opencv2/opencv.hpp>

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
  for (int i=0; i < static_cast<int>(annotations.size()); i++, ++show_progress)
  {
    PRINT("Evaluate image: " << annotations[i].url);

    // Load image
    cv::Mat img = cv::imread(annotations[i].url, cv::IMREAD_COLOR);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert image to gray scale
    cv::Mat img_gray;
    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    std::vector<Face> faces;
    Face face;
    ff.analyzeFace(img_gray, annotations[i].bbox, face);
    faces.push_back(face);

    // Draw results
    viewer.resizeCanvas(img.cols, img.rows);
    viewer.beginDrawing();
    viewer.image(img, 0, 0, img.cols, img.rows);
    ff.showResults(faces, viewer);
    PRINT("Head-pose: predict=" << face.headpose << " real=" << annotations[i].pose);
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
  // Evaluate head-pose forest
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
