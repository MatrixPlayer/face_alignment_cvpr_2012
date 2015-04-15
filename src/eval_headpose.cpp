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
#include <Viewer.hpp>
#include <FaceForest.hpp>
#include <face_utils.hpp>

#include <vector>
#include <string>
#include <cstdlib>
#include <fstream>
#include <boost/progress.hpp>
#include <opencv2/opencv.hpp>

#define VIEWER

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
evalForest
  (
  FaceForestOptions ff_options,
  std::vector<FaceAnnotation> &annotations
  )
{
  // Initialize face forest
  FaceForest ff(ff_options);

  #ifdef VIEWER
  upm::Viewer viewer;
  viewer.init(0, 0, "eval_headpose");
  #endif
  boost::progress_display show_progress(annotations.size());
  for (int i=0; i < static_cast<int>(annotations.size()); i++, ++show_progress)
  {
    // Load image
    TRACE("Evaluate image: " << annotations[i].url);
    cv::Mat img = loadImage(ff_options.hp_forest_param.image_path, annotations[i].url);
    if (img.empty())
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    Face face;
    ff.analyzeFace(img, annotations[i].bbox, face);

    #ifdef VIEWER
    std::vector<Face> faces;
    faces.push_back(face);

    // Draw results
    viewer.resizeCanvas(img.cols, img.rows);
    viewer.beginDrawing();
    viewer.image(img, 0, 0, img.cols, img.rows);
    ff.showResults(faces, viewer);
    viewer.endDrawing(0);
    #endif

    PRINT("Real:" << annotations[i].pose << " Predict:" << face.headpose);
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
  if (!loadAnnotations(hp_param.image_path, annotations))
    return EXIT_FAILURE;

  FaceForestOptions ff_options;
  ff_options.fd_option.path_face_cascade = face_cascade;
  ff_options.hp_forest_param = hp_param;
  ff_options.mp_forest_param = mp_param;

  evalForest(ff_options, annotations);

  return EXIT_SUCCESS;
};
