/** ****************************************************************************
 *  @file    demo.cpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/08
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Viewer.hpp>
#include <SplitGen.hpp>
#include <FaceForest.hpp>
#include <face_utils.hpp>
#include <vector>
#include <cstdlib>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/progress.hpp>
#include <boost/lexical_cast.hpp>

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

  for (int i=0; i < static_cast<int>(annotations.size()); ++i)
  {
    PRINT("Evaluate image: " << annotations[i].url);

    // Load image
    cv::Mat image = cv::imread(annotations[i].url, cv::IMREAD_COLOR);
    if (image.data == NULL)
    {
      ERROR("Could not load: " << annotations[i].url);
      continue;
    }

    // Convert image to gray scale
    cv::Mat img_gray;
    cv::cvtColor(image, img_gray, cv::COLOR_BGR2GRAY);

    std::vector<Face> faces;
    const bool use_predefined_bbox = false;
    if (use_predefined_bbox)
    {
      Face face;
      ff.analyzeFace(img_gray, annotations[i].bbox, face);
      faces.push_back(face);
    }
    else
    {
      ff.analyzeImage(img_gray, faces);
    }

    // Draw results
    viewer.resizeCanvas(image.cols, image.rows);
    viewer.beginDrawing();
    viewer.image(image, 0, 0, image.cols, image.rows);
    ff.showResults(img_gray, faces, viewer);
    viewer.endDrawing(0);
  }
};

int
main
  (
  int argc,
  char **argv
  )
{
  // Usage: 1 config_ffd.txt config_headpose.txt haarcascade_frontalface.xml
  if (argc < 3)
  {
    PRINT("Usage: mode ffd_config headpose_config face_xml");
    PRINT("Using default parameters ...");
  }

  // Mode 0: training forest
  // Mode 1: evaluate point detector
  int mode = 1;
  std::string ffd_config_file = "data/config_ffd.txt";
  std::string headpose_config_file = "data/config_headpose.txt";
  std::string face_cascade = "data/haarcascade_frontalface_alt.xml";
  if (argc > 3)
  {
    try
    {
      mode = boost::lexical_cast<int>(argv[1]);
      ffd_config_file = argv[2];
      headpose_config_file = argv[3];
      face_cascade = argv[4];
    }
    catch (char *err)
    {
      ERROR("Error during flag parsing: " << err);
      return EXIT_FAILURE;
    }
  }

  // Parse configuration file
  ForestParam mp_param;
  if (!loadConfigFile(ffd_config_file, mp_param))
    return EXIT_FAILURE;

  // Loading images annotations
  std::vector<FaceAnnotation> annotations;
  if (!loadAnnotations(mp_param.image_path, annotations))
    return EXIT_FAILURE;

  switch (mode)
  {
    case 0:
    {
      //trainForest(mp_param, annotations);
      break;
    }
    case 1:
    {
      ForestParam hp_param;
      if (!loadConfigFile(headpose_config_file, hp_param))
        return EXIT_FAILURE;

      FaceForestOptions ff_options;
      ff_options.fd_option.path_face_cascade = face_cascade;
      ff_options.hp_forest_param = hp_param;
      ff_options.mp_forest_param = mp_param;

      evalForest(ff_options, annotations);
      break;
    }
    default:
      PRINT("Unknown mode (0=training, 1=evaluate)");
  }

  return EXIT_SUCCESS;
};
