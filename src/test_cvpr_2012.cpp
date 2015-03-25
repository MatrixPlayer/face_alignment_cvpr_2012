/** ****************************************************************************
 *  @file    test_cvpr_2012.cpp
 *  @brief   Real-time facial pose and feature detection
 *  @author  Roberto Valle Fernandez
 *  @date    2015/02
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <Viewer.hpp>
#include <TreeNode.hpp>
#include <FaceForest.hpp>
#include <face_utils.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>

const double IMG_INPUT_WIDTH  = 640.0;
const double IMG_INPUT_HEIGHT = 480.0;

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
double
processFrame
  (
  cv::Mat frame,
  FaceForest &ff,
  std::vector<Face> &faces
  )
{
  double ticks = static_cast<double>(cv::getTickCount());
  ff.analyzeImage(frame, faces);
  ticks = static_cast<double>(cv::getTickCount()) - ticks;
  return ticks;
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
void
showResults
  (
  cv::Mat frame,
  FaceForest &ff,
  std::vector<Face> &faces,
  upm::Viewer &viewer,
  double ticks,
  int delay
  )
{
  std::ostringstream outs;
  outs << "FPS =" << std::setprecision(3);
  outs << static_cast<double>(cv::getTickFrequency())/ticks << std::ends;

  // Drawing results
  viewer.resizeCanvas(frame.cols, frame.rows);
  viewer.beginDrawing();
  viewer.image(frame, 0, 0, frame.cols, frame.rows);
  ff.showResults(faces, viewer);
  viewer.text(outs.str(), 20, frame.rows-20, cv::Scalar(255,0,255), 0.5);
  viewer.endDrawing(delay);
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
  // ---------------------------------------------------------------------------
  // Determine if we get the images from a camera or a video
  // ---------------------------------------------------------------------------
  cv::Mat frame;
  cv::VideoCapture capture;
  bool process_image_file    = false;
  bool process_video_capture = false;

  if (argc == 1)
  {
    ERROR("Usage: image | video input file required");
    return EXIT_FAILURE;
  }

  boost::filesystem::path dir(argv[1]);
  if (boost::filesystem::exists(dir) && (boost::filesystem::is_regular_file(dir)))
  {
    frame = cv::imread(dir.c_str(), cv::IMREAD_COLOR);
    if (!frame.empty()) // Trying image file ...
    {
      PRINT("Processing an image file ...");
      process_image_file = true;
    }
    else // Trying video file ...
    {
      PRINT("Capturing from AVI file ...");
      capture.open(dir.string());
      if (!capture.isOpened())
      {
        ERROR("Could not grab images from AVI file");
        return EXIT_FAILURE;
      }
      process_video_capture = true;
    }
  }
  else
  {
    PRINT("Capturing from camera ...");
    capture.open(0);
    capture.set(CV_CAP_PROP_FRAME_WIDTH, IMG_INPUT_WIDTH);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, IMG_INPUT_HEIGHT);
    if (!capture.isOpened())
    {
      ERROR("Could not grab images from camera");
      return EXIT_FAILURE;
    }
    process_video_capture = true;
  }

  // Initialize face forest
  ForestParam hp_param, mp_param;
  CV_Assert(loadConfigFile("data/config_headpose.txt", hp_param));
  CV_Assert(loadConfigFile("data/config_ffd.txt", mp_param));

  FaceForestOptions ff_options;
  ff_options.fd_option.path_face_cascade = "data/haarcascade_frontalface_alt.xml";
  ff_options.hp_forest_param = hp_param;
  ff_options.mp_forest_param = mp_param;

  FaceForest ff(ff_options);

  upm::Viewer viewer;
  viewer.init(frame.cols, frame.rows, "cvpr_2012");

  if (process_image_file)
  {
    std::vector<Face> faces;
    double ticks = processFrame(frame, ff, faces);
    showResults(frame, ff, faces, viewer, ticks, 0);
  }
  else
  {
    for (;;)
    {
      if (!capture.grab())
        break;

      capture.retrieve(frame);

      if (frame.empty())
        break;

      std::vector<Face> faces;
      double ticks = processFrame(frame, ff, faces);
      showResults(frame, ff, faces, viewer, ticks, 1);
    }
  }

  return EXIT_SUCCESS;
};
