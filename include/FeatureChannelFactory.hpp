/** ****************************************************************************
 *  @file    FeatureChannelFactory.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/09
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FEATURE_CHANNEL_FACTORY_HPP
#define FEATURE_CHANNEL_FACTORY_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <ThreadPool.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/thread.hpp>

#define FC_GRAY     0
#define FC_GABOR    1
#define FC_SOBEL    2
#define FC_MIN_MAX  3
#define FC_CANNY    4
#define FC_NORM     5

/** ****************************************************************************
 * @class FeatureChannelFactory
 * @brief Feature channel extractor
 ******************************************************************************/
class FeatureChannelFactory
{
public:
  FeatureChannelFactory
    () {};

  void
  extractChannel
    (
    int feature,
    bool use_integral,
    const cv::Mat &img,
    std::vector<cv::Mat> &channels
    )
  {
    switch (feature)
    {
      case FC_GRAY:
      {
        if (use_integral)
        {
          cv::Mat integral_img;
          cv::integral(img, integral_img, CV_32F);
          channels.push_back(integral_img);
        }
        else
          channels.push_back(img);
        break;
      }
      case FC_NORM:
      {
        cv::Mat equalize_img;
        cv::equalizeHist(img, equalize_img);
        if (use_integral)
        {
          cv::Mat integral_img;
          cv::integral(equalize_img, integral_img, CV_32F);
          channels.push_back(integral_img);
        } else
          channels.push_back(equalize_img);
        break;
      }
      case FC_GABOR:
      {
        // Check if kernels are initialized
        if (reals.size() == 0)
          initGaborKernels();

        int old_size = channels.size();
        bool multithreaded = true;
        if (multithreaded)
        {
          channels.resize(channels.size() + reals.size());
          int num_treads = boost::thread::hardware_concurrency();
          boost::thread_pool::ThreadPool e(num_treads);
          for (unsigned int i=0; i < reals.size(); i++)
            e.submit(boost::bind(&FeatureChannelFactory::gaborTransform, this, img, &channels[old_size+i], use_integral, i, old_size));
          e.join_all();
        }
        else
        {
          for (unsigned int i=0; i < reals.size(); i++)
          {
            cv::Mat final;
            cv::Mat r_mat;
            cv::Mat i_mat;
            cv::filter2D(img, r_mat, CV_32F, reals[i]);
            cv::filter2D(img, i_mat, CV_32F, imags[i]);
            cv::pow(r_mat, 2, r_mat);
            cv::pow(i_mat, 2, i_mat);
            cv::add(i_mat, r_mat, final);
            cv::pow(final, 0.5, final);
            cv::normalize(final, final, 0, 1, cv::NORM_MINMAX, CV_32F);

            if (use_integral)
            {
              cv::Mat imga;
              final.convertTo(imga, CV_8UC1, 255);
              cv::Mat integral_img;
              cv::integral(imga, integral_img, CV_32F);
              channels.push_back(integral_img);
            }
            else
            {
              final.convertTo(final, CV_8UC1, 255);
              channels.push_back(final);
            }
          }
        }
        break;
      }
      case FC_SOBEL:
      {
        cv::Mat sob_x(img.size(), CV_8U);
        cv::Mat sob_y(img.size(), CV_8U);
        cv::Sobel(img, sob_x, CV_8U, 0, 1);
        cv::Sobel(img, sob_y, CV_8U, 1, 0);

        if (use_integral)
        {
          cv::Mat sob_x_int, sob_y_int;
          cv::integral(sob_x, sob_x_int, CV_32F);
          cv::integral(sob_y, sob_y_int, CV_32F);
          channels.push_back(sob_x_int);
          channels.push_back(sob_y_int);
        }
        else
        {
          channels.push_back(sob_x);
          channels.push_back(sob_y);
        }
        break;
      }
      case FC_MIN_MAX:
      {
        cv::Mat kernel(cv::Size(3, 3), CV_8UC1);
        kernel.setTo(cv::Scalar(1));
        cv::Mat img_min(img.size(), CV_8U);
        cv::Mat img_max(img.size(), CV_8U);
        cv::erode(img, img_min, kernel);
        cv::dilate(img, img_max, kernel);

        if (use_integral)
        {
          cv::Mat img_min_int, img_max_int;
          cv::integral(img_min, img_min_int, CV_32F);
          cv::integral(img_max, img_max_int, CV_32F);
          channels.push_back(img_min_int);
          channels.push_back(img_max_int);
        }
        else
        {
          channels.push_back(img_min);
          channels.push_back(img_max);
        }
        break;
      }
      case FC_CANNY:
      {
        cv::Mat canny_img;
        cv::Canny(img, canny_img, -1, 5);
        if (use_integral)
        {
          cv::Mat integral_img;
          cv::integral(canny_img, integral_img, CV_32F);
          channels.push_back(integral_img);
        }
        else
          channels.push_back(canny_img);
        break;
      }
      default:
        ERROR("(!) Unknown feature channel to extract");
    }
  };

private:
  void
  createKernel
    (
    int iMu,
    int iNu,
    double sigma,
    double dF
    )
  {
    // Initialize the parameters
    double F = dF;
    double k = (CV_PI / 2) / pow(F, (double) iNu);
    double phi = CV_PI * iMu / 8;

    double width = round((sigma / k) * 6 + 1);
    if (fmod(width, 2.0) == 0.0)
      width++;

    // Create kernel
    cv::Mat real = cv::Mat(width, width, CV_32FC1);
    cv::Mat imag = cv::Mat(width, width, CV_32FC1);

    int x, y;
    double dReal;
    double dImag;
    double dTemp1, dTemp2, dTemp3;

    int off_set = (width - 1) / 2;
    for (int i = 0; i < width; i++)
    {
      for (int j = 0; j < width; j++)
      {
        x = i - off_set;
        y = j - off_set;
        dTemp1 = (pow(k, 2) / pow(sigma, 2)) * exp(-(pow((double) x, 2) + pow((double) y, 2)) * pow(k, 2) / (2 * pow(sigma, 2)));
        dTemp2 = cos(k * cos(phi) * x + k * sin(phi) * y) - exp(-(pow(sigma, 2) / 2));
        dTemp3 = sin(k * cos(phi) * x + k * sin(phi) * y);
        dReal = dTemp1 * dTemp2;
        dImag = dTemp1 * dTemp3;
        real.at<float>(j, i) = dReal;
        imag.at<float>(j, i) = dImag;
      }
    }

    reals.push_back(real);
    imags.push_back(imag);
  };

  void
  initGaborKernels()
  {
    // Create kernels
    int NuMin = 0;
    int NuMax = 4;
    int MuMin = 0;
    int MuMax = 7;
    double sigma = 1.0 / 2.0 * CV_PI;
    double dF = sqrt(2.0);

    int iMu = 0;
    int iNu = 0;

    for (iNu = NuMin; iNu <= NuMax; iNu++)
      for (iMu = MuMin; iMu < MuMax; iMu++)
        createKernel(iMu, iNu, sigma, dF);
  };

  void gaborTransform
    (
    const cv::Mat &src,
    cv::Mat *dst,
    bool useIntegral,
    int index,
    int old_size
    ) const
  {
    cv::Mat final;
    cv::Mat r_mat;
    cv::Mat i_mat;
    cv::filter2D(src, r_mat, CV_32F, reals[index]);
    cv::filter2D(src, i_mat, CV_32F, imags[index]);
    cv::pow(r_mat, 2, r_mat);
    cv::pow(i_mat, 2, i_mat);
    cv::add(i_mat, r_mat, final);
    cv::pow(final, 0.5, final);
    cv::normalize(final, final, 0, 1, cv::NORM_MINMAX);

    if (useIntegral)
    {
      cv::Mat img;
      final.convertTo(img, CV_8UC1, 255);
      cv::Mat integral_img;
      cv::integral(img, integral_img, CV_32F);
      *dst = integral_img;
    }
    else
    {
      final.convertTo(final, CV_8UC1, 255);
      *dst = final;
    }
  };

  // Gabor kernels parameters
  std::vector<cv::Mat> reals; // real
  std::vector<cv::Mat> imags; // imaginary
};

#endif /* FEATURECHANNELEXTRACTOR_H_ */
