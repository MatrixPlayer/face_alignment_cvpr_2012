/** ****************************************************************************
 *  @file    MeanShift.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2012/07
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef MEAN_SHIFT_HPP
#define MEAN_SHIFT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <opencv2/core/core.hpp>

struct MeanShiftOption
{
  MeanShiftOption
    () :
      kernel_size(10), max_iterations(7), stopping_criteria(0.05) {};

  int kernel_size;
  int max_iterations;
  float stopping_criteria;
};

/** ****************************************************************************
 * @class MeanShift
 * @brief Technique for locating the maximum of a density function
 ******************************************************************************/
class MeanShift
{
public:
  MeanShift
    () {};

  virtual
  ~MeanShift
    () {};

  static void
  shift
    (
    const std::vector<Vote> &votes,
    cv::Point_<int> &result,
    MeanShiftOption &option
    )
  {
    shift(votes, result, option.max_iterations, option.kernel_size, option.stopping_criteria);
  };

  static void
  shift
    (
    const std::vector<Vote> &votes,
    cv::Point_<int> &result,
    int num_iterations,
    int kernel,
    float stopping_criteria
    )
  {
    bool coverg = false;
    cv::Point_<float> mean;
    getMean(votes, mean);

    for (int i=0; (i < num_iterations) && (coverg == false); i++)
    {
      cv::Point_<float> shifted_mean;
      getWeightedMean(votes, mean, kernel, shifted_mean);

      if (cv::norm(shifted_mean-mean) < stopping_criteria)
        coverg = true;
      mean = shifted_mean;
    }
    result = mean;
  };

private:
  static void
  getMean
    (
    const std::vector<Vote> &votes,
    cv::Point_<float> &mean
    )
  {
    mean = cv::Point(0.0, 0.0);
    float sum_w = 0;
    for (unsigned int i=0; i < votes.size(); i++)
    {
      if (!votes[i].check)
        continue;

      float w = votes[i].weight;
      mean.x += votes[i].pos.x * w;
      mean.y += votes[i].pos.y * w;
      sum_w += w;
    }

    if (sum_w > 0)
    {
      mean.x /= sum_w;
      mean.y /= sum_w;
    }
  };

  static void
  getWeightedMean
    (
    const std::vector<Vote> &votes,
    const cv::Point_<float> mean,
    float lamda,
    cv::Point_<float> &shifted_mean
    )
  {
    shifted_mean = cv::Point(0.0, 0.0);
    float sum_w = 0;
    for (unsigned int i=0; i < votes.size(); i++)
    {
      if (!votes[i].check)
        continue;

      float d = cv::norm(mean-cv::Point2f(votes[i].pos.x, votes[i].pos.y));
      d = expf(-d/lamda);
      float w = votes[i].weight * d;
      shifted_mean.x += votes[i].pos.x * w;
      shifted_mean.y += votes[i].pos.y * w;
      sum_w += w;
    }

    if (sum_w > 0)
    {
      shifted_mean.x /= sum_w;
      shifted_mean.y /= sum_w;
    }
  };
};

#endif /* MEAN_SHIFT_HPP */
