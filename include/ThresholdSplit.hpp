/** ****************************************************************************
 *  @file    ThresholdSplit.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef THRESHOLD_SPLIT_HPP
#define THRESHOLD_SPLIT_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <boost/serialization/access.hpp>
#include <boost/random/mersenne_twister.hpp>

/** ****************************************************************************
 * @class ThresholdSplit
 * @brief Split data related to a tree node
 ******************************************************************************/
template<typename Feature>
class ThresholdSplit
{
public:
  ThresholdSplit
    ()
  {
    margin = 0;
  };

  /*void
  generate
    (
    int patch_size,
    boost::mt19937 *rng,
    int num_feature_channels = 0
    )
  {
    feature.generate(patch_size, rng, num_feature_channels);
    margin = 0;
    num_thresholds = 25;
  };

  void
  print()
  {
    feature.print();
    PRINT(" " << threshold);
  };*/

  Feature feature;
  double info;
  double oob;
  int threshold;
  int margin;
  int depth;
  int num_thresholds;
  float split_mode;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & feature;
    ar & info;
    ar & threshold;
  }
};

#endif /* THRESHOLD_SPLIT_HPP */
