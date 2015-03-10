/** ****************************************************************************
 *  @file    SplitGen.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/08
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef SPLIT_GEN_HPP
#define SPLIT_GEN_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ThreadPool.hpp>
#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/numeric/conversion/bounds.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>

struct ForestParam
{
  int max_d;
  int min_s;
  int nTests;
  int nTrees;
  int nSamplesPerTree;
  int nPatchesPerSample;
  int faceSize;
  int measuremode;
  int nFeatureChannels;
  float patchSizeRatio;
  std::string treePath;
  std::string imgPath;
  std::string featurePath;
  std::vector<int> features;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & max_d;
    ar & min_s;
    ar & nTests;
    ar & nTrees;
    ar & nSamplesPerTree;
    ar & nPatchesPerSample;
    ar & faceSize;
    ar & measuremode;
    ar & nFeatureChannels;
    ar & patchSizeRatio;
    ar & treePath;
    ar & imgPath;
    ar & featurePath;
    ar & features;
  }
};

typedef std::pair<int, unsigned int> IntIndex;

struct less_than
{
  bool operator()(const IntIndex &a, const IntIndex &b) const
  {
    return a.first < b.first;
  };

  bool operator()(const IntIndex &a, const int &b) const {
    return a.first < b;
  };
};

/** ****************************************************************************
 * @class SplitGen
 * @brief Conditional regression split generator
 ******************************************************************************/
template<typename Sample>
class SplitGen
{
public:
  typedef typename Sample::Split Split;

  // Called from "findOptimalSplit"
  SplitGen
    (
    const std::vector<Sample*> &samples,
    std::vector<Split> &splits,
    boost::mt19937 *rng,
    ForestParam fp,
    int depth,
    float split_mode
    ) :
      m_samples(samples), m_splits(splits), m_rng(rng), m_fp(fp),
      m_depth(depth), m_split_mode(split_mode) {};

  virtual
  ~SplitGen
    () {};

  // Called from Tree "findOptimalSplit"
  void
  generate
    ()
  {
    int num_treads = boost::thread::hardware_concurrency();
    boost::thread_pool::ThreadPool e(num_treads);
    for (int stripe=0; stripe < static_cast<int>(m_splits.size()); stripe++)
      e.submit(boost::bind(&SplitGen::generateMT, this, stripe));
    e.join_all();
  };

  void
  generateMT
    (
    int stripe
    )
  {
    // Randomly estimate the best pair of sub-patches
    boost::mt19937 rng(abs(stripe+1) * std::time(NULL));
    if (Sample::generateSplit(m_samples, &rng, m_fp, m_splits[stripe]))
    {
      // Process each patch with the selected R1 and R2
      std::vector<IntIndex> val_set(m_samples.size());
      for (unsigned int i=0; i < m_samples.size(); ++i)
      {
        val_set[i].first  = m_samples[i]->evalTest(m_splits[stripe]);
        val_set[i].second = i;
      }
      std::sort(val_set.begin(), val_set.end()); // sort by f_theta
      findThreshold(m_samples, val_set, m_splits[stripe], &rng);
      m_splits[stripe].oob = 0;
    }
    else
    {
      m_splits[stripe].threshold = 0;
      m_splits[stripe].info = boost::numeric::bounds<double>::lowest();
      m_splits[stripe].gain = boost::numeric::bounds<double>::lowest();
      m_splits[stripe].oob  = boost::numeric::bounds<double>::highest();
    }
  };

  // Called from Tree "split"
  static void splitVec(const std::vector<Sample*>& data,
      const std::vector<IntIndex>& valSet, std::vector<Sample*>& setA,
      std::vector<Sample*>& setB, int threshold, int margin = 0) {

    // search largest value such that val<t
    std::vector<IntIndex>::const_iterator it_first, it_second;

    it_first = lower_bound(valSet.begin(), valSet.end(), threshold - margin, less_than());
    if (margin == 0)
      it_second = it_first;
    else
      it_second = lower_bound(valSet.begin(), valSet.end(), threshold + margin, less_than());

    // Split training data into two sets A,B accroding to threshold t
    setA.resize(it_second - valSet.begin());
    setB.resize(valSet.end() - it_first);

    std::vector<IntIndex>::const_iterator it = valSet.begin();
    typename std::vector<Sample*>::iterator itSample;
    for (itSample = setA.begin(); itSample < setA.end(); ++itSample, ++it)
      (*itSample) = data[it->second];

    it = it_first;
    for (itSample = setB.begin(); itSample < setB.end(); ++itSample, ++it)
      (*itSample) = data[it->second];

  };

private:
  // Called from "generate_mt"
  void
  findThreshold
    (
    const std::vector<Sample*> &samples,
    const std::vector<IntIndex> &val_set,
    Split &split,
    boost::mt19937 *rng
    ) const
  {
    split.info = boost::numeric::bounds<double>::lowest();
    split.gain = boost::numeric::bounds<double>::lowest();

    int min_val = val_set.front().first;
    int max_val = val_set.back().first;
    int range   = max_val - min_val;

    if (range > 0)
    {
      // Find best threshold
      int nthresholds = split.num_thresholds; // 25
      bool use_margin = false;
      if (use_margin)
        nthresholds = 20;

      boost::uniform_int<> dist_thr(0, range);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_thr(*rng, dist_thr);

      // Only if use_margin == true
      int m = std::min(abs(min_val), abs(max_val));
      m = (m > 0) ? m : 1;
      boost::uniform_int<> dist_mar(0, m);
      boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_mar(*m_rng, dist_mar);

      for (int i=0; i < nthresholds; ++i)
      {
        // Generate some random thresholds
        int thresh = rand_thr() + min_val;
        int margin = 0;

        if (use_margin)
          margin = rand_mar();

        std::vector< std::vector<Sample*> > sets;
        splitSamples(samples, val_set, sets, thresh, margin);

        // Each set must have more than 1 sample
        unsigned int min_set_size = 2;
        if (sets[0].size() < min_set_size || sets[1].size() < min_set_size)
          continue;

        // Evaluate split using information gain IG
        double info = Sample::evalSplit(sets[0], sets[1], m_split_mode, m_depth);

        if (info > split.info)
        {
          split.threshold = thresh;
          split.info = info;
          split.gain = info;
          split.margin = margin;
        }
      }
    }
  };

  // Called from "findThreshold"
  static void
  splitSamples
    (
    const std::vector<Sample*> &samples,
    const std::vector<IntIndex> &val_set,
    std::vector< std::vector<Sample*> > &sets,
    int thresh,
    int margin
    )
  {
    // Search largest value such that value < t
    std::vector<IntIndex>::const_iterator it_first, it_second;
    it_first = lower_bound(val_set.begin(), val_set.end(), thresh-margin, less_than());
    if (margin == 0)
      it_second = it_first;
    else
      it_second = lower_bound(val_set.begin(), val_set.end(), thresh+margin, less_than());

    // Split training samples into two different sets A, B according to threshold t
    if (it_first == it_second)
    {
      // No intersection between the two thresholds
      sets.resize(2);
      sets[0].resize(it_first - val_set.begin());
      sets[1].resize(samples.size() - sets[0].size());

      std::vector<IntIndex>::const_iterator it;
      typename std::vector<Sample*>::iterator it_sample;

      it = val_set.begin();
      for (it_sample = sets[0].begin(); it_sample < sets[0].end(); ++it_sample, ++it)
        (*it_sample) = samples[it->second];

      it = val_set.begin() + sets[0].size();
      for (it_sample = sets[1].begin(); it_sample < sets[1].end(); ++it_sample, ++it)
        (*it_sample) = samples[it->second];

      CV_Assert((sets[0].size() + sets[1].size()) == samples.size());
    }
    else
    {
      sets.resize(3);
      sets[0].resize(it_first - val_set.begin());
      sets[1].resize(it_second - it_first);
      sets[2].resize(val_set.end() - it_second);

      std::vector<IntIndex>::const_iterator it;
      typename std::vector<Sample*>::iterator it_sample;

      it = val_set.begin();
      for (it_sample = sets[0].begin(); it_sample < sets[0].end(); ++it_sample, ++it)
        (*it_sample) = samples[it->second];

      it = val_set.begin() + sets[0].size();
      for (it_sample = sets[1].begin(); it_sample < sets[1].end(); ++it_sample, ++it)
        (*it_sample) = samples[it->second];

      it = val_set.begin() + sets[0].size() + sets[1].size();
      for (it_sample = sets[2].begin(); it_sample < sets[2].end(); ++it_sample, ++it)
        (*it_sample) = samples[it->second];

      CV_Assert((sets[0].size() + sets[1].size() + sets[2].size()) == samples.size());
    }
  };

  const std::vector<Sample*> &m_samples;
  std::vector<Split> &m_splits;
  boost::mt19937 *m_rng;
  ForestParam m_fp;
  float m_depth;
  float m_split_mode;
};

#endif /* SPLIT_GEN_HPP */
