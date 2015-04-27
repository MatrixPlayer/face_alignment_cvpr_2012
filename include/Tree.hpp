/** ****************************************************************************
 *  @file    Tree.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TREE_HPP
#define TREE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <trace.hpp>
#include <TreeNode.hpp>
#include <SplitGen.hpp>

#include <vector>
#include <string>
#include <fstream>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

/** ****************************************************************************
 * @class Tree
 * @brief Conditional regression tree
 ******************************************************************************/
template<typename Sample>
class Tree
{
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  Tree
    () {};

  Tree
    (
    const std::vector<Sample*> &samples,
    ForestParam param,
    boost::mt19937 *rng,
    std::string save_path
    )
  {
    m_rng = rng;
    m_save_path = save_path;
    m_param = param;
    m_num_nodes = powf(2.0f, m_param.max_depth) - 1;
    i_node = 0;
    i_leaf = 0;

    PRINT("Start training");
    m_ticks = static_cast<double>(cv::getTickCount());
    root = new TreeNode<Sample>(0);
    grow(root, samples);
    save();
  };

  virtual
  ~Tree
    ()
  {
    delete root;
  };

  bool
  isFinished
    ()
  {
    if (m_num_nodes == 0)
      return false;
    return i_node == m_num_nodes;
  };

  void
  update
    (
    const std::vector<Sample*> &samples,
    boost::mt19937* rng
    )
  {
    m_rng = rng;
    PRINT(100*static_cast<float>(i_node)/static_cast<float>(m_num_nodes) << "% : update tree");
    if (!isFinished())
    {
      i_node = 0;
      i_leaf = 0;

      PRINT("Start training");
      m_ticks = static_cast<double>(cv::getTickCount());
      grow(root, samples);
      save();
    }
  };

  void
  grow
    (
    TreeNode<Sample> *node,
    const std::vector<Sample*> &samples
    )
  {
    int depth = node->getDepth();
    int nelements = static_cast<int>(samples.size());
    if (nelements < m_param.min_patches || depth >= m_param.max_depth || node->isLeaf())
    {
      node->createLeaf(samples);
      i_node += powf(2.0f, m_param.max_depth-depth) - 1;
      i_leaf++;
      PRINT("  (1) " << 100*static_cast<float>(i_node)/static_cast<float>(m_num_nodes)
            << "% : make leaf(depth: " << depth << ", elements: " << samples.size()
            << ") [i_leaf: " << i_leaf << "]");
    }
    else
    {
      if (node->hasSplit()) // only in reload mode
      {
        Split best_split = node->getSplit();
        std::vector< std::vector<Sample*> > sets;
        applyOptimalSplit(samples, best_split, sets);
        i_node++;
        PRINT("  (2) " << 100*static_cast<float>(i_node)/static_cast<float>(m_num_nodes)
              << "% : split(depth: " << depth << ", elements: " << nelements << ") "
              << "[A: " << sets[0].size() << ", B: " << sets[1].size() << "]");

        grow(node->left, sets[0]);
        grow(node->right, sets[1]);
      }
      else
      {
        Split best_split;
        if (findOptimalSplit(samples, best_split, depth))
        {
          std::vector< std::vector<Sample*> > sets;
          applyOptimalSplit(samples, best_split, sets);
          node->setSplit(best_split);
          i_node++;

          TreeNode<Sample> *left = new TreeNode<Sample>(depth+1);
          node->addLeftChild(left);

          TreeNode<Sample> *right = new TreeNode<Sample>(depth+1);
          node->addRightChild(right);

          saveAuto();
          PRINT("  (3) " << 100*static_cast<float>(i_node)/static_cast<float>(m_num_nodes)
                << "% : split(depth: " << depth << ", elements: " << nelements  << ") "
                << "[A: " << sets[0].size() << ", B: " << sets[1].size() << "]");

          grow(left, sets[0]);
          grow(right, sets[1]);
        }
        else
        {
          PRINT("  No valid split found");
          node->createLeaf(samples);
          i_node += powf(2.0f, m_param.max_depth-depth) - 1;
          i_leaf++;
          PRINT("  (4) " << 100*static_cast<float>(i_node)/static_cast<float>(m_num_nodes)
                << "% : make leaf(depth: " << depth << ", elements: "  << samples.size() << ") "
                << "[i_leaf: " << i_leaf << "]");
        }
      }
    }
  };

  // Called from "evaluateMT" on Forest
  static void
  evaluateMT
    (
    const Sample *sample,
    TreeNode<Sample> *node,
    Leaf **leaf
    )
  {
    if (node->isLeaf())
      *leaf = node->getLeaf();
    else
    {
      if (node->eval(sample))
        evaluateMT(sample, node->left, leaf);
      else
        evaluateMT(sample, node->right, leaf);
    }
  };

  static bool
  load
    (
    Tree **tree,
    std::string path
    )
  {
    // Check if file exist
    std::ifstream ifs(path.c_str());
    if (!ifs.is_open())
    {
      PRINT("  File not found: " << path);
      return false;
    }

    try
    {
      boost::archive::text_iarchive ia(ifs);
      Tree *tree_aux = new Tree();
      ia >> *tree_aux;
      *tree = tree_aux;
      if ((*tree)->isFinished())
      {
        PRINT("  Complete tree reloaded");
      }
      else
      {
        PRINT("  Unfinished tree reloaded");
      }
      ifs.close();
      return true;
    }
    catch (boost::archive::archive_exception &ex)
    {
      ERROR("  Exception during tree serialization: " << ex.what());
      ifs.close();
      return false;
    }
    catch (int ex)
    {
      ERROR("  Exception: " << ex);
      ifs.close();
      return false;
    }
  };

  void
  save
    ()
  {
    try
    {
      std::ofstream ofs(m_save_path.c_str());
      boost::archive::text_oarchive oa(ofs);
      oa << *this; // it can also save unfinished trees
      ofs.flush();
      ofs.close();
      PRINT("Complete tree saved: " << m_save_path);
    }
    catch (boost::archive::archive_exception &ex)
    {
      ERROR("Exception during tree serialization: " << ex.what());
    }
  };

  TreeNode<Sample> *root; // root node of the tree

private:
  bool
  findOptimalSplit
    (
    const std::vector<Sample*> &samples,
    Split &best_split,
    int depth
    )
  {
    // Number of tests to find the best split
    std::vector<Split> splits(m_param.ntests);
    boost::uniform_int<> dist_split(0, 100);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_split(*m_rng, dist_split);
    int split_mode = rand_split();
    SplitGen<Sample> sg(samples, splits, m_rng, m_param.getPatchSize(), depth, split_mode);
    sg.generate();

    // Select the splitting which maximizes the information gain
    best_split.info = boost::numeric::bounds<double>::lowest();
    best_split.oob  = boost::numeric::bounds<double>::highest();

    for (unsigned int i=0; i < splits.size(); i++)
      if (splits[i].info > best_split.info)
        best_split = splits[i];

    if (best_split.info != boost::numeric::bounds<double>::lowest())
      return true;

    return false;
  };

  void
  applyOptimalSplit
    (
    const std::vector<Sample*> &samples,
    Split &best_split,
    std::vector< std::vector<Sample*> > &sets
    )
  {
    // Process each patch with the optimal R1 and R2
    std::vector<IntIndex> val_set(samples.size());
    for (unsigned int i=0; i < samples.size(); ++i)
    {
      val_set[i].first  = samples[i]->evalTest(best_split);
      val_set[i].second = i;
    }
    std::sort(val_set.begin(), val_set.end());
    SplitGen<Sample>::splitSamples(samples, val_set, sets, best_split.threshold, best_split.margin);
  };

  void
  saveAuto
    ()
  {
    double ticks = static_cast<double>(cv::getTickCount());
    double time = (ticks-m_ticks)/cv::getTickFrequency();
    TRACE("Time: " << time*1000 << " ms");
    // Save every 10 minutes
    if (time > 600)
    {
      m_ticks = ticks;
      PRINT("Automatic tree saved at " << m_ticks);
      save();
    }
  };

  boost::mt19937 *m_rng;
  double m_ticks;
  int m_num_nodes;
  int i_node;
  int i_leaf;
  ForestParam m_param;
  std::string m_save_path;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & m_num_nodes;
    ar & i_node;
    ar & m_param;
    ar & m_save_path;
    ar & root;
  }
};

#endif /* TREE_HPP */
