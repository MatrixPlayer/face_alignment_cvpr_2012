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
#include <Timing.hpp>
#include <TreeNode.hpp>
#include <SplitGen.hpp>

#include <fstream>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

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
    ()
  {
    timer.restart();
    m_last_save_point = 0;
  };

  // Train a random tree from "train_headpose" or "train_ffd"
  Tree
    (
    const std::vector<Sample*> &samples,
    ForestParam fp,
    boost::mt19937 *rng,
    std::string save_path
    )
  {
    m_rng = rng;
    timer = Timing();
    m_last_save_point = 0;
    m_fp = fp;
    m_num_nodes = pow(2.0, m_fp.max_d+1) - 1;
    i_node = 0;
    i_leaf = 0;
    m_save_path = save_path;
    Sample::calcWeightClasses(m_class_weights, samples);
    root = new TreeNode<Sample>(0);

    PRINT("(+) Start Training");
    grow(root, samples);
    //save(m_save_path);
  };

  virtual
  ~Tree
    ()
  {
    if (root)
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

  /*std::vector<float> getClassWeights() {
    return m_class_weights;
  };*/

  /*//start growing the tree
  void grow(const std::vector<Sample*>& data, Timing jobTimer, boost::mt19937* rng_) {
    m_rng = rng_;
    timer = jobTimer;
    timer.restart();
    m_last_save_point = timer.elapsed();

    std::cout << int((i_node / m_num_nodes) * 100) << "% : LOADED TREE " << std::endl;
    if (!isFinished()) {
      i_node = 0;
      i_leaf = 0;
      grow(root, data);
      save(m_save_path);
    }
  }*/

  void
  grow
    (
    TreeNode<Sample> *node,
    const std::vector<Sample*> &samples
    )
  {
    int depth = node->getDepth();
    int nelements = static_cast<int>(samples.size());
    if (nelements < m_fp.min_s || depth >= m_fp.max_d || node->isLeaf())
    {
      node->createLeaf(samples, m_class_weights, i_leaf);
      i_node += pow(2.0, m_fp.max_d-depth+1) - 1;
      i_leaf++;
      PRINT("  (1) " << (i_node/m_num_nodes)*100 << "% : make leaf(depth: " << depth <<
            ", elements: " << samples.size() << ") [i_leaf: " << i_leaf << "]");
    }
    else
    {
      Split best_split;
      std::vector<Sample*> setA, setB;
      if (node->hasSplit()) // only in reload mode
      {
        best_split = node->getSplit();
        split(samples, best_split, setA, setB);
        i_node++;
        PRINT("  (2) " << (i_node/m_num_nodes)*100 << "% : split(depth: " << depth <<
              ", elements: " << nelements << ") [A: " << setA.size() << ", B: " << setB.size() << "]");

        grow(node->left, setA);
        grow(node->right, setB);
      }
      else
      {
        if (findOptimalSplit(samples, best_split, depth))
        {
          split(samples, best_split, setA, setB);
          node->setSplit(best_split);
          i_node++;

          TreeNode<Sample> *left = new TreeNode<Sample>(depth + 1);
          node->addLeftChild(left);

          TreeNode<Sample> *right = new TreeNode<Sample>(depth + 1);
          node->addRightChild(right);

          autoSave();
          PRINT("  (3) " << (i_node/m_num_nodes)*100 << "% : split(depth: " << depth <<
                ", elements: " << nelements << ") [A: " << setA.size() << ", B: " << setB.size() << "]");

          grow(left, setA);
          grow(right, setB);
        }
        else
        {
          PRINT("  No valid split found");
          node->createLeaf(samples, m_class_weights, i_leaf);
          i_node += pow(2.0, m_fp.max_d-depth+1) - 1;
          i_leaf++;
          PRINT("  (4) " << (i_node/m_num_nodes)*100 << "% : make leaf(depth: " << depth <<
                ", elements: "  << samples.size() << ") [i_leaf: " << i_leaf << "]");
        }
      }
    }
  };

  //sends the sample down the tree and return a pointer to the leaf.
  /*static void evaluate(const Sample* sample, TreeNode<Sample>* node,
      std::vector<Leaf*>& leafs) {
    if (node->isLeaf())
      leafs.push_back(node->getLeaf());
    else {
      if (node->eval(sample)) {
        evaluate(sample, node->left, leafs);
      } else {
        evaluate(sample, node->right, leafs);
      }
    }
  };*/

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
      boost::archive::binary_iarchive ia(ifs);
      ia >> *tree;
      if ((*tree)->isFinished())
      {
        PRINT("  Complete tree reloaded");
      }
      else
      {
        PRINT("  Unfinished tree reloaded, keep growing ...");
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
    (
    std::string path
    )
  {
    try
    {
      std::ofstream ofs(path.c_str());
      boost::archive::binary_oarchive oa(ofs);
      oa << *this;  // it can also save unfinished trees
      ofs.flush();
      ofs.close();
      PRINT("  Saved: " << path);
    }
    catch (boost::archive::archive_exception &ex)
    {
      ERROR("  Exception during tree serialization: " << ex.what());
    }
  };

  TreeNode<Sample> *root; // root node of the tree

private:
  // Called from "grow"
  bool
  findOptimalSplit
    (
    const std::vector<Sample*> &samples,
    Split &best_split,
    int depth
    )
  {
    best_split.info = boost::numeric::bounds<double>::lowest();
    best_split.gain = boost::numeric::bounds<double>::lowest();
    best_split.oob  = boost::numeric::bounds<double>::highest();

    int num_splits = m_fp.nTests; // 500 tests to find the best split
    std::vector<Split> splits(num_splits);

    float time_stamp = timer.elapsed();
    boost::uniform_int<> dist_split(0, 100);
    boost::variate_generator< boost::mt19937&, boost::uniform_int<> > rand_split(*m_rng, dist_split);
    int split_mode = rand_split();
    SplitGen<Sample> sg(samples, splits, m_rng, m_fp, depth, split_mode);
    sg.generate();
    PRINT("  Optimal split mode: " << split_mode);
    PRINT("  Time: " << timer.elapsed()-time_stamp << " ms (" << timer.elapsed() << " ms)");

    // Select the splitting which maximizes the information gain
    for (unsigned i=0; i < splits.size(); i++)
      if (splits[i].info > best_split.info)
        best_split = splits[i];

    if (best_split.info != boost::numeric::bounds<double>::lowest())
      return true;

    return false;
  };

  // Called from "grow"
  void
  split
    (
    const std::vector<Sample*> &samples,
    Split &best_split,
    std::vector<Sample*> &set_a,
    std::vector<Sample*> &set_b
    )
  {
    // Generate value for each patch
    std::vector<IntIndex> valSet(samples.size());
    for (unsigned int l=0; l < samples.size(); ++l)
    {
      valSet[l].first = samples[l]->evalTest(best_split);
      valSet[l].second = l;
    }
    std::sort(valSet.begin(), valSet.end());

    SplitGen<Sample>::splitVec(samples, valSet, set_a, set_b, best_split.threshold, best_split.margin);
  };

  void
  autoSave
    ()
  {
    int time_stamp = timer.elapsed();
    int save_interval = 150000;
    // Save every 10 minutes
    if ((time_stamp - m_last_save_point) > save_interval)
    {
      m_last_save_point = timer.elapsed();
      PRINT(timer.elapsed() << " ms (autoSave at " << m_last_save_point << ")");
      save(m_save_path);
    }
  };

  boost::mt19937 *m_rng;
  Timing timer;
  int m_last_save_point; // the latest saving timestamp
  float m_num_nodes; //for statistic reason
  float i_node;
  int i_leaf;
  ForestParam m_fp;
  std::string m_save_path; // saving path of the trees
  std::vector<float> m_class_weights; //population throw classes

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & m_num_nodes;
    ar & i_node;
    ar & m_fp;
    ar & m_save_path;
    ar & m_class_weights;
    ar & root;
  }
};

#endif /* TREE_HPP */
