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

  /*Tree
    (
    const std::vector<Sample*> &samples,
    ForestParam fp,
    boost::mt19937 *rng,
    std::string save_path,
    Timing job_timer = Timing()
    )
  {
    m_rng = rng;
    timer = job_timer;
    timer.restart();
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
    save(m_save_path);
  };*/

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

  /*void
  grow
    (
    TreeNode<Sample> *node,
    const std::vector<Sample*> &samples
    )
  {
    int depth = node->getDepth();
    int nElements = samples.size(); // count elements
    std::vector<Sample*> setA, setB;
    if (nElements < m_fp.min_s || depth >= m_fp.max_d || node->isLeaf())
    {
      node->createLeaf(samples, m_class_weights, i_leaf);
      i_node += pow(2.0, int((m_fp.max_d - depth) + 1)) - 1;
      i_leaf++;
      PRINT("  (a) " << int((i_node/m_num_nodes)*100) << "% : make leaf(depth: " << depth
            << ", elements: " << samples.size() << ") [i_leaf: " << i_leaf << "]");
    }
    else
    {
      Split bestSplit;
      if (node->hasSplit()) //only in reload mode.
      {
        bestSplit = node->getSplit();
        split(samples, bestSplit, setA, setB);
        i_node++;
        PRINT("  (b) " << int((i_node/m_num_nodes)*100) << "% : split(depth: " << depth << ", elements: "
              << nElements << ") [setA: " << setA.size() << ", setB: " << setB.size() << "], oob: 0");

        grow(node->left, setA);
        grow(node->right, setB);
      }
      else
      {
        bool testFound = findOptimalSplit(samples, bestSplit, setA, setB, depth);
        if (testFound)
        {
          split(samples, bestSplit, setA, setB);
          node->setSplit(bestSplit);

          i_node++;

          TreeNode<Sample> *left = new TreeNode<Sample>(depth + 1);
          node->addLeftChild(left);

          TreeNode<Sample> *right = new TreeNode<Sample>(depth + 1);
          node->addRightChild(right);

          autoSave();
          PRINT("  (c) " << int((i_node/m_num_nodes)*100) << "% : split(depth: " << depth << ", elements: "
                << nElements << ") [setA: " << setA.size() << ", setB: " << setB.size() << "]");

          grow(left, setA);
          grow(right, setB);
        }
        else
        {
          PRINT("  No valid split found");
          node->createLeaf(samples, m_class_weights, i_leaf);
          i_leaf++;
          i_node += (int) pow(2.0, int((m_fp.max_d - depth) + 1)) - 1;
          PRINT("  (d) " << int((i_node/m_num_nodes)*100) << "% : make leaf(depth: " << depth
                << ", elements: "  << samples.size() << ") [i_leaf: " << i_leaf << "]");
        }
      }
    }
  };

  bool
  findOptimalSplit
    (
    const std::vector<Sample*>& data,
    Split& best_split,
    std::vector<Sample*>& set_a,
    std::vector<Sample*>& set_b,
    int depth
    )
  {
    best_split.info = boost::numeric::bounds<double>::lowest();
    best_split.gain = boost::numeric::bounds<double>::lowest();
    best_split.oob = boost::numeric::bounds<double>::highest();
    int num_splits = m_fp.nTests;

    std::vector<Split> splits(num_splits);

    double timeStamp = timer.elapsed();

    boost::uniform_int<> dist_split(0, 100);
    boost::variate_generator<boost::mt19937&, boost::uniform_int<> > rand_split(*m_rng, dist_split);
    int split_mode = rand_split();
    SplitGen<Sample> sg(data, splits, m_rng, m_fp, m_class_weights, depth, split_mode);
    sg.generate();

    std::cout << timer.elapsed() << ": for split: " << (timer.elapsed() - timeStamp) << " ("
        << (timer.elapsed() - timeStamp) / float(data.size()) << ") mode: " << split_mode << std::endl;
    for (unsigned i = 0; i < splits.size(); i++) {

      if (splits[i].info > best_split.info) {
        best_split = splits[i];
      }
    }
    if (best_split.info != boost::numeric::bounds<double>::lowest())
      return true;
    return false;
  };

  void split(const std::vector<Sample*>& data, Split& best_split,
      std::vector<Sample*>& set_a, std::vector<Sample*>& set_b) {
    //generate Value for each Patch
    std::vector<IntIndex> valSet(data.size());
    for (unsigned int l = 0; l < data.size(); ++l) {
      valSet[l].first = data[l]->evalTest(best_split);
      valSet[l].second = l;
    }
    std::sort(valSet.begin(), valSet.end());

    SplitGen<Sample>::splitVec(data, valSet, set_a, set_b, best_split.threshold, best_split.margin);
  };*/

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

  /*void autoSave() {
    int tStamp = timer.elapsed();
    int saveInterval = 150000;
    //save every 10 minutes
    if ((tStamp - m_last_save_point) > saveInterval) {
      m_last_save_point = timer.elapsed();
      std::cout << timer.elapsed() << ": save at " << m_last_save_point << std::endl;
      save(m_save_path);
    }
  };

  //saves the tree recursive
  //it can also save unfinished trees
  void save(std::string path) {
    try {
      std::ofstream ofs(path.c_str());
      boost::archive::binary_oarchive oa(ofs);
      oa << *this;
      ofs.flush();
      ofs.close();
      std::cout << "saved " << path << std::endl;
    } catch (boost::archive::archive_exception& ex) {
      std::cout << "Archive Exception during serializing:" << std::endl;
      std::cout << ex.what() << std::endl;
      std::cout << "it was tree: " << path << std::endl;
    }
  };*/

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
      ERROR("  File not found: " << path);
      return false;
    }

    try
    {
      boost::archive::binary_iarchive ia(ifs);
      ia >> *tree;
      ifs.close();
      return true;
    }
    catch (boost::archive::archive_exception &ex)
    {
      ERROR("  Exception during tree deserializing: " << ex.what());;
      ifs.close();
      return false;
    }
    catch (int e)
    {
      ERROR("  Exception: " << e);
      ifs.close();
      return false;
    }
  };

  TreeNode<Sample> *root; // root node of the tree

private:
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
