/** ****************************************************************************
 *  @file    Forest.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FOREST_HPP
#define FOREST_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <Tree.hpp>
#include <trace.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

/** ****************************************************************************
 * @class Forest
 * @brief Conditional regression forest
 ******************************************************************************/
template<typename Sample>
class Forest
{
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  Forest() {};

  /*Forest
    (
    ForestParam fp
    ) : m_forest_param(fp) {};

  Forest(const std::vector<Sample*> data, ForestParam fp, boost::mt19937* rng)
  {
    for (int i=0; i < fp.nTrees; i++)
    {
      Tree<Sample>* tree = new Tree<Sample>(data, fp, rng);
      m_trees.push_back(tree);
    }
  };*/

  void
  addTree
    (
    Tree<Sample> *tree
    )
  {
    m_trees.push_back(tree);
  };

  Tree<Sample>*
  getTree
    (
    int idx
    )
  {
    return m_trees[idx];
  };

  //sends the Sample down the tree
  /*void evaluate(const Sample* f, std::vector<Leaf*>& leafs) const {
    for (unsigned int i = 0; i < numberOfTrees(); i++)
      m_trees[i]->evaluate(f, m_trees[i]->root, leafs);
  }*/

  // Called from "getHeadPoseVotesMT" and "getFacialFeaturesVotesMT"
  void
  evaluateMT
    (
    const Sample *sample,
    Leaf **leafs
    ) const
  {
    for (unsigned int i=0; i < numberOfTrees(); i++)
    {
      m_trees[i]->evaluateMT(sample, m_trees[i]->root, leafs);
      leafs++;
    }
  };

  /*void save(std::string url, int offset = 0) {
    for (unsigned int i = 0; i < numberOfTrees(); i++) {

      char buffer[200];
      sprintf(buffer, "%s%03d.txt", url.c_str(), i + offset);

      std::string path = buffer;
      m_trees[i]->save(buffer);
    }
  }*/

  bool
  load
    (
    std::string path,
    ForestParam fp,
    int max_trees = -1
    )
  {
    setParam(fp);
    if (max_trees == -1)
      max_trees = fp.ntrees;
    PRINT("> Trees to load: " << fp.ntrees);

    for (int i=0; i < fp.ntrees; i++)
    {
      if (numberOfTrees() > max_trees)
        continue;

      char buffer[200];
      sprintf(buffer, "%s/tree_%03d.txt", path.c_str(), i);
      std::string tree_path = buffer;
      PRINT("  Load " << tree_path);
      if (!load_tree(tree_path, m_trees))
        return false;
    }
    return true;
  };

  static bool
  load_tree
    (
    std::string url,
    std::vector<Tree<Sample>*> &trees
    )
  {
    Tree<Sample> *tree;
    if (!Tree<Sample>::load(&tree, url))
      return false;

    if (tree->isFinished())
    {
      trees.push_back(tree);
      return true;
    }
    else
    {
      PRINT("  Tree is not finished successfully")
      delete tree;
      return false;
    }
  };

  void
  setParam
    (
    ForestParam fp
    )
  {
    m_forest_param = fp;
  };

  ForestParam
  getParam
    () const
  {
    return m_forest_param;
  };

  /*std::vector<float> getClassWeights() {
    return m_trees[0]->getClassWeights();
  }

  void getAllLeafs(std::vector<std::vector<Leaf*> >& leafs) {
    leafs.resize(numberOfTrees());
    for (unsigned int i = 0; i < numberOfTrees(); i++)
      m_trees[i]->root->collectLeafs(leafs[i]);

  }*/

  int
  numberOfTrees
    () const
  {
    return static_cast<int>(m_trees.size());
  };

  void
  cleanForest
    ()
  {
    m_trees.clear();
  };

private:
  std::vector<Tree<Sample>*> m_trees;
  ForestParam m_forest_param;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & m_trees;
  }
};

#endif /* FOREST_HPP */
