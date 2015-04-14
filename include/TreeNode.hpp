/** ****************************************************************************
 *  @file    TreeNode.hpp
 *  @brief   Real-time facial feature detection
 *  @author  Matthias Dantone
 *  @date    2011/05
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef TREE_NODE_HPP
#define TREE_NODE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <boost/serialization/access.hpp>

/** ****************************************************************************
 * @class TreeNode
 * @brief Conditional regression tree node
 ******************************************************************************/
template<typename Sample>
class TreeNode
{
public:
  typedef typename Sample::Split Split;
  typedef typename Sample::Leaf Leaf;

  TreeNode
    () :
    depth(-1), right(NULL), left(NULL), is_leaf(false), has_split(false) {};

  TreeNode
    (
    int d
    ) :
    depth(d), right(NULL), left(NULL), is_leaf(false), has_split(false) {};

  ~TreeNode
    ()
  {
    if (left)
      delete left;
    if (right)
      delete right;
  };

  // Called from "grow"
  int
  getDepth
    ()
  {
    return depth;
  };

  bool
  isLeaf
    () const
  {
    return is_leaf;
  };

  Leaf*
  getLeaf
    ()
  {
    return &leaf;
  };

  /*void
  setLeaf
    (
    Leaf l
    )
  {
    is_leaf = true;
    leaf = l;
  };*/

  // Called from "grow"
  void
  createLeaf
    (
    const std::vector<Sample*> &samples
    )
  {
    Sample::makeLeaf(leaf, samples);
    is_leaf = true;
    has_split = false;
  };

  /*void
  collectLeafs
    (
    std::vector<Leaf*> &leafs
    )
  {
    if (!is_leaf)
    {
      right->collectLeafs(leafs);
      left->collectLeafs(leafs);
    } else {
      leaf.depth = depth;
      leafs.push_back(&leaf);
    }
  };*/

  // Called from "grow"
  bool
  hasSplit
    () const
  {
    return has_split;
  };

  // Called from "grow"
  Split
  getSplit
    ()
  {
    return split;
  };

  // Called from "grow"
  void
  setSplit
    (
    Split s
    )
  {
    has_split = true;
    is_leaf = false;
    split = s;
  }

  // Called from "grow"
  void
  addLeftChild
    (
    TreeNode<Sample> *left_child
    )
  {
    left = left_child;
  };

  // Called from "grow"
  void
  addRightChild
    (
    TreeNode<Sample> *right_child
    )
  {
    right = right_child;
  };

  // Called from "evaluateMT" on Tree
  bool
  eval
    (
    const Sample *s
    ) const
  {
    return s->eval(split);
  };

  Leaf leaf;
  Split split;
  TreeNode<Sample> *right;
  TreeNode<Sample> *left;

private:
  int depth;
  bool is_leaf;
  bool has_split;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned int version)
  {
    ar & depth;
    ar & is_leaf;
    ar & has_split;
    if (has_split)
      ar & split;
    if (!is_leaf)
    {
      ar & left;
      ar & right;
    }
    else
      ar & leaf;
  }
};

#endif /* TREE_NODE_HPP */
