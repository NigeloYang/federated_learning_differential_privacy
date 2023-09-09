# -*- coding: utf-8 -*-
# @Time    : 2023/7/25

'''二叉树的最近公共祖先
给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。
百度百科中最近公共祖先的定义为：“对于有根树 T 的两个节点 p、q，最近公共祖先表示为一个节点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root or root == p or root == q:
            return root
        leftval = self.lowestCommonAncestor(root.left, p, q)
        rightval = self.lowestCommonAncestor(root.right, p, q)
        if not leftval:
            return rightval
        if not rightval:
            return leftval
        return root


if __name__ == "__main__":
    pass
