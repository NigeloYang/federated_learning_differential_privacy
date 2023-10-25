# -*- coding: utf-8 -*-
# @Time    : 2023/10/14

''' 牛群排列的最大深度
在一个牧场中，有很多牛。为了方便管理，牧场主将牛的编号排列成一棵二叉树。请你编写一个程序，给定一棵二叉树的根节点 root，计算出这棵树的最大深度。

二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

叶子节点是指没有子节点的节点。
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
    
if __name__ == "__main__":
    print()
