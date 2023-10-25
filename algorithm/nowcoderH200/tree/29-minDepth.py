# -*- coding: utf-8 -*-
# @Time    : 2023/10/15

'''牛群的最短路径
在一个牧场中，有很多牛。为了方便管理，牧场主将牛的编号排列成一棵二叉树。每个牛的编号为一个正整数。请你编写一个程序，给定一棵二叉树的根节点 root，计算出这棵树中，从根节点到叶子节点的最短路径上的节点数。

叶子节点是指没有子节点的节点。
'''

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def minDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
    
        if not root.left and not root.right:
            return 1
    
        if root.left and not root.right:
            return 1 + self.minDepth(root.left)
    
        if not root.left and root.right:
            return 1 + self.minDepth(root.right)
    
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
    
if __name__ == "__main__":
    print()
