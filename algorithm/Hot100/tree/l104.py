# -*- coding: utf-8 -*-
# @Time    : 2023/7/22

'''二叉树的最大深度
给定一个二叉树，找出其最大深度。
二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
说明: 叶子节点是指没有子节点的节点。'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root) -> int:
        def getdepth(root):
            if not root:
                return 0
            l = getdepth(root.left)
            r = getdepth(root.right)
            return max(l, r) + 1
        
        return getdepth(root)


if __name__ == "__main__":
    pass
