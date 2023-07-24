# -*- coding: utf-8 -*-
# @Time    : 2023/7/22

'''翻转二叉树
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def invertTree(self, root):
        if not root:
            return root
        
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)


if __name__ == "__main__":
    pass
