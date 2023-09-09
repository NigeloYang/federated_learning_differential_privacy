# -*- coding: utf-8 -*-
# @Time    : 2023/7/25

'''从前序与中序遍历序列构造二叉树
给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def buildTree(self, preorder, inorder):
        if not preorder or not inorder:
            return
        root = TreeNode(preorder[0])
        leftidx = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:1 + leftidx], inorder[:leftidx])
        root.right = self.buildTree(preorder[1 + leftidx:], inorder[leftidx + 1:])
        return root


if __name__ == "__main__":
    pass
