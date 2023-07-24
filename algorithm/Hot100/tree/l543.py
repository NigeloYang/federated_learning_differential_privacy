# -*- coding: utf-8 -*-
# @Time    : 2023/7/22

'''二叉树的直径
给你一棵二叉树的根节点，返回该树的 直径 。
二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。
两节点之间路径的 长度 由它们之间边数表示。'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root) -> int:
        self.res = 0
        
        def getDepth(root):
            if not root:
                return 0
            l = getDepth(root.left)
            r = getDepth(root.right)
            self.res = max(self.res, l + r + 1)
            return max(l, r) + 1
        
        getDepth(root)
        return self.res - 1


if __name__ == "__main__":
    pass
