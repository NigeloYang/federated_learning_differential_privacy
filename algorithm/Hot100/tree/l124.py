# -*- coding: utf-8 -*-
# @Time    : 2023/7/25

'''二叉树中的最大路径和
二叉树中的 路径 被定义为一条节点序列，序列中每对相邻节点之间都存在一条边。同一个节点在一条路径序列中 至多出现一次 。该路径 至少包含一个 节点，且不一定经过根节点。
路径和 是路径中各节点值的总和。
给你一个二叉树的根节点 root ，返回其 最大路径和 。'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root) -> int:
        self.maxSum = float("-inf")
    
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            innerMaxSum = left + right + node.val
            self.maxSum = max(self.maxSum, innerMaxSum)
            outMaxSum = node.val + max(left, right)
            return 0 if outMaxSum < 0 else outMaxSum
    
        dfs(root)
        return self.maxSum

if __name__ == "__main__":
    print(float('-inf'))
