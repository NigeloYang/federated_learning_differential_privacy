# -*- coding: utf-8 -*-
# @Time    : 2023/7/23

''' 二叉搜索树中第K小的元素
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 个最小元素（从 1 开始计数）。'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root, k: int) -> int:
        if not root:
            return
        res = []
        nodes = []
        nodes.append(root)
        while len(nodes) > 0:
            node = nodes.pop(0)
            res.append(node.val)
            if node.left:
                nodes.append(node.left)
            if node.right:
                 nodes.append(node.right)
        res.sort()
        return res[k-1]
    
    def kthSmallest2(self, root, k: int) -> int:
        res = []
        def inorder(node):
            if not node:
                return
            inorder(node.left)
            res.append(node.val)
            inorder(node.right)
        inorder(root)
        return res[k-1]
    
    def kthSmallest3(self, root, k: int) -> int:
        stack = []
        curr = root
        
        while curr or len(stack):
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            
            k -= 1
            if k == 0:
                return curr.val
            curr = curr.right
            
if __name__ == "__main__":
    pass
