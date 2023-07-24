# -*- coding: utf-8 -*-
# @Time    : 2023/7/24

''' 二叉树的右视图
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rightSideView(self, root):
        if not root:
            return []
    
        nodes = [root]
        res = []
        while nodes:
            res.append([node.val for node in nodes][-1])
            subnodes = []
            for node in nodes:
                if node.left:
                    subnodes.append(node.left)
                if root.right:
                    subnodes.append(node.right)
            nodes = subnodes
        return res
    
    def rightSideView2(self, root):
        if not root:
            return []
        
        queue = [root]
        res = []
        while queue:
            res.append([node.val for node in queue][-1])
            ll = []
            for node in queue:
                if node.left:
                    ll.append(node.left)
                if node.right:
                    ll.append(node.right)
            queue = ll
        return res
    
if __name__ == "__main__":
    pass
