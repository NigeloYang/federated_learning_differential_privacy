# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

''' 第k轻的牛牛
在一个牧场中，有很多牛。为了方便管理，牧场主将牛按照体重排列成一棵二叉搜索树。现在牧场主想知道牛群中第 k 轻的牛牛体重是多少。
请你编写一个程序，给定一棵二叉搜索树的根节点 root 和一个整数 k，查找其中第 k 个最小元素（从 1 开始计数）
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def kthLighest(self, root: TreeNode, k: int) -> int:
        if not root:
            return root
        res = []
        self.inorder(root, res)
        
        return res[k - 1]
    
    def inorder(self, root, res):
        if root:
            self.inorder(root.left, res)
            res.append(root.val)
            self.inorder(root.right, res)
            
if __name__ == "__main__":
    print()
