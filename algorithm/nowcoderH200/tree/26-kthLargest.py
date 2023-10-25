# -*- coding: utf-8 -*-
# @Time    : 2023/10/14

''' 二叉树之寻找第k大
农场主人有一群牛，他给每只牛都打了一个编号。这些牛按照编号的大小形成了一颗二叉搜索树。现在农场主人想知道编号第k大的牛是哪一只，你能帮他设计一个算法来实现这个功能吗？
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def kthLargest(self, root: TreeNode, k: int) -> int:
        if not root:
            return root
        
        res = []
        self.inorder(root, res)
        return res[-k]
    
    def inorder(self, root, res):
        if root:
            self.inorder(root.left, res)
            res.append(root.val)
            self.inorder(root.right, res)
            
if __name__ == "__main__":
    print()
