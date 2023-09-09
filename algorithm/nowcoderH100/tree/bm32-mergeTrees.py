# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1 and not t2:
            return t1
        elif not t1:
            return t2
        elif not t2:
            return t1
        
        head = TreeNode(t1.val + t2.val)
        head.left = self.mergeTrees(t1.left, t2.left)
        head.right = self.mergeTrees(t1.right, t2.right)
        
        return head
    

if __name__ == "__main__":
    print()
