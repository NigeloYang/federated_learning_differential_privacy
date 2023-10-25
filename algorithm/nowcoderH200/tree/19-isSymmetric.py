# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def isSymmetric(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        return self.dfsTree(root.left, root.right)
    
    def dfsTree(self, left: TreeNode, right: TreeNode):
        if not left and not right:
            return True
        
        if not left or not right or left.val != right.val:
            return False
        
        return self.dfsTree(left.left, right.right) and self.dfsTree(left.right, right.left)



if __name__ == "__main__":
    print()
