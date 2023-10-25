# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def isValidBST(self , root: TreeNode) -> bool:
        if not root:
            return True

        if (root.left and root.left.val > root.val) or (root.right and root.right.val < root.val):
            return False

        return self.isValidBST(root.left) and self.isValidBST(root.right)
    
if __name__ == "__main__":
    print()
