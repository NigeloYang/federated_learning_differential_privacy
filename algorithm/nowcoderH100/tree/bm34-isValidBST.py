# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

import sys
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    pre = -sys.maxsize - 1
    
    def isValidBST(self, root: TreeNode) -> bool:
        if not root:
            return True
        
        if not self.isValidBST(root.left):
            return False
        
        if root.val <= self.pre:
            return False
        
        self.pre = root.val
        
        if not self.isValidBST(root.right):
            return False
        
        return True
    
if __name__ == "__main__":
    print()
