# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, o1: int, o2: int) -> int:
        if not root or root.val == o1 or root.val == o2:
            return root.val
        
        lval = self.lowestCommonAncestor(root.left, o1, o2)
        rval = self.lowestCommonAncestor(root.right, o1, o2)
        
        if not lval:
            return rval
        if not rval:
            return lval
        
        return root.val
    
if __name__ == "__main__":
    print()
