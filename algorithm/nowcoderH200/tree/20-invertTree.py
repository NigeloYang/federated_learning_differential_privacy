# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def invertTree(self , root: TreeNode) -> TreeNode:
        if not root:
            return root
        root.left,root.right = root.right,root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
    
if __name__ == "__main__":
    print()
