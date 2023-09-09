# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.depth(root) != -1
    
    def depth(self, root):
        if not root:
            return 0
        
        left = self.depth(root.left)
        if left == -1:
            return -1
        
        right = self.depth(root.right)
        if right == -1:
            return -1
        
        return max(left, right) + 1 if abs(left - right) <= 1 else -1



if __name__ == "__main__":
    print()
