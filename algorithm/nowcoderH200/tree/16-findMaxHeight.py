# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def findMaxHeight(self, root: TreeNode) -> int:
        if not root:
            return 0
        ans = root.val
        left = self.findMaxHeight(root.left)
        if left > ans:
            ans = left
        right = self.findMaxHeight(root.right)
        if right > ans:
            ans = right
        
        return ans
    
if __name__ == "__main__":
    print()
