# -*- coding: utf-8 -*-
# @Time    : 2023/10/22


import collections

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def __init__(self):
        self.prefix = collections.defaultdict(int)
        self.prefix[0] = 1
        self.ans = 0
    
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        self.dfs(root, 0, targetSum)
        return self.ans
    
    def dfs(self, root, curr, targetSum):
        if not root:
            return 0
        
        curr += root.val
        self.ans += self.prefix[curr - targetSum]
        self.prefix[curr] += 1
        self.dfs(root.left, curr, targetSum)
        self.dfs(root.right, curr, targetSum)
        self.prefix[curr] -= 1


class Solution2:
    def __init__(self):
        self.ans = 0
    
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        if not root:
            return 0
        
        self.nodeSum(root, targetSum)
        self.pathSum(root.left, targetSum)
        self.pathSum(root.right, targetSum)
        
        return self.ans
    
    def nodeSum(self, root, targetSum):
        if not root:
            return 0
        
        if root.val == targetSum:
            self.ans += 1
        
        self.nodeSum(root.left, targetSum - root.val)
        self.nodeSum(root.right, targetSum - root.val)
        
if __name__ == "__main__":
    print()
