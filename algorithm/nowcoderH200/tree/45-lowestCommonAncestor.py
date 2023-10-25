# -*- coding: utf-8 -*-
# @Time    : 2023/10/22

'''二叉树最长公共祖先

'''

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: int, q: int) -> int:
        if not root:
            return -1
    
        ans = self.dfs(root, p, q)
        return ans.val

    def dfs(self, root: TreeNode, p: int, q: int):
        if not root or root.val == p or root.val == q:
            return root
        leftval = self.lowestCommonAncestor(root.left, p, q)
        rightval = self.lowestCommonAncestor(root.right, p, q)
        if not leftval:
            return rightval
        if not rightval:
            return leftval
        return root
    
if __name__ == "__main__":
    print()
