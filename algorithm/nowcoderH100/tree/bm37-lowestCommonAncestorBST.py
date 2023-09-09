# -*- coding: utf-8 -*-
# @Time    : 2023/9/7

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: int, q: int) -> int:
        while root:
            if root.val < p and root.val < q:
                root = root.right
            elif root.val > p and root.val > q:
                root = root.left
            else:
                break
        return root.val
    
if __name__ == "__main__":
    print()
