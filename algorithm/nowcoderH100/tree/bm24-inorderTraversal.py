# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        res = []
        self.inorder(root, res)
        return res
    
    def inorder(self, root: TreeNode, res: List[int]):
        if not root:
            return
        
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)
        
if __name__ == "__main__":
    print()
