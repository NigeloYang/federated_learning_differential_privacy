# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def preorderTraversal(self , root: TreeNode) -> List[int]:
        res = []
        self.preorder(root,res)
        return res

    def preorder(self,root:TreeNode,res:List[int]):
        if not root:
            return
        res.append(root.val)
        self.preorder(root.left,res)
        self.preorder(root.right,res)
        
if __name__ == "__main__":
    print()
