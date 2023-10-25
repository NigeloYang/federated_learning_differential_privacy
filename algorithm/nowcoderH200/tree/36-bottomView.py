# -*- coding: utf-8 -*-
# @Time    : 2023/10/16

class Solution:
    def bottomView(self, root: TreeNode) -> List[int]:
        if not root:
            return root
        
        res = []
        self.dfsTree(root, res)
        return res
    
    def dfsTree(self, root, res):
        if not root:
            return
        
        self.dfsTree(root.left, res)
        self.dfsTree(root.right, res)
        
        if not root.left and not root.right:
            res.append(root.val)
            
if __name__ == "__main__":
    print()
