# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def levelOrder(self , root: TreeNode) -> List[List[int]]:
        if not root:
            return root
        res = []
        cur = [root]
        while cur:
            curval = []
            nextnodes = []
            for node in cur:
                if node:
                    curval.append(node.val)
                if node.left:
                    nextnodes.append(node.left)
                if node.right:
                    nextnodes.append(node.right)
            if curval:
                res.append(curval)
            cur = nextnodes
        return res
    
if __name__ == "__main__":
    print()
