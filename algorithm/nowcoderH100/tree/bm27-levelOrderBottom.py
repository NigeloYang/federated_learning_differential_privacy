# -*- coding: utf-8 -*-
# @Time    : 2023/9/6

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrderBottom(self, pRoot: Optional[TreeNode]) -> List[List[int]]:
        if not pRoot:
            return pRoot
        
        res = []
        curnodes = [pRoot]
        flag = True
        while curnodes:
            curval = []
            nextnodes = []
            flag = not flag
            for node in curnodes:
                if node:
                    curval.append(node.val)
                if node.left:
                    nextnodes.append(node.left)
                if node.right:
                    nextnodes.append(node.right)
            if flag:
                curval = curval[::-1]
            
            res.append(curval)
            curnodes = nextnodes
        
        return res
    
if __name__ == "__main__":
    print()
