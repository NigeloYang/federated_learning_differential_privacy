# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def ZLevelOrder(self, root: TreeNode) -> List[List[int]]:
        if not root:
            return root
        
        nodes = []
        nodes.append(root)
        flag = False
        res = []
        while nodes:
            levenodes = []
            for i in range(len(nodes)):
                node = nodes.pop(0)
                levenodes.append(node.val)
                
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            
            if flag:
                res.append(levenodes[::-1])
            else:
                res.append(levenodes)
            flag = not flag
        
        return res
if __name__ == "__main__":
    print()
