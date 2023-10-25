# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def levelOrder(self, root: TreeNode) -> List[str]:
        if not root:
            return root
        
        nodes = []
        nodes.append(root)
        res = []
        while nodes:
            levenodes = ''
            for i in range(len(nodes)):
                node = nodes.pop(0)
                levenodes += str(node.val)
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            res.append(levenodes)
        return res
    
if __name__ == "__main__":
    print()
