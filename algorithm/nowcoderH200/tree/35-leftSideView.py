# -*- coding: utf-8 -*-
# @Time    : 2023/10/16

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def leftSideView(self, root: TreeNode) -> List[int]:
        if not root:
            return root
        
        nodes = [root]
        ans = []
        while nodes:
            subweight = []
            for i in range(len(nodes)):
                node = nodes.pop(0)
                subweight.append(node.val)
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            ans.append(subweight[0])
        return ans
    
if __name__ == "__main__":
    print()
