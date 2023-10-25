# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxLevelSum(self, root: TreeNode) -> int:
        if not root:
            return root
        
        nodes = []
        nodes.append(root)
        leve = 1
        maxsum = 0
        ans = 0
        while nodes:
            levesum = 0
            for i in range(len(nodes)):
                node = nodes.pop(0)
                levesum += int(node.val)
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            if levesum > maxsum:
                ans = leve
                maxsum = levesum
            leve += 1
        return ans
    
if __name__ == "__main__":
    print()
