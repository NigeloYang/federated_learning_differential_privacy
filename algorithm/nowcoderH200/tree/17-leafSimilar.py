# -*- coding: utf-8 -*-
# @Time    : 2023/10/13

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def leafSimilar(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        
        pnums = self.levelOrder(p)
        qnums = self.levelOrder(q)
        
        if len(pnums) != len(qnums):
            return False
        else:
            n = len(pnums)
            for i in range(n):
                if pnums[i] != qnums[n - i - 1]:
                    return False
            return True
    
    def levelOrder(self, root: TreeNode):
        nodes = []
        nodes.append(root)
        nums = []
        while nodes:
            node = nodes.pop(0)
            if not node.left and not node.right:
                nums.append(node.val)
            
            if node.left:
                nodes.append(node.left)
            if node.right:
                nodes.append(node.right)
        return nums
    
if __name__ == "__main__":
    print()
