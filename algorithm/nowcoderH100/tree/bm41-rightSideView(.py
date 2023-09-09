# -*- coding: utf-8 -*-
# @Time    : 2023/9/8

from collections import deque

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
#


class Solution:
    def solve(self, preOrder: List[int], inOrder: List[int]) -> List[int]:
        
        root = self.reconstructtree(preOrder, inOrder)
        
        max_depth = -1
        curnodes = deque([(root, 0)])
        res = dict()
        
        while curnodes:
            node, depth = curnodes.popleft()
            
            if node:
                max_depth = max(depth, max_depth)
                
                res[max_depth] = node.val
                
                curnodes.append((node.left, depth + 1))
                curnodes.append((node.right, depth + 1))
        
        return [res[i] for i in range(max_depth + 1)]
    
    def reconstructtree(self, preo: List[int], ino: List[int]):
        if len(preo) == 0 or len(ino) == 0:
            return
        
        root = TreeNode(preo[0])
        
        for i in range(len(ino)):
            if preo[0] == ino[i]:
                root.left = self.reconstructtree(preo[1:i + 1], ino[:i])
                root.right = self.reconstructtree(preo[i + 1:], ino[i + 1:])
        
        return root
    
if __name__ == "__main__":
    print()
