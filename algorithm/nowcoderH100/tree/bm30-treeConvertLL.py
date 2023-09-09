# -*- coding: utf-8 -*-
# @Time    : 2023/9/6


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#
#
# @param pRootOfTree TreeNode类
# @return TreeNode类
#
class Solution:
    head = None
    pre = None
    
    def Convert(self, pRootOfTree):
        if not pRootOfTree:
            return None
        
        self.Convert(pRootOfTree.left)
        
        if not self.pre:
            self.pre = pRootOfTree
            self.head = pRootOfTree
        else:
            self.pre.right = pRootOfTree
            pRootOfTree.left = self.pre
            self.pre = pRootOfTree
        self.Convert(pRootOfTree.right)
        return self.head
    
if __name__ == "__main__":
    print()
