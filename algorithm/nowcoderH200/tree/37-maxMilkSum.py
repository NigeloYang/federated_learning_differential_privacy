# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
'''' 农场最大产奶牛群
农场里有许多牛，每头牛都有一个产奶量值，范围在0到1000之间。为了方便管理，农场主将牛按照产奶量排列成二叉树的形式。现在，他想知道他从任意结点出发，一直到叶子结点走出这棵树（中间不回头），每次遇到一头牛就收获他的产量奶，他能收获的最大产奶量总和是多少？
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def __init__(self):
        self.maxmilk = 0
    
    def maxMilkSum(self, root: TreeNode) -> int:
        if not root:
            return root
        self.postorder(root)
        
        return self.maxmilk
    
    def postorder(self, root):
        if not root:
            return 0
        
        left = self.postorder(root.left)
        right = self.postorder(root.right)
        
        pathmilk = left + right + root.val
        self.maxmilk = max(self.maxmilk, pathmilk)
        return 0 if max(left, right) + root.val < 0 else max(left, right) + root.val
    
if __name__ == "__main__":
    print()
