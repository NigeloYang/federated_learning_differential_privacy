# -*- coding: utf-8 -*-
# @Time    : 2023/10/16

'''牛奶产量总和
农场里有许多牛，每头牛经过一个二叉树状的草料区域后，进行产奶，每头牛各自走一条从根结点到叶子结点的路径。草料区域也就是二叉树的每个结点有一个产奶量值，范围在0到9之间，将一头牛经过的路径的所有数字拼接起来就是该头牛的最终产奶量，牧场主人想知道最终所有牛的产奶量之和。
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def sumNumbers(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        res = []
        self.preorder(root, res, 0)
        return sum(res)
    
    def preorder(self, root, res, num):
        if not root:
            return
        
        num = num * 10 + root.val
        if not root.left and not root.right:
            res.append(num)
        
        self.preorder(root.left, res, num)
        self.preorder(root.right, res, num)
        
if __name__ == "__main__":
    print()
