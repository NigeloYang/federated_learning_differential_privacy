# -*- coding: utf-8 -*-
# @Time    : 2023/10/16

'''牛群最小体重差
在一个牧场中，有很多牛。为了方便管理，牧场主将牛的体重排列成一棵二叉搜索树，假设所有牛的体重都不同。现在牧场主想知道牛群中任意两牛体重之间的最小差值。请你编写一个程序，给定一棵二叉搜索树的根节点 root，返回树中任意两不同节点值之间的最小差值，这个体重差至少是个正数。

'''


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        if not root:
            return root
        res = []
        self.inorder(root, res)
        ans = float('inf')
        for i in range(len(res) - 1):
            if res[i + 1] - res[i] < ans:
                ans = res[i + 1] - res[i]
        return ans
    
    def inorder(self, root, res):
        if not root:
            return
        
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)


class Solution2:
    def __init__(self):
        self.ans = float('inf')
    
    def getMinimumDifference(self, root: TreeNode) -> int:
        if not root:
            return root
        self.inorder(root, float('-inf'))
        
        return self.ans
    
    def inorder(self, root, pre):
        if not root:
            return
        
        self.inorder(root.left, self.ans, pre)
        self.ans = min(self.ans, root.val - pre)
        pre = root.val
        self.inorder(root.right, self.ans, pre)


if __name__ == "__main__":
    print()
