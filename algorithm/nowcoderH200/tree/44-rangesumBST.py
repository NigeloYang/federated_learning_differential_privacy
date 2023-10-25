# -*- coding: utf-8 -*-
# @Time    : 2023/10/22

''' 牛德最佳观赏区间

农场里有许多牛，每头牛都有一个观赏指数值，范围在1到10^4之间。农场主想知道，在给定的观赏指数区间内，哪些牛可以被安排到最佳观赏区。为了方便管理，农场主将牛的观赏指数按照二叉搜索树的形式排列。

给定一棵表示牛观赏指数的二叉搜索树的根节点 root 和一个整数区间 [low, high]，返回所有在区间内观赏指数的牛的观赏指数之和。
'''

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution2:
    def __init__(self):
        self.ans = 0
    
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return
        
        self.rangeSumBST(root.left, low, high)
        if root.val >= low and root.val <= high:
            self.ans += root.val
        self.rangeSumBST(root.right, low, high)
        
        return self.ans
    
class Solution:
    def rangeSumBST(self, root: TreeNode, low: int, high: int) -> int:
        if not root:
            return
        
        ans = []
        self.preorder(root, ans, low, high)
        
        return sum(ans)
    
    def preorder(self, root, ans, low, high):
        if not root:
            return
        
        if root.val >= low and root.val <= high:
            ans.append(root.val)
        
        self.preorder(root.left, ans, low, high)
        self.preorder(root.right, ans, low, high)
        
if __name__ == "__main__":
    print()
