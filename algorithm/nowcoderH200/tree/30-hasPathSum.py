# -*- coding: utf-8 -*-
# @Time    : 2023/10/15

'''牛的奶量统计

农场里有许多牛，每头牛都有一个奶量值。农场主想知道，是否存在一组牛，它们的奶量和等于给定的目标值。每头牛都有一个唯一的编号，编号从1开始，表示它们的产奶量。为了方便管理，农场主将牛的编号按照二叉树的形式排列。

给定一棵表示牛编号的二叉树的根节点 root 和一个表示目标奶量和的整数 targetSum。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有牛的奶量和等于目标奶量和 targetSum。如果存在，返回 true ；否则，返回 false。
'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root:
            return False
        
        targetSum -= root.val
        if not root.left and not root.right and targetSum == 0:
            return True
        
        return self.hasPathSum(root.left, targetSum) or self.hasPathSum(root.right, targetSum)
    
if __name__ == "__main__":
    print()
