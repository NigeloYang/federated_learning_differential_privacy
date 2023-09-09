# -*- coding: utf-8 -*-
# @Time    : 2023/7/25

'''路径总和 III
给定一个二叉树的根节点 root ，和一个整数 targetSum ，求该二叉树里节点值之和等于 targetSum 的 路径 的数目。
路径 不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def pathSum(self, root, targetSum: int) -> int:
        def dfs(node, sumlists):
            if not node:
                return 0
            
            sumlists = [sums + node.val for sums in sumlists]
            sumlists.append(node.val)
            
            count = 0
            for sums in sumlists:
                if sums == targetSum:
                    count += 1
            
            return count + dfs(node.left, sumlists) + dfs(node.right, sumlists)
        
        return dfs(root, [])

    def pathSum2(self, root: TreeNode, sum: int) -> int:
        prefixSumTree = {0: 1}
        self.count = 0

        prefixSum = 0
        self.dfs(root, sum, prefixSum, prefixSumTree)

        return self.count

    def dfs(self, root, sum, prefixSum, prefixSumTree):
        if not root:
            return 0
        prefixSum += root.val
        oldSum = prefixSum - sum
        if oldSum in prefixSumTree:
            self.count += prefixSumTree[oldSum]
        prefixSumTree[prefixSum] = prefixSumTree.get(prefixSum, 0) + 1

        self.dfs(root.left, sum, prefixSum, prefixSumTree)
        self.dfs(root.right, sum, prefixSum, prefixSumTree)

        '''一定要注意在递归回到上一层的时候要把当前层的prefixSum的个数-1，类似回溯，要把条件重置'''
        prefixSumTree[prefixSum] -= 1


if __name__ == "__main__":
    pass
