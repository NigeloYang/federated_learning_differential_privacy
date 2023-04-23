'''将有序数组转换为二叉搜索树
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵 高度平衡 二叉搜索树。
高度平衡 二叉树是一棵满足「每个节点的左右两个子树的高度差的绝对值不超过 1 」的二叉树。

示例 1：
输入：nums = [-10,-3,0,5,9]
输出：[0,-3,9,-10,null,5]
解释：[0,-10,5,null,-3,null,9] 也将被视为正确答案：

示例 2：
输入：nums = [1,3]
输出：[3,1]
解释：[1,null,3] 和 [3,1] 都是高度平衡二叉搜索树。

提示：
1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums 按 严格递增 顺序排列
'''

from random import randint

import TreeNode as trees


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# 不会这个
class Solution:
    def sortedArrayToBST(self, nums):
        def helper(left, right):
            if left > right:
                return None
            
            # 选择任意一个中间位置数字作为根节点
            mid = (left + right + randint(0, 1)) // 2
            
            root = TreeNode(nums[mid])
            root.left = helper(left, mid - 1)
            root.right = helper(mid + 1, right)
            return root
        
        return helper(0, len(nums) - 1)

    def inorder(self, root):
        """先序遍历"""
        if root is not None:
            self.inorder(root.left)
            print(root.val, end=",")
            
            self.inorder(root.right)
            
if __name__ == "__main__":
    nums = [-10, -3, 0, 5, 9]
    nums2 = [1, 3]
    
    print('\n 验证二叉搜索树结果:', Solution().inorder(Solution().sortedArrayToBST(nums)))
    print('\n 验证二叉搜索树结果:', Solution().sortedArrayToBST(nums2))
    # tree1 = trees.tree()
    # tree2 = trees.tree()
    #
    # tree1.createTree(data)
    # tree1.breadth_travel()
    # print()
    # tree2.createTree(data2)
    # tree2.breadth_travel()
