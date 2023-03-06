'''验证二叉搜索树
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
有效 二叉搜索树定义如下：
节点的左子树只包含 小于 当前节点的数。
节点的右子树只包含 大于 当前节点的数。
所有左子树和右子树自身必须也是二叉搜索树。

示例 1：
输入：root = [2,1,3]
输出：true

示例 2：
输入：root = [5,1,4,null,null,3,6]
输出：false
解释：根节点的值是 5 ，但是右子节点的值是 4 。

提示：
树中节点数目范围在[1, 104] 内
-2^31 <= Node.val <= 2^31 - 1
'''

import TreeNode as trees


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# class Solution:
#     # 前序遍历
#     # def isValidBST(self, root, left=float('-inf'), right=float('inf')):
#     #     if root is None:
#     #         return True
#     #     val = root.val
#     #     return left < val < right and self.isValidBST(root.left, left, val) and self.isValidBST(root.right, val, right)
#
#
#     # 中序遍历
#     pre = float('-inf')
#     def isValidBST(self, root):
#         if root is None:
#             return True
#         return self.isValidBST(root.left) and  and self.isValidBST(root.right)


class Solution:
    def isValidBST(self, root):
        stack, val = [], float('-inf')
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= val:
                return False
            val = root.val
            root = root.right
        return True


if __name__ == "__main__":
    data = [2, 1, 3]
    data2 = [5, 1, 4, '', '', 3, 6]
    tree1 = trees.Tree()
    tree2 = trees.Tree()
    tree1.createTree(data)
    tree1.breadth_travel()
    print()
    tree2.createTree(data2)
    tree2.breadth_travel()
    print('\n 验证二叉搜索树结果:', Solution().isValidBST(tree1.root))
    print('\n 验证二叉搜索树结果:', Solution().isValidBST(tree2.root))
