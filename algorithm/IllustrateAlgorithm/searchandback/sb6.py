# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

'''剑指 Offer 32 - I. 从上到下打印二叉树'''
from Tree import Tree


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Solution:
    def levelOrder(self, root):
        if not root:
            return []
        queues = []
        res = []
        queues.append(root)
        while len(queues) > 0:
            node = queues.pop(0)
            res.append(node.val)
            if node.left:
                queues.append(node.left)
            if node.right:
                queues.append(node.right)
        return res


if __name__ == "__main__":
    data = [3, 9, 20, ' ', ' ', 15, 7]
    tree = Tree()
    tree.createTree(data)

    # 遍历
    # tree.breadth_travel()
    
    print(Solution().levelOrder(tree.root))
