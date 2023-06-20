# -*- coding: utf-8 -*-
# @Time    : 2023/6/19
'''剑指 Offer 34. 二叉树中和为某一值的路径

给你二叉树的根节点 root 和一个整数目标和 targetSum ，找出所有从根节点到叶子节点路径总和等于给定目标和的路径。
叶子节点是指没有子节点的节点
'''

from Tree import Tree


class Solution:
    def pathSum(self, root, target):
        if root is None:
            return None
        def preorder(root, target, val):
            if not root:
                return
            print(root.val)
            subroad.append(root.val)
            val += root.val
            if val == target and not root.left and not root.right:
                road.append(list(subroad))
            preorder(root.left, target, val)
            preorder(root.right, target, val)
            subroad.pop(-1)
            
        road = []
        subroad = []
        val = 0
        preorder(root, target, val)
        return road



if __name__ == "__main__":
    data = [5, 4, 8, 11, None, 13, 4, 7, 2, None, None, 5, 1]
    tree = Tree()
    tree.createTree(data)
    # 遍历
    # tree.preorder(tree.root)
    # print()
    # tree.inorder(tree.root)
    # print()
    # tree.preorder(tree.root)
    # print()
    print(Solution().pathSum(tree.root, 22))
    
    data2 = [1, 2, 3]
    tree2 = Tree()
    tree2.createTree(data2)
    print(Solution().pathSum(tree2.root, 5))
