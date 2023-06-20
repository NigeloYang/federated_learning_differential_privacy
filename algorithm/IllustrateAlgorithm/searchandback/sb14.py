# -*- coding: utf-8 -*-
# @Time    : 2023/6/20

'''剑指 Offer 55 - I. 二叉树的深度
输入一棵二叉树的根节点，求该树的深度。从根节点到叶节点依次经过的节点（含根、叶节点）形成树的一条路径，最长路径的长度为树的深度。
'''

from Tree import Tree


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # if root is None:
        #     return 0
        #
        # que = []
        # que.append(root)
        # depth = 0
        # while len(que) > 0:
        #     depth += 1
        #     for _ in range(len(que)):
        #         node = que.pop(0)
        #         if node.left:
        #             que.append(node.left)
        #         if node.right:
        #             que.append(node.right)
        # return depth
        
        # Krahets 作品
        if not root: return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1


if __name__ == "__main__":
    data = [3, 9, 20, None, None, 15, 7]
    tree = Tree()
    tree.createTree(data)
    # 遍历
    # tree.preorder(tree.root)
    # print()
    # tree.inorder(tree.root)
    # print()
    # tree.preorder(tree.root)
    # print()
    print(Solution().maxDepth(tree.root))
