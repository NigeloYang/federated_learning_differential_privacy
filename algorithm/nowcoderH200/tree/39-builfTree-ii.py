# -*- coding: utf-8 -*-
# @Time    : 2023/10/22

'''牛群的树形结构重建
农场里有一群牛，牛群的成员分布在山坡上，形成了一个二叉树结构。每头牛都有一个编号，表示它在牛群中的身份。现在农场主想要重建牛群的二叉树结构。
给定两个整数数组 inOrder 和 postOrder，其中 inOrder 是牛群二叉树的中序遍历，postOrder 是同一棵树的后序遍历，请构造二叉树并返回其根节点。

'''


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def buildTreeII(self, preOrder: List[int], inOrder: List[int]) -> TreeNode:
        if not preOrder or not inOrder:
            return None
    
        root = TreeNode(preOrder[0])
        ri = inOrder.index(preOrder[0])
        root.left = self.buildTreeII(preOrder[1:1 + ri], inOrder[:ri])
        root.right = self.buildTreeII(preOrder[ri + 1:], inOrder[ri + 1:])
        return root


if __name__ == "__main__":
    print()
