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
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        if not inorder or not postorder:
            return
        
        return self.dfsTree(inorder, 0, len(inorder) - 1, postorder, 0, len(postorder) - 1)
    
    
    def dfsTree(self, inorder, il, ir, postorder, pl, pr):
        if il > ir or pl > pr:
            return None
        if il == ir:
            return TreeNode(inorder[il])
        
        # 在中序数组寻找根节点
        ri = il
        while ri <= ir:
            if inorder[ri] == postorder[pr]:
                break
            ri += 1
        
        # 分别遍历左右子树
        root = TreeNode(postorder[pr])
        root.left = self.dfsTree(inorder, il, ri - 1, postorder, pl, pl + ri - il - 1)
        root.right = self.dfsTree(inorder, ri + 1, ir, postorder, pl + ri - il, pr - 1)
        return root

if __name__ == "__main__":
    print()
