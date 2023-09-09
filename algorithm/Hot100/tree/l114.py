# -*- coding: utf-8 -*-
# @Time    : 2023/7/25

'''二叉树展开为链表
给你二叉树的根结点 root ，请你将它展开为一个单链表：
展开后的单链表应该同样使用 TreeNode ，其中 right 子指针指向链表中下一个结点，而左子指针始终为 null 。
展开后的单链表应该与二叉树 先序遍历 顺序相同。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def flatten(self, root) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        res = []
        
        def preorder(node):
            if node:
                res.append(node.val)
                preorder(node.left)
                preorder(node.right)
        
        preorder(root)
        if not res:
            return None
        
        cur = root
        for i in res[1:]:
            cur.left = None
            cur.right = TreeNode(i)
            cur = cur.right
    
    def flatten2(self, root):
        while root:
            if root.left:  # 左子树存在的话才进行操作
                sub_left = root.left
                while sub_left.right:  # 左子树的右子树找到最深
                    sub_left = sub_left.right
                sub_left.right = root.right  # 将root的右子树挂到左子树的右子树的最深
                root.right = root.left  # 将root的左子树挂到右子树
                root.left = None  # 将root左子树清空
            root = root.right  # 继续下一个节点的操作


if __name__ == "__main__":
    pass
