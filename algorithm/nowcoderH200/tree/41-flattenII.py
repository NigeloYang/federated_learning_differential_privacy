# -*- coding: utf-8 -*-
# @Time    : 2023/10/22

'''树形结构展开II
农场里有一群牛，牛群的成员分布在山坡上，形成了一个二叉树结构。每头牛都有一个编号，表示它在牛群中的身份。现在农场主想要将牛群的二叉树结构展开为一个链表。展开后的链表应该同样使用 TreeNode ，其中 left 子指针指向空，而 right 子指针指向链表中后一个结点。展开后的链表应该与二叉树中序遍历顺序相同。

'''
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def flattenII(self , root: TreeNode) -> TreeNode:
        if not root:
            return root
    
        nums = []
        self.inorder(root, nums)
    
        root = TreeNode(nums[0])
        temp = root
        for num in nums[1:]:
            tnode = TreeNode(num)
            temp.right = tnode
            temp = temp.right
    
        return root

    def inorder(self, root, res):
        if not root:
            return
    
        self.inorder(root.left, res)
        res.append(root.val)
        self.inorder(root.right, res)
    
if __name__ == "__main__":
    print()
