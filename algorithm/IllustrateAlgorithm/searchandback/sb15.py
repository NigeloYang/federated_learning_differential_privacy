# -*- coding: utf-8 -*-
# @Time    : 2023/6/20

'''剑指 Offer 55 - II. 平衡二叉树
输入一棵二叉树的根节点，判断该树是不是平衡二叉树。如果某二叉树中任意节点的左右子树的深度相差不超过1，那么它就是一棵平衡二叉树。
'''

from Tree import Tree


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        def dfs(root):
            if root is None:
                return 0
            return max(dfs(root.left), dfs(root.right)) + 1
        
        if root is None:
            return True
        
        return abs(dfs(root.left) - dfs(root.right)) <= 1 and self.isBalanced(root.left) and self.isBalanced(root.right)
    
    # 作者：Krahets


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
    print(Solution().isBalanced(tree.root))
    
    data2 = [1, 2, 2, 3, 3, None, None, 4, 4]
    tree2 = Tree()
    tree2.createTree(data2)
    tree2.inorder(tree2.root)
    print(Solution().isBalanced(tree2.root))
