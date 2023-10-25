# -*- coding: utf-8 -*-
# @Time    : 2023/10/22

'''树的最长距离
农场里有一些牛，每头牛都有一个编号（1-n）。这些牛之间存在一种特殊的关系，我们可以把这些关系看作是一棵二叉树，牛的编号就是二叉树的节点。现在农场主想知道，这些牛之间的最长距离是多少。这里的距离定义为二叉树中任意两个节点之间路径的长度。
'''


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def __init__(self):
        self.ans = 0
    
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        self.maxDepth(root)
        
        return self.ans
    
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        self.ans = max(self.ans, left + right)
        return max(left, right) + 1


if __name__ == "__main__":
    print()
