# -*- coding: utf-8 -*-
# @Time    : 2023/7/7

'''剑指 Offer 26. 树的子结构
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。'''


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B: return True
            if not A or A.val != B.val: return False
            return recur(A.left, B.left) and recur(A.right, B.right)
        
        if not A and not B:
            return True
        
        return recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B)


class Solutions:
    def isSubStructure(self, A: TreeNode, B: TreeNode) -> bool:
        def recur(A, B):
            if not B: return True
            if not A or A.val != B.val: return False
            return recur(A.left, B.left) and recur(A.right, B.right)
        
        return bool(A and B) and (recur(A, B) or self.isSubStructure(A.left, B) or self.isSubStructure(A.right, B))


if __name__ == "__main__":
    A = [3, 4, 5, 1, 2]
    B = [4, 1]
    print(Solution().isSubStructure(A, B))
    print(Solutions().isSubStructure(A, B))
