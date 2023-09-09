# -*- coding: utf-8 -*-
# @Time    : 2023/9/8

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
#
# 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
#
#
# @param preOrder int整型一维数组
# @param vinOrder int整型一维数组
# @return TreeNode类
#
class Solution:
    def reConstructBinaryTree(self, preOrder: List[int], vinOrder: List[int]) -> TreeNode:
        n = len(preOrder)
        m = len(vinOrder)
        
        if n == 0 or m == 0:
            return None
        
        root = TreeNode(preOrder[0])
        
        for i in range(len(vinOrder)):
            if preOrder[0] == vinOrder[i]:
                leftpre = preOrder[1:i + 1]
                righti = vinOrder[:i]
                root.left = self.reConstructBinaryTree(leftpre, righti)
                
                rightpre = preOrder[i + 1:]
                rightin = vinOrder[i + 1:]
                root.right = self.reConstructBinaryTree(rightpre, rightin)
        
        return root

if __name__ == "__main__":
    print()
