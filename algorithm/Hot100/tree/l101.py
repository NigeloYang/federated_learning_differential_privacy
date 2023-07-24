# -*- coding: utf-8 -*-
# @Time    : 2023/7/22

'''对称二叉树
给你一个二叉树的根节点 root ， 检查它是否轴对称。'''


# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSymmetric(self, root) -> bool:
        if not root:
            return True
        
        return self.checknode(root.left, root.right)
    
    def checknode(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        
        return self.checknode(left.left, right.right) and self.checknode(left.rigth, right.left)
    
    def isSymmetric2(self, root) -> bool:
        if root is None or not (root.left or root.right):
            return True
        queue = [root.left, root.right]
        while len(queue) > 0:
            left = queue.pop(0)
            right = queue.pop(0)
            if not (left or right):
                continue
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            queue.append(left.left)
            queue.append(right.right)
            
            queue.append(left.right)
            queue.append(right.left)
        
        return True


if __name__ == "__main__":
    pass
