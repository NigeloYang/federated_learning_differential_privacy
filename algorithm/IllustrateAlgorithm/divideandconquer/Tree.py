# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Tree(object):
    def __init__(self):
        self.root = None
    
    # 创建树
    def createTree(self, data):
        for i in data:
            self.add(i)
    
    # 层序添加树节点
    def add(self, value):
        if self.root is None:
            self.root = TreeNode(value)
        else:
            queue = []
            queue.append(self.root)
            
            while len(queue) > 0:
                node = queue.pop(0)
                if node.left is None:
                    node.left = TreeNode(value)
                    return
                else:
                    queue.append(node.left)
                    
                if node.right is None:
                    node.right = TreeNode(value)
                    return
                else:
                    queue.append(node.right)
    
    """广度优先遍历"""
    def breadth_travel(self):
        
        if self.root is None:
            return
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            print(node.val, end=",")
            if node.left is not None:
                queue.append(node.left)
            if node.right is not None:
                queue.append(node.right)
    
    """先序遍历"""
    def preorder(self, root):
        
        if root is not None:
            print(root.val, end=",")
            self.preorder(root.left)
            self.preorder(root.right)
    
    """中序遍历"""
    def inorder(self, root):
        
        if root is not None:
            self.inorder(root.left)
            print(root.val, end=",")
            self.inorder(root.right)
    
    """后序遍历"""
    def postorder(self, root):
        if root is not None:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.val, end=",")
