# -*- coding: utf-8 -*-
# @Time    : 2023/9/8


from collections import deque


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def Serialize(self, root):
        if not root:
            return ""
        dq = deque([root])
        res = []
        while dq:
            node = dq.popleft()
            if node:
                res.append(str(node.val))
                dq.append(node.left)
                dq.append(node.right)
            else:
                res.append("None")
        return ','.join(res)
    
    def Deserialize(self, s):
        if not s:
            return
        
        datalist = s.split(',')
        root = TreeNode(int(datalist[0]))
        
        dq = deque([root])
        i = 1
        while dq:
            node = dq.popleft()
            if datalist[i] != 'None':
                node.left = TreeNode(int(datalist[i]))
                dq.append(node.left)
            i += 1
            if datalist[i] != 'None':
                node.right = TreeNode(int(datalist[i]))
                dq.append(node.right)
            i += 1
        return root


if __name__ == "__main__":
    print()
