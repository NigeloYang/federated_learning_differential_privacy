# -*- coding: utf-8 -*-
# @Time    : 2023/10/16

''' 牛群平均重量
在一个牧场中，有很多牛。为了方便管理，牧场主将牛的重量排列成一棵二叉树。现在牧场主想知道每层牛的平均重量。请按照从上到下的顺序，返回每层牛的平均重量。
'''

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        
class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        if not root:
            return root
        
        nodes = [root]
        ans = []
        while nodes:
            subweight = []
            for i in range(len(nodes)):
                node = nodes.pop(0)
                subweight.append(node.val)
                if node.left:
                    nodes.append(node.left)
                if node.right:
                    nodes.append(node.right)
            ans.append(sum(subweight) / len(subweight))
        return ans
    
if __name__ == "__main__":
    print()
