# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

'''剑指 Offer 32 - II. 从上到下打印二叉树 II'''

from Tree import Tree

class Solution:
    def levelOrder(self, root):
        while not root:
            return []
        que = []
        res = []
        que.append(root)
        while len(que)>0:
            sub = []
            for i in range(len(que)):
                node = que.pop(0)
                sub.append(node.val)
                if node.left is not None:
                    que.append(node.left)
                if node.right is not None:
                    que.append(node.right)
            res.append(sub)
        return res

if __name__ == "__main__":
    data = [3, 9, 20, ' ', ' ', 15, 7]
    tree = Tree()
    tree.createTree(data)
    
    # 遍历
    # tree.breadth_travel()
    
    print(Solution().levelOrder(tree.root))
