# -*- coding: utf-8 -*-
# @Time    : 2023/6/19

'''剑指 Offer 32 - III. 从上到下打印二叉树 III

请实现一个函数按照之字形顺序打印二叉树，
即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。
'''

from Tree import Tree


class Solution:
    def levelOrder(self, root):
        while not root:
            return []
        que = []
        res = []
        que.append(root)
        while len(que) > 0:
            sub = []
            for _ in range(len(que)):
                node = que.pop(0)
                sub.append(node.val)
                if node.left is not None:
                    que.append(node.left)
                if node.right is not None:
                    que.append(node.right)
            if int(len(res) % 2) != 0:
                sub.reverse()
            res.append(sub)
        return res


if __name__ == "__main__":
    data = [3, 9, 20, None, None, 15, 7]
    tree = Tree()
    tree.createTree(data)
    # 遍历
    # tree.breadth_travel()
    print(Solution().levelOrder(tree.root))
    
    data2 = [1, 2, 3, 4, None, None, 5]
    tree2 = Tree()
    tree2.createTree(data2)
    print(Solution().levelOrder(tree2.root))
