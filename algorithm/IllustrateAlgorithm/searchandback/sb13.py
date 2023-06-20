# -*- coding: utf-8 -*-
# @Time    : 2023/6/20

'''剑指 Offer 54. 二叉搜索树的第 k 大节点
给定一棵二叉搜索树，请找出其中第 k 大的节点的值。'''

from Tree import Tree


class Solution:
    def kthLargest(self, root, k: int) -> int:
        if root is None:
            return None
        que = []
        
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            que.append(root.val)
            inorder(root.right)
        
        inorder(root)
        que.sort()
        return que[-k]


if __name__ == "__main__":
    data = [5, 3, 6, 2, 4, None, None, 1]
    tree = Tree()
    tree.createTree(data)
    # 遍历
    # tree.preorder(tree.root)
    # print()
    # tree.inorder(tree.root)
    # print()
    # tree.preorder(tree.root)
    # print()
    print(Solution().kthLargest(tree.root, 3))
