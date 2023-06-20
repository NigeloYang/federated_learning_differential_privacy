# -*- coding: utf-8 -*-
# @Time    : 2023/6/20

'''剑指 Offer 36. 二叉搜索树与双向链表

输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。
为了让您更好地理解问题，以下面的二叉搜索树为例：
'''
from Tree import Tree


class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if root is None:
            return None
        que = []
        
        def inorder(root):
            if not root:
                return
            inorder(root.left)
            print(root.val)
            que.append(root)
            inorder(root.right)
        
        inorder(root)
        for i in range(len(que)):
            que[i].left = que[i - 1]
        for i in range(-1, len(que) - 1):
            que[i].right = que[i + 1]
        
        return que[0]
        
        # Krahets 指南
        # def dfs(cur):
        #     if not cur: return
        #     dfs(cur.left)  # 递归左子树
        #     if self.pre:  # 修改节点引用
        #         self.pre.right, cur.left = cur, self.pre
        #     else:  # 记录头节点
        #         self.head = cur
        #     self.pre = cur  # 保存 cur
        #     dfs(cur.right)  # 递归右子树
        #
        # if not root: return
        # self.pre = None
        # dfs(root)
        # self.head.left, self.pre.right = self.pre, self.head
        # return self.head
    
if __name__ == "__main__":
    data = [4, 2, 5, 1, 3, None, None]
    tree = Tree()
    tree.createTree(data)
    # 遍历
    # tree.preorder(tree.root)
    # print()
    # tree.inorder(tree.root)
    # print()
    # tree.preorder(tree.root)
    # print()
    print(Solution().treeToDoublyList(tree.root))
