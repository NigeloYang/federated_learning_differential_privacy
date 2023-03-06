# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Tree(object):
    def __init__(self):
        self.root = None
    
    # 创建二叉树
    def createTree(self, data):
        for i in data:
            self.add(i)
    
    # 添加树节点
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
    
    def breadth_travel(self):
        """广度优先遍历"""
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
        
    
    def preorder(self, root):
        """先序遍历"""
        if root is not None:
            print(root.val, end=",")
            self.preorder(root.left)
            self.preorder(root.right)
    
    def inorder(self, root):
        """中序遍历"""
        if root is not None:
            self.inorder(root.left)
            print(root.val, end=",")
            self.inorder(root.right)
    
    def postorder(self, root):
        """后序遍历"""
        if root is not None:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.val, end=",")


if __name__ == '__main__':
    # data = [i for i in range(10)]
    data = [3, 9, 20, ' ', ' ', 15, 7]
    tree = Tree()
    tree.createTree(data)
    
    tree.breadth_travel()
    print()
    tree.preorder(tree.root)
    print()
    tree.inorder(tree.root)
    print()
    tree.postorder(tree.root)
